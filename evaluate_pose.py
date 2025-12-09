from __future__ import absolute_import, division, print_function

import os
import torch
import networks
import numpy as np

from torch.utils.data import DataLoader
from utils import (
    readlines,
    transformation_from_parameters,
    transformation_from_parameters_6D,
    transformation_from_parameters_9D
)
from options import MonodepthOptions
from datasets import SCAREDRAWDataset

from tqdm import tqdm

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]

def online_gen_gt_poses(opt, dataloader, gt_path):
    print(f"Computing gt poses for {opt.test_data_file}...")
    gt_local_poses = []
    for i, inputs in enumerate(dataloader):
        if i == 0:
            gt_local_poses.append(inputs[("gt_c2w_poses", 0)].squeeze().cpu().numpy())
        gt_local_pose = inputs[("gt_c2w_poses", 1)].squeeze()
        gt_local_poses.append(gt_local_pose.cpu().numpy())
    gt_rel_poses = [np.linalg.inv(gt_local_poses[i+1]) @ gt_local_poses[i] for i in range(len(gt_local_poses) - 1)]
    gt_local_poses = np.array(gt_rel_poses)
    print(f"Loaded {len(gt_local_poses)} rel gt poses")
    gt_local_poses_meter = gt_local_poses.copy()
    gt_local_poses_meter[:, :3, 3] = gt_local_poses_meter[:, :3, 3] #/ 1000
    np.savez_compressed(gt_path, data=np.array(gt_local_poses_meter))
    print(f"Saved gt poses to {gt_path}") 

def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "endovis",
                     opt.test_data_file))

    dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    # Initialize pose decoder with the same parameters used during training
    pose_decoder = networks.PoseDecoder(
        pose_encoder.num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2,
        trans_scale_factor=getattr(opt, 'trans_scale_factor', 0.001),
        rot_scale_factor=getattr(opt, 'rot_scale_factor', 0.001),
        rot_representation=getattr(opt, 'rot_representation', 'angle_axis'),
        explicit_bias_init_6d9d=getattr(opt, 'explicit_bias_init_6d9d', False)
    )
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Computing pose predictions"):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

            features = [pose_encoder(all_color_aug)]
            rot_output, translation = pose_decoder(features)

            # Use appropriate transformation function based on rotation representation
            rot_representation = getattr(opt, 'rot_representation', 'angle_axis')
            if rot_representation == "angle_axis":
                pose_matrix = transformation_from_parameters(
                    rot_output[:, 0], translation[:, 0])
            elif rot_representation == "6D":
                pose_matrix = transformation_from_parameters_6D(
                    rot_output[:, 0], translation[:, 0])
            elif rot_representation == "9D":
                pose_matrix = transformation_from_parameters_9D(
                    rot_output[:, 0], translation[:, 0])
            else:
                raise ValueError(f"Unsupported rotation representation: {rot_representation}")

            pred_poses.append(pose_matrix.cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    # Save pose predictions locally
    pred_poses_root = getattr(opt, 'save_poses_root', None)
    if pred_poses_root is None:
        pred_poses_root = os.path.join(os.path.dirname(__file__), "splits", opt.dataset)
    os.makedirs(pred_poses_root, exist_ok=True)
    
    # Extract sequence number from test_data_file (e.g., "test_files_sequence2.txt" -> "sequence2")
    # seq_suffix = ""
    # if "sequence" in opt.test_data_file:
    #     # Extract sequence number (e.g., "sequence2" from "test_files_sequence2.txt")
    #     seq_suffix = "_sq" + opt.test_data_file.split("sequence")[1].split(".")[0]
    
    pred_poses_path = os.path.join(pred_poses_root, f"pred_poses_{opt.test_data_file.split('.')[0]}.npz")
    np.savez_compressed(pred_poses_path, data=pred_poses)
    print(f"-> Saved {len(pred_poses)} pose predictions to {pred_poses_path}")

    assert os.path.exists(os.path.join(os.path.dirname(__file__), "splits", opt.dataset))
    if opt.test_data_file in ["test_files_sequence2.txt", "test_files_sequence1.txt"]:
        gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_sq{}.npz".format(opt.test_data_file.split('.')[0][-1]))
        assert os.path.exists(gt_path), f"GT path {gt_path} does not exist"
    else:
        # online gen: we verified being the same as the gt_poses_sq{}.npz
        gt_path = os.path.join(os.path.dirname(__file__), "splits", opt.dataset, "gt_poses_{}.npz".format(opt.test_data_file.split('.')[0]))
        online_gen_gt_poses(opt, dataloader, gt_path)


    gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    print(f"Loaded {len(gt_local_poses)} rel gt poses")

    ates = []
    res = []
    num_frames = gt_local_poses.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        res.append(compute_re(local_rs, gt_rs))

    print("\n   Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))
    print("\n   Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res), np.std(res)))


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
