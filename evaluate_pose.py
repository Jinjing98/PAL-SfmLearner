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


def compute_rpe_translation(gt_poses, pred_poses):
    """Compute Relative Pose Error for translation"""
    rpe_trans = []
    
    for i in range(len(gt_poses) - 1):
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        gt_trans = gt_rel[:3, 3]
        
        # Predicted relative pose
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        pred_trans = pred_rel[:3, 3]
        
        # Translation error
        trans_error = np.linalg.norm(gt_trans - pred_trans)
        rpe_trans.append(trans_error)
    
    return np.array(rpe_trans)


def construct_poses(positions, rotations):
    """Construct pose matrices from positions and rotations"""
    poses = []
    for pos, rot in zip(positions, rotations):
        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = rot
        poses.append(pose)
    return np.array(poses)


def compute_rpe_rotation(gt_poses, pred_poses):
    """Compute Relative Pose Error for rotation"""
    rpe_rot = []
    assert len(gt_poses) == len(pred_poses), "gt_poses and pred_poses must have the same length"
    assert len(gt_poses) > 1, "gt_poses and pred_poses must have at least 2 frames"
    
    for i in range(len(gt_poses) - 1):
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        gt_rot = gt_rel[:3, :3]
        
        # Predicted relative pose
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        pred_rot = pred_rel[:3, :3]
        
        # Rotation error (angle between rotation matrices)
        R = gt_rot @ np.linalg.inv(pred_rot)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        angle = np.arctan2(s, c)
        rpe_rot.append(angle)

    return np.array(rpe_rot)

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

    # Compute evaluation metrics
    print("\n-> Computing evaluation metrics...")
    
    def compute_metrics(gt_local_poses, pred_poses, track_length):
        ates = []
        res = []
        rpes_trans = []
        rpes_rot = []
        num_frames = gt_local_poses.shape[0]
        
        assert len(gt_local_poses) == len(pred_poses), f'len(gt_local_poses): {len(gt_local_poses)}, len(pred_poses): {len(pred_poses)}'

        for i in range(0, num_frames - 1):
            end_id = i + (track_length - 1)

            # Optional fix for short track length
            # Guarantee reasonable metrics with other variant track_length
            # Keep consistent stats when track_length is 5, while support reasonable stats when track_length is traj_len+1
            if end_id > len(pred_poses) and track_length != 5:
                break
            
            pred_abs_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_abs_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
            esti_abs_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
            gt_abs_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

            ates.append(compute_ate(gt_abs_xyzs, pred_abs_xyzs))
            res.append(compute_re(gt_abs_rs, esti_abs_rs))

            # RPE metrics
            # Construct gt_abs_poses from gt_abs_xyzs and gt_abs_rs
            if end_id > len(pred_poses):
                continue

            gt_abs_poses = construct_poses(gt_abs_xyzs, gt_abs_rs)
            esti_abs_poses = construct_poses(pred_abs_xyzs, esti_abs_rs)
            esti_snipt_scale = np.sum(gt_abs_poses[:, :3, 3] * esti_abs_poses[:, :3, 3]) / np.sum(esti_abs_poses[:, :3, 3] ** 2)
            esti_abs_poses_scale = esti_abs_poses.copy()
            esti_abs_poses_scale[:, :3, 3] = esti_abs_poses_scale[:, :3, 3] * esti_snipt_scale

            rpes_trans.append(compute_rpe_translation(gt_abs_poses, esti_abs_poses_scale))
            rpes_rot.append(compute_rpe_rotation(gt_abs_poses, esti_abs_poses))
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS GIVEN TRACK LENGTH = {}".format(track_length))
        print('NUM OF SNIPPETS IN COMPUTATION: {}'.format(len(ates)))
        print("="*60)
        
        float_digits = 4
        print(f"Absolute Trajectory Error (ATE):")
        print(f"   Mean: {np.mean(ates):.{float_digits}f}, Std: {np.std(ates):.{float_digits}f}")
        print(f"Rotation Error (RE):")
        print(f"   Mean: {np.mean(res):.{float_digits}f}, Std: {np.std(res):.{float_digits}f}")

        if len(rpes_trans) > 0:
            print(f"Relative Pose Error - Translation (RPE-T):")
            print(f"   Mean: {np.mean(rpes_trans):.{float_digits}f}, Std: {np.std(rpes_trans):.{float_digits}f}")
            # print(f"Relative Pose Error - Rotation (RPE-R (radian)):")
            # print(f"   Mean: {np.mean(rpes_rot):.{float_digits}f}, Std: {np.std(rpes_rot):.{float_digits}f}")
            print(f"Relative Pose Error - Rotation (RPE-R (deg)):")
            print(f"   Mean: {np.mean(rpes_rot) * 180 / np.pi:.{float_digits}f}, Std: {np.std(rpes_rot) * 180 / np.pi:.{float_digits}f}")

            # Latex print format
            print(f"Method & {np.mean(ates):.{float_digits}f}$\pm{np.std(ates):.{float_digits}f}$ & {np.mean(res):.{float_digits}f}$\pm{np.std(res):.{float_digits}f}$ & {np.mean(rpes_trans):.{float_digits}f}$\pm{np.std(rpes_trans):.{float_digits}f}$ & {np.mean(rpes_rot):.{float_digits}f}$\pm{np.std(rpes_rot):.{float_digits}f}$ \\\\")
        else:
            # Latex print format (without RPE if not computed)
            print(f"Method & {np.mean(ates):.{float_digits}f}$\pm{np.std(ates):.{float_digits}f}$ & {np.mean(res):.{float_digits}f}$\pm{np.std(res):.{float_digits}f}$ \\\\")

        print("="*60)

        return ates, res, rpes_trans, rpes_rot

    # Support multiple track lengths
    track_lengths = getattr(opt, 'track_lengths', [5])  # Default to [5] if not specified
    if track_lengths is None:
        track_lengths = [5]
    if isinstance(track_lengths, int):
        track_lengths = [track_lengths]  # Convert single int to list
    
    metrics_dict = {}
    for track_length in track_lengths:
        metrics_dict[track_length] = compute_metrics(gt_local_poses, pred_poses, track_length=track_length)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
