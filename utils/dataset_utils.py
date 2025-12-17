from __future__ import absolute_import, division, print_function

import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_vec_to_mat(pose_vec):
    """
    Convert a 7D vector (xyz + quaternion) to a 4x4 transformation matrix.
    pose_vec: array-like, shape (7,) -> [x, y, z, qx, qy, qz, qw]
    """
    pose_vec = np.asarray(pose_vec)
    assert pose_vec.shape[-1] == 7, f"{pose_vec} Expected input shape (..., 7)"

    trans = pose_vec[:3]
    quat = pose_vec[3:]

    rot = R.from_quat(quat).as_matrix()  # 3x3 rotation matrix

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def read_freiburg_scipy(path: str, ret_stamps=False, no_stamp=False, trans_scale=1000):
    """
    Read trajectory file in Freiburg format.
    """
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list_data = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                    len(line) > 0 and line[0] != "#"]

    if no_stamp:
        trans_np = np.asarray([l[0:3] for l in list_data if len(l) > 0], dtype=float)
        quat_np = np.asarray([l[3:] for l in list_data if len(l) > 0], dtype=float)
        trans_quat_np = np.hstack((trans_np, quat_np))
        trans_quat_np[:, :3] *= trans_scale
        pose_matrix = np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
    else:
        time_stamp = [l[0] for l in list_data if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp]) * 100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        trans_np = np.asarray([l[1:4] for l in list_data if len(l) > 0], dtype=float)
        quat_np = np.asarray([l[4:] for l in list_data if len(l) > 0], dtype=float)
        trans_quat_np = np.hstack((trans_np, quat_np))
        trans_quat_np[:, :3] *= trans_scale
        pose_matrix = np.asarray([pose_vec_to_mat(l) for l in trans_quat_np if len(l) > 0], dtype=float)
        if ret_stamps:
            return pose_matrix, time_stamp
    return pose_matrix


def map_traj_search_SCARED_DEPTH(folder, traj_data_root):
    """
    Map folder path to trajectory search path.
    """
    parts = folder.split('/')
    if len(parts) == 2:
        sequence = int(parts[0].replace('dataset', ''))
        keyframe = int(parts[1].replace('keyframe', ''))

        split_dir_ori = 'test_' if sequence in range(8, 10) else ''
        trajfolder = f'{traj_data_root}{split_dir_ori}dataset_{sequence}/keyframe_{keyframe}'
        return trajfolder
    else:
        raise ValueError(f"Invalid folder format: {folder}")

def map_traj_search(folder, traj_data_root):
    """
    Map folder path to trajectory search path.
    folder expected format: "dataset_X/keyframe_Y"
    """
    parts = folder.split('/')
    if len(parts) == 2:
        sequence = int(parts[0].replace('dataset', ''))
        keyframe = int(parts[1].replace('keyframe', ''))

        split_dir_ori = 'testing' if sequence in range(8, 10) else 'training'
        sequence_ori = sequence
        keyframe_ori = keyframe - 1 if sequence in range(8, 10) else keyframe

        trajfolder = f'{traj_data_root}/{split_dir_ori}/dataset_{sequence_ori}/keyframe_{keyframe_ori}/'
        return trajfolder
    else:
        raise ValueError(f"Invalid folder format: {folder}")


def get_gt_poses(filenames, traj_data_root, trans_scale=1000):
    """
    Load ground truth poses for all folders in filenames.
    Returns a dictionary mapping folder to trajectory data.
    """
    trajs_dict = {}
    unique_folders = set()
    for filename in filenames:
        line = filename.split()
        if len(line) >= 1:
            folder = line[0]
            unique_folders.add(folder)
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    print(f"Loading trajectories for {len(unique_folders)} unique folders...")
    for folder in unique_folders:
        traj_full_folder = map_traj_search(folder, traj_data_root)
        traj_path = f'{traj_full_folder}/groundtruth.txt'

        if os.path.exists(traj_path):
            traj = read_freiburg_scipy(
                traj_path,
                ret_stamps=False,
                no_stamp=False,
                trans_scale=trans_scale
            )
            trajs_dict[folder] = traj
        else:
            raise FileNotFoundError(f'Trajectory file {traj_path} does not exist for {folder}')

    print(f"Successfully loaded {len([v for v in trajs_dict.values() if v is not None])} trajectories")
    return trajs_dict


def get_poses_for_frames(trajs_dict, folder, frame_indices, offset):
    """
    Get ground truth poses for specific frame indices in a folder.
    offset: should be -1 for SCARED. Offset to add to frame indices to get trajectory indices.
    """
    poses = []
    for frame_idx in frame_indices:
        traj_idx = frame_idx + offset
        if 0 <= traj_idx < len(trajs_dict[folder]):
            pose = trajs_dict[folder][traj_idx].astype(np.float32)
            poses.append(pose)
        else:
            raise ValueError(f"Frame index {frame_idx} out of range for trajectory in {folder}. Using identity pose.")
    return np.array(poses).squeeze()

