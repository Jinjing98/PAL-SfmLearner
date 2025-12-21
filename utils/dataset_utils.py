from __future__ import absolute_import, division, print_function

import os
import numpy as np
import yaml
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
        # traj_full_folder = map_traj_search_SCARED_DEPTH(folder, traj_data_root)
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


def get_k_for_frames(gt_Ks_dict_registered, folder):
    """
    Get ground truth camera intrinsics K matrix for a folder.
    Returns K matrix (3x3 numpy array).
    
    Args:
        gt_Ks_dict_registered: Dictionary mapping folder to K matrix (3x3)
        folder: Folder key (e.g., "dataset_x/keyframe_x")
    
    Returns:
        K matrix (3x3 numpy array)
    """
    if folder in gt_Ks_dict_registered:
        return gt_Ks_dict_registered[folder].astype(np.float32)
    else:
        raise KeyError(f"Folder {folder} not found in gt_Ks_dict_registered")


def opencv_matrix_constructor(loader, node):
    """Custom YAML constructor for !!opencv-matrix tag."""
    # Load the mapping (dict) from the YAML node
    mapping = loader.construct_mapping(node, deep=True)
    return mapping


def parse_opencv_matrix(yaml_data):
    """
    Parse OpenCV matrix from YAML data.
    Handles !!opencv-matrix format:
        M1: !!opencv-matrix
           rows: 3
           cols: 3
           dt: f
           data: [ 1.03530811e+03, 0., 5.96955017e+02, ...]
    
    Returns numpy array of shape (rows, cols).
    """
    if isinstance(yaml_data, dict):
        rows = yaml_data.get('rows', 3)
        cols = yaml_data.get('cols', 3)
        data = yaml_data.get('data', [])
        
        # Handle case where data might be a string or list
        if isinstance(data, str):
            # Parse string representation of list
            import ast
            data = ast.literal_eval(data)
        
        # Convert data to numpy array and reshape
        data_array = np.array(data, dtype=np.float32)
        if data_array.size != rows * cols:
            raise ValueError(f"Data size {data_array.size} does not match rows*cols {rows*cols}")
        matrix = data_array.reshape(rows, cols)
        return matrix
    else:
        raise ValueError(f"Expected dict for OpenCV matrix, got {type(yaml_data)}")


def get_gt_Ks(filenames, traj_data_root):
    """
    Load ground truth camera intrinsics (K matrices) for all folders in filenames.
    Returns a dictionary mapping folder to K matrix (3x3 numpy array).
    
    K matrices are loaded from {traj_full_folder}/endoscope_calibration.yaml
    and extracted from the M1 field (OpenCV matrix format).
    """
    Ks_dict = {}
    unique_folders = set()
    for filename in filenames:
        line = filename.split()
        if len(line) >= 1:
            folder = line[0]
            unique_folders.add(folder)
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    print(f"Loading camera intrinsics for {len(unique_folders)} unique folders...")
    for folder in unique_folders:
        traj_full_folder = map_traj_search_SCARED_DEPTH(folder, traj_data_root)
        K_path = f'{traj_full_folder}/endoscope_calibration.yaml'

        if os.path.exists(K_path):
            with open(K_path, 'r') as f:
                content = f.read()
                # Skip OpenCV-specific %YAML:1.0 header if present
                lines = content.split('\n')
                if lines[0].startswith('%YAML'):
                    content = '\n'.join(lines[1:])
                # Use FullLoader with custom constructor for !!opencv-matrix tag
                loader = yaml.FullLoader
                # Add custom constructor for OpenCV matrix tag
                yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor, loader)
                yaml_data = yaml.load(content, Loader=loader)
            
            # Extract M1 matrix (OpenCV format)
            if 'M1' in yaml_data:
                M1_data = yaml_data['M1']
                K = parse_opencv_matrix(M1_data)
                Ks_dict[folder] = K.astype(np.float32)
            else:
                raise KeyError(f'M1 not found in calibration file {K_path}')
        else:
            raise FileNotFoundError(f'Calibration file {K_path} does not exist for {folder}')

    print(f"Successfully loaded {len([v for v in Ks_dict.values() if v is not None])} camera intrinsics")
    return Ks_dict

