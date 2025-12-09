from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.transform import Rotation as R
from options import MonodepthOptions

opt = MonodepthOptions().parse()

# Extract file base name from test_data_file (e.g., "test_files_sequence2.txt" -> "test_files_sequence2")
file_base = opt.test_data_file.split('.')[0] if opt.test_data_file else "test_files"

# Determine save root directory
save_root = getattr(opt, 'save_poses_root', None)
if save_root is None:
    save_root = os.path.join(os.path.dirname(__file__), "splits", opt.dataset)

# Construct paths based on new naming convention
gt_path = os.path.join(save_root, f"gt_poses_{file_base}.npz")
pred_path = os.path.join(save_root, f"pred_poses_{file_base}.npz")

# Handle optional model appendix if provided
eval_model_appendix = getattr(opt, 'eval_model_appendix', '')
if eval_model_appendix:
    # Try with model appendix if provided
    pred_path_with_appendix = os.path.join(save_root, f"pred_poses_{file_base}{eval_model_appendix}.npz")
    if os.path.exists(pred_path_with_appendix):
        pred_path = pred_path_with_appendix

print(f'Loading GT poses from: {gt_path}')
if not os.path.exists(gt_path):
    raise FileNotFoundError(f"GT poses file not found at {gt_path}")
gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

print(f'Loading predicted poses from: {pred_path}')
if not os.path.exists(pred_path):
    raise FileNotFoundError(f"Predicted poses file not found at {pred_path}")
pred_local_poses = np.load(pred_path, fix_imports=True, encoding='latin1')["data"]

# Handle StereoMIS dataset specific transformations
if opt.dataset == 'StereoMIS':
    # Conduct inverse for the gt_poses
    # SM gt npz saved in meter format
    gt_local_poses = np.linalg.inv(gt_local_poses)
    gt_local_poses[:, :3, 3] *= 1000


def dump(source_to_target_transformations):
    """Convert relative poses to absolute poses"""
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms


def compute_scale(gtruth, pred):
    """Compute optimal scaling factor between ground truth and prediction"""
    scale = np.sum(gtruth[:, :3, 3] * pred[:, :3, 3]) / np.sum(pred[:, :3, 3] ** 2)
    print(f'Scale factor: {scale}')
    return scale


def extract_xyz_rpy(transformation_matrices):
    """
    Extract x, y, z, roll, pitch, yaw from transformation matrices.
    
    Args:
        transformation_matrices: numpy array of shape (N, 4, 4)
        
    Returns:
        xyz_rpy: numpy array of shape (N, 6) where columns are [x, y, z, roll, pitch, yaw]
    """
    xyz_rpy = np.zeros((len(transformation_matrices), 6))
    
    for i, T in enumerate(transformation_matrices):
        # Extract translation (x, y, z)
        xyz_rpy[i, :3] = T[:3, 3]
        
        # Extract rotation matrix
        R_matrix = T[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        # Using XYZ convention
        r = R.from_matrix(R_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        
        xyz_rpy[i, 3:] = [roll, pitch, yaw]
    
    return xyz_rpy


# Optional: Limit number of frames for debugging
debug_only = getattr(opt, 'debug_only', False)
plot_num = getattr(opt, 'plot_num', None)
if debug_only and plot_num is not None:
    gt_local_poses = gt_local_poses[:plot_num]
    pred_local_poses = pred_local_poses[:plot_num]

# Convert relative poses to absolute poses
dump_gt = np.array(dump(gt_local_poses))
dump_pred = np.array(dump(pred_local_poses))

# Scale predictions to match ground truth scale
scale_pred = dump_pred.copy()
scale_pred[:, :3, 3] *= compute_scale(dump_gt, dump_pred)

# Extract trajectory points
num = gt_local_poses.shape[0]
points_pred = []
points_gt = []
origin = np.array([[0], [0], [0], [1]])

for i in range(num):
    point_pred = np.dot(scale_pred[i], origin)
    point_gt = np.dot(dump_gt[i], origin)
    points_pred.append(point_pred)
    points_gt.append(point_gt)

points_pred = np.array(points_pred)
points_gt = np.array(points_gt)

# Load confidence values if available
conf_values = None
if hasattr(opt, 'plot_conf') and opt.plot_conf:
    conf_path = pred_path.replace("pred_poses_", "pred_conf_")
    if os.path.exists(conf_path):
        conf_values = np.load(conf_path, fix_imports=True, encoding='latin1')["data"]
        print("Loaded confidence values for plotting")
    else:
        print(f"Confidence file not found at {conf_path}")

# Extract xyz_rpy if requested
gt_xyz_rpy = None
pred_xyz_rpy = None
if hasattr(opt, 'plot_xyz_rpy') and opt.plot_xyz_rpy:
    print("Generating combined 3D trajectory and XYZ-RPY plots...")
    gt_xyz_rpy = extract_xyz_rpy(dump_gt)
    pred_xyz_rpy = extract_xyz_rpy(scale_pred)

# Create figure
plot_xyz = gt_xyz_rpy is not None and pred_xyz_rpy is not None
plot_conf = conf_values is not None

fig = plt.figure(figsize=(20, 12))

# 3D trajectory subplot
if not plot_xyz and not plot_conf:
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
else:
    ax_3d = fig.add_subplot(2, 4, 1, projection='3d')

# Plot 3D trajectory
ax_3d.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], 
          label='GT', linestyle='-', c='blue', linewidth=1.6)
ax_3d.plot(points_pred[:, 0, 0], points_pred[:, 1, 0], points_pred[:, 2, 0], 
           label='Prediction', linestyle='-', c='red', linewidth=1.6)

ax_3d.set_xlabel("x [mm]")
ax_3d.set_ylabel("y [mm]")
ax_3d.set_zlabel("z [mm]")
ax_3d.set_title("3D Trajectory")
ax_3d.legend()

# Plot XYZ-RPY components if requested
if plot_xyz:
    labels = ['x [mm]', 'y [mm]', 'z [mm]', 'roll [deg]', 'pitch [deg]', 'yaw [deg]']
    gt_color = 'blue'
    pred_color = 'red'
    time_steps = np.arange(len(gt_xyz_rpy))
    subplot_positions = [2, 3, 4, 6, 7, 8]
    
    for i in range(6):
        ax = fig.add_subplot(2, 4, subplot_positions[i])
        
        # Plot GT and Prediction
        ax.plot(time_steps, gt_xyz_rpy[:, i], label='GT', color=gt_color, 
               linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, pred_xyz_rpy[:, i], label='Prediction', color=pred_color, 
               linewidth=1.5, linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i].split()[0].upper()} Component')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot confidence on secondary y-axis if available
        if plot_conf:
            print('Plotting confidence on the right y-axis.')
            ax_conf = ax.twinx()
            ax_conf.plot(time_steps[:-1], conf_values, label='Confidence', 
                        linestyle='-', color='g', linewidth=1.6, alpha=0.9)
            ax_conf.set_ylabel('Confidence')
            ax_conf.legend(loc='upper right')
    
    # Plot confidence histogram if available
    if plot_conf:
        outer_gs = fig.add_gridspec(2, 4, wspace=0.4, hspace=0.6)
        inner_gs = outer_gs[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # Main confidence plot
        ax_main = fig.add_subplot(inner_gs[0])
        print(f'Confidence min: {conf_values.min():.4f}, max: {conf_values.max():.4f}')
        ax_main.plot(time_steps[:-1], conf_values, label='Confidence', 
                    linestyle='-', c='g', linewidth=2.6)
        ax_main.set_xlabel('Time Step')
        ax_main.set_ylabel('Confidence')
        ax_main.set_title('Confidence Over Time')
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Histogram
        ax_hist = fig.add_subplot(inner_gs[1])
        ax_hist.hist(conf_values, bins=20, color='g', alpha=0.6)
        ax_hist.set_xlabel('Confidence Value')
        ax_hist.set_ylabel('Count')
        ax_hist.grid(True, alpha=0.3)

# Set overall title
conf_detail = ''
if conf_values is not None:
    conf_detail = f'raw conf minmax {conf_values.min():.4f}-{conf_values.max():.4f}'

# Extract sequence identifier for title
seq_id = file_base.replace('test_files_', '') if 'test_files' in file_base else file_base
model_id = eval_model_appendix if eval_model_appendix else 'default'
fig.suptitle(f'Pose Analysis - Seq {seq_id} - Model {model_id} - {conf_detail}', fontsize=16)

# Save the plot
output_filename = f'pose_vis_{file_base}{eval_model_appendix}.png'
plt.savefig(output_filename, dpi=600, bbox_inches='tight')
print(f'Combined pose analysis plot saved as: {output_filename}')

plt.show()
