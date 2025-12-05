from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import torch
import os


def visualize_disp(disp):
    """Visualize disparity map
    Args:
        disp: (H, W) or (1, H, W) disparity tensor
    Returns:
        colormapped_im: (3, H, W) uint8 image
    """
    disp = disp.squeeze()
    x = disp.cpu().detach().numpy()
    vmax = np.percentile(x, 95)

    normalizer = mpl.colors.Normalize(vmin=x.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = np.transpose(colormapped_im, (2, 0, 1))
    return colormapped_im


def visualize_depth(depth):
    """Visualize depth map
    Args:
        depth: (H, W) or (1, H, W) depth tensor
    Returns:
        colormapped_im: (3, H, W) uint8 image
    """
    depth = depth.squeeze()
    x = depth.cpu().detach().numpy()
    vmax = np.percentile(x, 95)

    normalizer = mpl.colors.Normalize(vmin=x.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = np.transpose(colormapped_im, (2, 0, 1))
    return colormapped_im


def visualize_alpha_map(alpha_map, cmap='viridis'):
    """Visualize PABA alpha map (correction strength)
    Args:
        alpha_map: (H, W) or (1, H, W) alpha tensor
        cmap: colormap name ('viridis', 'plasma', 'inferno', 'magma')
    Returns:
        colormapped_im: (3, H, W) uint8 image
    """
    alpha_map = alpha_map.squeeze()
    x = alpha_map.cpu().detach().numpy()
    
    # Normalize to [0, 1] range for visualization
    vmin = x.min()
    vmax = x.max()
    if vmax > vmin:
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
    
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = np.transpose(colormapped_im, (2, 0, 1))
    return colormapped_im


def compute_depth_error_map(gt_depth, pred_depth, min_depth=1e-3, max_depth=150, apply_scaling=True):
    """Compute per-pixel depth error map
    
    Args:
        gt_depth: (H, W) numpy array of ground truth depth
        pred_depth: (H, W) numpy array of predicted depth
        min_depth: minimum valid depth
        max_depth: maximum valid depth
        apply_scaling: if True, apply median scaling
    
    Returns:
        error_map: (H, W) numpy array of absolute relative error
        mask: (H, W) boolean mask of valid pixels
    """
    # Create mask based on depth cutoff
    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    
    if not mask.any():
        return np.zeros_like(gt_depth), mask
    
    # Apply mask
    pred_depth_masked = pred_depth[mask]
    gt_depth_masked = gt_depth[mask]
    
    if apply_scaling:
        # Apply median scaling (mono evaluation)
        ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
        pred_depth_masked = pred_depth_masked * ratio
    
    # Clip to valid range
    pred_depth_masked = np.clip(pred_depth_masked, min_depth, max_depth)
    
    # Compute absolute relative error per pixel
    error_masked = np.abs(gt_depth_masked - pred_depth_masked) / gt_depth_masked
    
    # Create full error map
    error_map = np.zeros_like(gt_depth)
    error_map[mask] = error_masked
    
    return error_map, mask


def visualize_depth_err(gt_depth, pred_depth, min_depth=1e-3, max_depth=150, apply_scaling=True, cmap='viridis'):
    """Visualize depth error map
    
    Args:
        gt_depth: (H, W) or (1, H, W) ground truth depth tensor
        pred_depth: (H, W) or (1, H, W) predicted depth tensor
        min_depth: minimum valid depth
        max_depth: maximum valid depth
        apply_scaling: if True, apply median scaling
        cmap: colormap name ('viridis' or 'cividis')
    
    Returns:
        colormapped_im: (3, H, W) uint8 image
    """
    # Convert to numpy and handle shapes
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.squeeze().cpu().numpy()
    else:
        gt_depth = np.squeeze(gt_depth)
    
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.squeeze().cpu().numpy()
    else:
        pred_depth = np.squeeze(pred_depth)
    
    # Resize pred_depth to match gt_depth if needed
    if pred_depth.shape != gt_depth.shape:
        H_gt, W_gt = gt_depth.shape
        pred_depth = cv2.resize(pred_depth, (W_gt, H_gt))
    
    # Compute error map
    error_map, mask = compute_depth_error_map(gt_depth, pred_depth, min_depth, max_depth, apply_scaling)
    
    # Normalize error map (use 95th percentile to avoid outliers)
    if mask.any():
        vmax = np.percentile(error_map[mask], 95)
        vmin = error_map[mask].min()
    else:
        vmax = error_map.max()
        vmin = error_map.min()
    
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(error_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = np.transpose(colormapped_im, (2, 0, 1))
    
    return colormapped_im


def _get_image_from_outputs(outputs, key, sample_idx=0):
    """Extract and select sample from outputs dictionary"""
    if key in outputs:
        img = outputs[key]
    elif isinstance(key, tuple) and key in outputs:
        img = outputs[key]
    else:
        # Try to find matching key
        for k in outputs.keys():
            if isinstance(k, tuple) and len(k) >= len(key) and k[:len(key)] == key:
                img = outputs[k]
                break
            elif isinstance(k, str) and k == key:
                img = outputs[k]
                break
        else:
            return None
    
    # Select sample from batch if needed
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 4:  # (B, C, H, W)
            img = img[sample_idx]
        return img
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
            img = img[sample_idx]
        return torch.from_numpy(img) if not isinstance(img, torch.Tensor) else img
    return img


def _tensor_to_rgb(img_tensor, img_height=192):
    """Convert tensor to RGB numpy array and resize
    
    Handles images that may have values outside [0, 1] range (e.g., PABA outputs).
    Clamps to [0, 1] first, then scales to [0, 255] for proper visualization.
    """
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().detach().numpy()
    else:
        img_np = np.array(img_tensor)
    
    # Handle shape: (C, H, W) -> (H, W, C)
    if len(img_np.shape) == 3 and img_np.shape[0] <= 4:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Normalize to [0, 255]
    # First clamp to [0, 1] if values are outside this range (important for PABA outputs)
    if img_np.max() > 1.0 or img_np.min() < 0.0:
        img_np = np.clip(img_np, 0, 1)
    
    # Scale to [0, 255] and convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    
    # Resize
    H, W = img_np.shape[:2]
    aspect_ratio = W / H
    img_width = int(img_height * aspect_ratio)
    img_np = cv2.resize(img_np, (img_width, img_height))
    
    return img_np


def _compute_color_error_map(target_img, source_img, img_height):
    """Compute error map between two color images using HOT colormap.
    
    Args:
        target_img: Target image tensor (C, H, W) or (H, W, C) numpy array
        source_img: Source image tensor (C, H, W) or (H, W, C) numpy array
        img_height: Target height for output
    
    Returns:
        error_vis: (H, W, 3) RGB error map with HOT colormap
    """
    # Convert to numpy if needed
    if isinstance(target_img, torch.Tensor):
        target_np = target_img.cpu().detach().numpy()
        if len(target_np.shape) == 3 and target_np.shape[0] in [1, 3]:
            target_np = np.transpose(target_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    else:
        target_np = np.array(target_img)
    
    if isinstance(source_img, torch.Tensor):
        source_np = source_img.cpu().detach().numpy()
        if len(source_np.shape) == 3 and source_np.shape[0] in [1, 3]:
            source_np = np.transpose(source_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    else:
        source_np = np.array(source_img)
    
    # Ensure RGB format
    if len(target_np.shape) == 2:
        target_np = cv2.cvtColor(target_np, cv2.COLOR_GRAY2RGB)
    elif len(target_np.shape) == 3 and target_np.shape[2] == 1:
        target_np = cv2.cvtColor(target_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if len(source_np.shape) == 2:
        source_np = cv2.cvtColor(source_np, cv2.COLOR_GRAY2RGB)
    elif len(source_np.shape) == 3 and source_np.shape[2] == 1:
        source_np = cv2.cvtColor(source_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Resize to same size if needed
    H, W = target_np.shape[:2]
    if source_np.shape[:2] != (H, W):
        source_np = cv2.resize(source_np, (W, H))
    
    # Convert to float [0, 1] for error computation
    target_float = target_np.astype(np.float32) / 255.0
    source_float = source_np.astype(np.float32) / 255.0
    
    # Compute absolute error per pixel per channel
    error = np.abs(target_float - source_float)  # (H, W, 3)
    
    # Average across RGB channels
    error = error.mean(axis=2)  # (H, W) in [0, 1]
    
    # Normalize using 95th percentile to handle outliers
    max_error = np.percentile(error, 95)
    if max_error > 0:
        error_norm = np.clip(error / max_error, 0, 1) * 255
        error_norm = error_norm.astype(np.uint8)
    else:
        error_norm = np.zeros_like(error, dtype=np.uint8)
    
    # Apply HOT colormap (expects uint8 single channel, returns BGR 3-channel)
    error_colored = cv2.applyColorMap(error_norm, cv2.COLORMAP_HOT)
    
    # Convert BGR to RGB
    error_colored = cv2.cvtColor(error_colored, cv2.COLOR_BGR2RGB)
    
    # Resize to target height
    H_orig, W_orig = error_colored.shape[:2]
    aspect_ratio = W_orig / H_orig
    img_width = int(img_height * aspect_ratio)
    error_colored = cv2.resize(error_colored, (img_width, img_height))
    
    return error_colored


def _process_image_key(key, merged_dict, sample_idx, img_height):
    """Helper function to process a single image key and return visualization"""
    # Handle None placeholder
    if key is None:
        placeholder_width = int(img_height * 1.5)  # Default aspect ratio
        white_img = np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255
        return white_img, ""
    
    # Special handling for depth_err
    if key == "depth_err":
        gt_key = ("depth_gt", 0, 0)
        pred_key = ("depth", 0, 0)
        
        if gt_key not in merged_dict or pred_key not in merged_dict:
            print(f"Warning: 'depth_err' requires {gt_key} and {pred_key} in merged_dict, using placeholder...")
            placeholder_width = int(img_height * 1.5)
            return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
        
        gt_depth = _get_image_from_outputs(merged_dict, gt_key, sample_idx)
        pred_depth = _get_image_from_outputs(merged_dict, pred_key, sample_idx)
        
        if gt_depth is None or pred_depth is None:
            print(f"Warning: Could not extract depth for depth_err, using placeholder...")
            placeholder_width = int(img_height * 1.5)
            return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
        
        # Use existing visualization function
        vis_img = visualize_depth_err(gt_depth, pred_depth, cmap='viridis')
        vis_img = np.transpose(vis_img, (1, 2, 0))
        H, W = vis_img.shape[:2]
        aspect_ratio = W / H
        img_width = int(img_height * aspect_ratio)
        vis_img = cv2.resize(vis_img, (img_width, img_height))
        return vis_img, "depth_err"
    
    # Special handling for color_warp_err_before and color_warp_err_after
    if key == "color_warp_err_before" or key == "color_warp_err_after":
        # Target is always ("color_aug", 0, 0)
        target_key = ("color_aug", 0, 0)
        
        if target_key not in merged_dict:
            print(f"Warning: '{key}' requires {target_key} in merged_dict, using placeholder...")
            placeholder_width = int(img_height * 1.5)
            return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
        
        # Find the corresponding warped/aligned image
        # Search merged_dict for matching keys
        source_key = None
        if key == "color_warp_err_before":
            # Look for ("color_warp", 0, frame_id) - search all available keys
            for k in merged_dict.keys():
                if isinstance(k, tuple) and len(k) == 3:
                    if k[0] == "color_warp" and k[1] == 0:
                        source_key = k
                        break
        else:  # color_warp_err_after
            # Look for ("paba_color_warp", 0, frame_id) - search all available keys
            for k in merged_dict.keys():
                if isinstance(k, tuple) and len(k) == 3:
                    if k[0] == "paba_color_warp" and k[1] == 0:
                        source_key = k
                        break
        
        if source_key is None:
            print(f"Warning: Could not find corresponding warped/aligned image for '{key}', using placeholder...")
            placeholder_width = int(img_height * 1.5)
            return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
        
        target_img = _get_image_from_outputs(merged_dict, target_key, sample_idx)
        source_img = _get_image_from_outputs(merged_dict, source_key, sample_idx)
        
        if target_img is None or source_img is None:
            print(f"Warning: Could not extract images for '{key}', using placeholder...")
            placeholder_width = int(img_height * 1.5)
            return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
        
        # Compute error map
        error_vis = _compute_color_error_map(target_img, source_img, img_height)
        return error_vis, key
    
    # Extract image from merged_dict
    img = _get_image_from_outputs(merged_dict, key, sample_idx)
    if img is None:
        print(f"Warning: Key {key} not found in merged_dict, using placeholder...")
        placeholder_width = int(img_height * 1.5)
        return np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255, ""
    
    # Determine visualization type based on key
    key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
    key_lower = key_str.lower()
    
    # Use appropriate visualization function
    if "disp" in key_lower:
        vis_img = visualize_disp(img)
        vis_img = np.transpose(vis_img, (1, 2, 0))
        H, W = vis_img.shape[:2]
        aspect_ratio = W / H
        img_width = int(img_height * aspect_ratio)
        vis_img = cv2.resize(vis_img, (img_width, img_height))
        return vis_img, key_str
        
    elif "depth" in key_lower and "err" not in key_lower:
        vis_img = visualize_depth(img)
        vis_img = np.transpose(vis_img, (1, 2, 0))
        H, W = vis_img.shape[:2]
        aspect_ratio = W / H
        img_width = int(img_height * aspect_ratio)
        vis_img = cv2.resize(vis_img, (img_width, img_height))
        return vis_img, key_str
        
    elif "paba_alpha" in key_lower or ("alpha" in key_lower and "paba" in key_lower):
        vis_img = visualize_alpha_map(img, cmap='viridis')
        vis_img = np.transpose(vis_img, (1, 2, 0))
        H, W = vis_img.shape[:2]
        aspect_ratio = W / H
        img_width = int(img_height * aspect_ratio)
        vis_img = cv2.resize(vis_img, (img_width, img_height))
        return vis_img, key_str
        
    else:
        # RGB/Color images - direct conversion
        vis_img = _tensor_to_rgb(img, img_height)
        return vis_img, key_str


def img_gen(merged_dict, image_keys=None, image_keys_row1=None, image_keys_row2=None, save_path=None, sample_idx=0, img_height=192, spacing=10, label_height=30):
    """Generate a single PNG image with multiple sub-images arranged in one or two rows with labels
    
    Args:
        merged_dict: Dictionary containing both inputs and outputs merged together
        image_keys: (Deprecated, use image_keys_row1) List of keys for single row layout
        image_keys_row1: List of keys for first row. Special keys:
                   - "depth_err": Computes depth error visualization (requires ("depth_gt", 0, 0) and ("depth", 0, 0) in merged_dict)
                   - Tuples: e.g., ("disp", 0), ("depth", 0, 0), ("color", 0, 0), ("depth_gt", 0, 0)
                   - None: Placeholder for white image
        image_keys_row2: Optional list of keys for second row. If provided, must have same length as row1.
                   - None: Placeholder for white image
        save_path: Path to save the generated image
        sample_idx: Index of the sample in the batch (default: 0)
        img_height: Height of each sub-image (default: 192)
        spacing: Spacing between images (default: 10)
        label_height: Height reserved for labels below each image (default: 30)
    
    Returns:
        None (saves image to save_path)
    
    Note:
        If depth metrics are available in merged_dict (keys like 'abs_rel', 'rmse', etc.),
        they will be displayed at the top of the image.
        If frame information is available (keys like 'dataset', 'keyframe', 'idx'), 
        it will be displayed at the top of the image.
    """
    # Backward compatibility: if image_keys is provided, use it as row1
    if image_keys is not None:
        if image_keys_row1 is not None:
            raise ValueError("Cannot specify both 'image_keys' and 'image_keys_row1'. Use 'image_keys_row1' only.")
        image_keys_row1 = image_keys
    
    if image_keys_row1 is None:
        raise ValueError("Must provide either 'image_keys' or 'image_keys_row1'")
    
    if save_path is None:
        raise ValueError("Must provide 'save_path'")
    
    # Sanity check: if both rows are provided, they must have the same length
    if image_keys_row2 is not None and len(image_keys_row2) > 0:
        if len(image_keys_row1) != len(image_keys_row2):
            raise ValueError(f"image_keys_row1 and image_keys_row2 must have the same length. "
                           f"Got {len(image_keys_row1)} and {len(image_keys_row2)}")
    
    # Process rows
    rows_images = []
    rows_labels = []
    
    for row_idx, row_keys in enumerate([image_keys_row1, image_keys_row2]):
        if row_keys is None or len(row_keys) == 0:
            continue  # Skip empty row
        
        images = []
        labels = []
        
        for key in row_keys:
            vis_img, label = _process_image_key(key, merged_dict, sample_idx, img_height)
            images.append(vis_img)
            labels.append(label)
        
        rows_images.append(images)
        rows_labels.append(labels)
    
    if len(rows_images) == 0:
        print("Warning: No images found to generate!")
        return
    
    # Determine number of columns (use the longest row)
    num_cols = max(len(images) for images in rows_images) if rows_images else 0
    if num_cols == 0:
        print("Warning: No images found to generate!")
        return
    
    # Ensure all rows have the same number of columns (pad with white placeholders if needed)
    for row_idx in range(len(rows_images)):
        while len(rows_images[row_idx]) < num_cols:
            placeholder_width = int(img_height * 1.5)
            white_img = np.ones((img_height, placeholder_width, 3), dtype=np.uint8) * 255
            rows_images[row_idx].append(white_img)
            rows_labels[row_idx].append("")
    
    # Normalize image widths within each column (all images in same column should have same width)
    # This ensures proper alignment across rows
    for col_idx in range(num_cols):
        # Find maximum width in this column across all rows
        max_width = max(rows_images[row_idx][col_idx].shape[1] for row_idx in range(len(rows_images)))
        
        # Resize all images in this column to max_width
        for row_idx in range(len(rows_images)):
            img = rows_images[row_idx][col_idx]
            if img.shape[1] != max_width:
                img = cv2.resize(img, (max_width, img_height))
                rows_images[row_idx][col_idx] = img
    
    # Calculate total width AFTER normalization (all rows should have same column widths now)
    if len(rows_images) > 0 and len(rows_images[0]) > 0:
        # Calculate width based on first row (all rows have same widths after normalization)
        total_width = sum(img.shape[1] for img in rows_images[0]) + spacing * (len(rows_images[0]) - 1)
    else:
        print("Warning: No images to display!")
        return
    
    # Collect metadata for display
    metadata_lines = []
    
    # Check for depth metrics
    metric_keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'median_scaling_ratio']
    metrics_text = []
    for key in metric_keys:
        if key in merged_dict:
            val = merged_dict[key]
            if isinstance(val, (torch.Tensor, np.ndarray)):
                val = float(val.item() if hasattr(val, 'item') else val)
            if isinstance(val, (int, float)):
                if key in ['a1', 'a2', 'a3']:
                    metrics_text.append(f"{key}: {val:.3f}")
                elif key == 'median_scaling_ratio':
                    metrics_text.append(f"scale: {val:.3f}")
                else:
                    metrics_text.append(f"{key}: {val:.4f}")
    
    if metrics_text:
        metadata_lines.append(" | ".join(metrics_text))
    
    # Check for frame information
    frame_info_parts = []
    if 'dataset' in merged_dict:
        dataset_val = merged_dict['dataset']
        if isinstance(dataset_val, (torch.Tensor, np.ndarray)):
            dataset_val = dataset_val.item() if hasattr(dataset_val, 'item') else str(dataset_val)
        frame_info_parts.append(f"dataset{dataset_val}")
    
    if 'keyframe' in merged_dict:
        keyframe_val = merged_dict['keyframe']
        if isinstance(keyframe_val, (torch.Tensor, np.ndarray)):
            keyframe_val = keyframe_val.item() if hasattr(keyframe_val, 'item') else str(keyframe_val)
        frame_info_parts.append(f"keyframe{keyframe_val}")
    
    if 'idx' in merged_dict:
        idx_val = merged_dict['idx']
        if isinstance(idx_val, (torch.Tensor, np.ndarray)):
            idx_val = idx_val.item() if hasattr(idx_val, 'item') else str(idx_val)
        frame_info_parts.append(f"idx{idx_val}")
    
    if frame_info_parts:
        metadata_lines.append("_".join(frame_info_parts))
    
    # Add metadata height if needed
    metadata_height = 0
    if metadata_lines:
        metadata_height = len(metadata_lines) * 20 + 10  # 20px per line + 10px padding
    
    # Calculate total height: metadata + (img_height + label_height) * num_rows + spacing between rows
    num_rows = len(rows_images)
    row_height = img_height + label_height
    row_spacing = spacing if num_rows > 1 else 0
    total_height = metadata_height + num_rows * row_height + (num_rows - 1) * row_spacing
    
    # Create canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Add metadata text at the top
    if metadata_lines:
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        
        # Try to use a nice font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        y_offset = 5
        for line in metadata_lines:
            draw.text((5, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 20
        
        canvas = np.array(pil_canvas)
    
    # Place images and labels for each row
    for row_idx, (images, labels) in enumerate(zip(rows_images, rows_labels)):
        # Calculate y position for this row
        image_y_start = metadata_height + row_idx * (row_height + row_spacing)
        
        # Place images and labels in this row
        x_offset = 0
        for col_idx, (img, label) in enumerate(zip(images, labels)):
            img_width = img.shape[1]
            # Ensure we don't exceed canvas width
            if x_offset + img_width > total_width:
                # Clip image width if necessary
                img_width = total_width - x_offset
                if img_width > 0:
                    img = cv2.resize(img, (img_width, img_height))
                else:
                    break  # No more space
            
            # Place image
            canvas[image_y_start:image_y_start+img_height, x_offset:x_offset+img_width] = img
            
            # Add label text using PIL
            pil_canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_canvas)
            
            # Try to use a nice font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            # Calculate text position (centered below image)
            if label:  # Only draw label if not empty
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x_offset + (img_width - text_width) // 2
                text_y = image_y_start + img_height + (label_height - 16) // 2
                
                # Draw text
                draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
                canvas = np.array(pil_canvas)
            
            x_offset += img_width + spacing
    
    # Save image
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    Image.fromarray(canvas).save(save_path)
    print(f"Saved image grid to {save_path}")

