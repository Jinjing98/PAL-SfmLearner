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
    """Convert tensor to RGB numpy array and resize"""
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().detach().numpy()
    else:
        img_np = np.array(img_tensor)
    
    # Handle shape: (C, H, W) -> (H, W, C)
    if len(img_np.shape) == 3 and img_np.shape[0] <= 4:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Normalize to [0, 255]
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
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


def img_gen(merged_dict, image_keys, save_path, sample_idx=0, img_height=192, spacing=10, label_height=30):
    """Generate a single PNG image with multiple sub-images arranged in a row with labels
    
    Args:
        merged_dict: Dictionary containing both inputs and outputs merged together
        image_keys: List of keys to extract from merged_dict. Special keys:
                   - "depth_err": Computes depth error visualization (requires ("depth_gt", 0, 0) and ("depth", 0, 0) in merged_dict)
                   - Tuples: e.g., ("disp", 0), ("depth", 0, 0), ("color", 0, 0), ("depth_gt", 0, 0)
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
    images = []
    labels = []
    
    for key in image_keys:
        # Special handling for depth_err
        if key == "depth_err":
            gt_key = ("depth_gt", 0, 0)
            pred_key = ("depth", 0, 0)
            
            if gt_key not in merged_dict:
                print(f"Warning: 'depth_err' requires {gt_key} in merged_dict, skipping...")
                continue
            if pred_key not in merged_dict:
                print(f"Warning: 'depth_err' requires {pred_key} in merged_dict, skipping...")
                continue
            
            gt_depth = _get_image_from_outputs(merged_dict, gt_key, sample_idx)
            pred_depth = _get_image_from_outputs(merged_dict, pred_key, sample_idx)
            
            if gt_depth is None or pred_depth is None:
                print(f"Warning: Could not extract depth for depth_err, skipping...")
                continue
            
            # Use existing visualization function
            vis_img = visualize_depth_err(gt_depth, pred_depth, cmap='viridis')
            # Convert from (3, H, W) to (H, W, 3) and resize
            vis_img = np.transpose(vis_img, (1, 2, 0))
            H, W = vis_img.shape[:2]
            aspect_ratio = W / H
            img_width = int(img_height * aspect_ratio)
            vis_img = cv2.resize(vis_img, (img_width, img_height))
            images.append(vis_img)
            labels.append("depth_err")
            continue
        
        # Extract image from merged_dict
        img = _get_image_from_outputs(merged_dict, key, sample_idx)
        if img is None:
            print(f"Warning: Key {key} not found in merged_dict, skipping...")
            continue
        
        # Determine visualization type based on key
        key_str = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        key_lower = key_str.lower()
        
        # Use appropriate visualization function
        if "disp" in key_lower:
            # Disparity visualization
            vis_img = visualize_disp(img)
            vis_img = np.transpose(vis_img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            H, W = vis_img.shape[:2]
            aspect_ratio = W / H
            img_width = int(img_height * aspect_ratio)
            vis_img = cv2.resize(vis_img, (img_width, img_height))
            images.append(vis_img)
            labels.append(key_str)
            
        elif "depth" in key_lower and "err" not in key_lower:
            # Depth visualization (applies to both "depth" and "depth_gt")
            vis_img = visualize_depth(img)
            vis_img = np.transpose(vis_img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            H, W = vis_img.shape[:2]
            aspect_ratio = W / H
            img_width = int(img_height * aspect_ratio)
            vis_img = cv2.resize(vis_img, (img_width, img_height))
            images.append(vis_img)
            labels.append(key_str)
            
        else:
            # RGB/Color images - direct conversion
            vis_img = _tensor_to_rgb(img, img_height)
            images.append(vis_img)
            labels.append(key_str)
    
    if len(images) == 0:
        print("Warning: No images found to generate!")
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
    
    # Calculate total width
    total_width = sum(img.shape[1] for img in images) + spacing * (len(images) - 1)
    
    # Add metadata height if needed
    metadata_height = 0
    if metadata_lines:
        metadata_height = len(metadata_lines) * 20 + 10  # 20px per line + 10px padding
    
    total_height = img_height + label_height + metadata_height
    
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
    
    # Adjust image placement to account for metadata
    image_y_start = metadata_height
    
    # Place images and labels
    x_offset = 0
    for img, label in zip(images, labels):
        img_width = img.shape[1]
        # Place image (offset by metadata height)
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
        
        # Calculate text position (centered below image, offset by metadata height)
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

