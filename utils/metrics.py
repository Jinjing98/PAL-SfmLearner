import numpy as np
import cv2

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Same as evaluate_depth.py compute_errors function
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_depth_metrics(inputs, outputs):
    """Compute depth metrics for a validation batch
    
    Args:
        inputs: Input batch dictionary (should contain ("depth_gt", 0, 0) if GT depths are available)
        outputs: Output batch dictionary (should contain ("depth", 0, 0) for predicted depth)
    
    Returns:
        Dictionary of depth metrics (empty if GT depths not available)
    """
    
    # Check if GT depth is available in inputs
    if ("depth_gt", 0, 0) not in inputs:
        return {}
    
    # Get predicted depth from outputs
    if ("depth", 0, 0) not in outputs:
        return {}
    
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150
    
    # Get GT depth and predicted depth
    gt_depth_tensor = inputs[("depth_gt", 0, 0)]  # (B, 1, H_gt, W_gt) or (B, H_gt, W_gt)
    pred_depth_tensor = outputs[("depth", 0, 0)]  # (B, 1, H, W)
    
    # Handle different tensor shapes
    if len(gt_depth_tensor.shape) == 3:
        gt_depth_tensor = gt_depth_tensor.unsqueeze(1)  # Add channel dimension
    
    B, _, H, W = pred_depth_tensor.shape
    _, _, H_gt, W_gt = gt_depth_tensor.shape
    
    # Convert to numpy
    gt_depth_np = gt_depth_tensor.cpu().numpy()  # (B, 1, H_gt, W_gt)
    pred_depth_np = pred_depth_tensor.cpu().numpy()  # (B, 1, H, W)
    
    errors = []
    ratios = []
    
    for b in range(B):
        # Get GT depth for this sample
        gt_depth = gt_depth_np[b, 0]  # (H_gt, W_gt)
        
        # Get predicted depth for this sample and resize to GT size
        pred_depth_b = pred_depth_np[b, 0]  # (H, W)
        pred_depth_resized = cv2.resize(pred_depth_b, (W_gt, H_gt))
        
        # Create mask based on depth cutoff
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        
        # Apply mask
        pred_depth_masked = pred_depth_resized[mask]
        gt_depth_masked = gt_depth[mask]
        
        if len(pred_depth_masked) == 0:
            continue
        
        # Apply median scaling (mono evaluation)
        ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
        ratios.append(ratio)
        pred_depth_masked = pred_depth_masked * ratio
        
        # Clip to valid range
        pred_depth_masked = np.clip(pred_depth_masked, MIN_DEPTH, MAX_DEPTH)
        
        # Compute errors
        error = compute_depth_errors(gt_depth_masked, pred_depth_masked)
        errors.append(error)
    
    if len(errors) == 0:
        return {}
    
    # Average errors across batch
    mean_errors = np.array(errors).mean(0)
    
    # Return as dictionary
    depth_metrics = {
        'abs_rel': mean_errors[0],
        'sq_rel': mean_errors[1],
        'rmse': mean_errors[2],
        'rmse_log': mean_errors[3],
        'a1': mean_errors[4],
        'a2': mean_errors[5],
        'a3': mean_errors[6],
    }
    
    if len(ratios) > 0:
        depth_metrics['median_scaling_ratio'] = np.median(ratios)
        depth_metrics['median_scaling_std'] = np.std(ratios / np.median(ratios))
    
    return depth_metrics