import numpy as np
import cv2
import torch

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
    pred_depth_tensor = outputs[("depth", 0, 0)].detach()  # (B, 1, H, W)
    
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


def transl_ang_loss(t, tgt, eps=1e-6):
    """
    Compute translation direction angular error.
    Args: 
        t: estimated translation vector [B, 3]
        tgt: ground-truth translation vector [B, 3]
    Returns: 
        T_err_mean: mean translation direction angular error (in radians)
        T_err: translation direction angular error per sample [B]
    """
    assert t.dim() == 2, f't: {t.shape}'
    assert tgt.dim() == 2, f'tgt: {tgt.shape}'
    assert t.shape[1] == 3, f't: {t.shape}'
    assert tgt.shape[1] == 3, f'tgt: {tgt.shape}'
    
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t_normed = t / (t_norm + eps)
    tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
    tgt_normed = tgt / (tgt_norm + eps)
    cosine = torch.sum(t_normed * tgt_normed, dim=1)
    T_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    return T_err.mean(), T_err


def transl_scale_loss(t, tgt, eps=1e-6, norm_gt=False, norm_esti=False):
    """
    Compute translation scale error.
    Args: 
        t: estimated translation vector [B, 3]
        tgt: ground-truth translation vector [B, 3]
        norm_gt: whether to normalize ground truth vectors before computing loss
        norm_esti: whether to normalize estimated vectors before computing loss
        eps: small value to prevent division by zero
    Returns: 
        T_err_mean: mean translation scale error
        T_err: translation scale error per sample [B]
    """
    if norm_esti:
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t_normed = t / (t_norm + eps)
    else:
        t_normed = t
    if norm_gt:
        tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
        tgt_normed = tgt / (tgt_norm + eps)
    else:
        tgt_normed = tgt
    
    T_err = torch.norm(t_normed - tgt_normed, dim=1)
    return T_err.mean(), T_err


def rot_ang_loss(R, Rgt, eps=1e-6):
    """
    Compute rotation angular error.
    Args:
        R: estimated rotation matrix [B, 3, 3]
        Rgt: ground-truth rotation matrix [B, 3, 3]
    Returns: 
        R_err_mean: mean rotation angular error (in radians)
        R_err: rotation angular error per sample [B]
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    return R_err.mean(), R_err



def rot_ang_loss_num_stable(R, Rgt, eps=1e-6, frob_tol=1e-6):
    """
    Numerically stable rotation angular error with orthonormalization.
    Args:
        R: estimated rotation matrix [B, 3, 3]
        Rgt: ground-truth rotation matrix [B, 3, 3]
        eps: clamp epsilon for acos
        frob_tol: tolerance on Frobenius norm to treat residual as identity
    Returns:
        R_err_mean: mean rotation angular error (in radians)
        R_err: rotation angular error per sample [B]
    """
    # Do the angle computation in float64 for better numerical stability and
    # re-orthonormalize the rotation blocks to suppress tiny residual angles.
    R_d = R.double()
    Rgt_d = Rgt.double()

    def _orthonormalize(rot):
        # SVD-based projection onto SO(3)
        U, _, Vh = torch.linalg.svd(rot)
        rot_ortho = U @ Vh
        # Fix possible reflection to ensure det=+1
        det = torch.linalg.det(rot_ortho)
        neg_mask = det < 0
        if neg_mask.any():
            Vh_fix = Vh.clone()
            Vh_fix[neg_mask, -1, :] *= -1
            rot_ortho = U @ Vh_fix
        return rot_ortho

    R_o = _orthonormalize(R_d)
    Rgt_o = _orthonormalize(Rgt_d)

    residual = torch.matmul(R_o.transpose(1, 2), Rgt_o)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)

    # If matrices are effectively identical, avoid tiny residual angles.
    # Use a Frobenius norm check on the relative rotation to be robust.
    I = torch.eye(3, device=residual.device, dtype=residual.dtype).unsqueeze(0)
    delta = residual - I
    frob_norm = torch.linalg.norm(delta, dim=(1, 2))
    close_to_identity = frob_norm < frob_tol

    R_err = torch.acos(cosine)
    R_err[close_to_identity] = 0.0
    return R_err.mean().to(R.dtype), R_err.to(R.dtype)


def compute_pose_error_v2(gt_rel_poses, pred_rel_poses, ret_raw=False):
    """
    Compute pose errors between ground truth and predicted relative poses.
    Args:
        gt_rel_poses: (B, 4, 4) Ground truth relative poses
        pred_rel_poses: (B, 4, 4) Predicted relative poses
    Returns:
        err_dict: Dictionary containing translation and rotation errors
        Metrics include:
            trans_err_ang_deg: translation direction angular error (degrees)
            trans_err_scale: translation scale error
            rot_err_deg: rotation angular error (degrees)
    """
    # Extract translation and rotation components
    t = pred_rel_poses[:, 0:3, -1]  # [B, 3]
    tgt = gt_rel_poses[:, 0:3, -1]  # [B, 3]
    R = pred_rel_poses[:, :3, :3]   # [B, 3, 3]
    Rgt = gt_rel_poses[:, :3, :3]   # [B, 3, 3]

    # Compute translation error with angular version and scale version
    trans_err_ang, trans_err_ang_raw = transl_ang_loss(t, tgt)
    trans_err_scale, trans_err_scale_raw = transl_scale_loss(t, tgt, norm_gt=False, norm_esti=False)

    # Compute rotation error
    # rot_err, rot_err_raw = rot_ang_loss(R, Rgt)
    # expensive but num stable: is able to report 0 rot_err if the rot mat are exacitly the same.
    # (0.08 deg  if not)
    rot_err, rot_err_raw = rot_ang_loss_num_stable(R, Rgt)
    
    err_dict = {
        'trans_err_ang_deg': trans_err_ang * 180 / torch.pi,
        'trans_err_scale': trans_err_scale,
        'rot_err_deg': rot_err * 180 / torch.pi
    }
        
    if ret_raw:
        trans_err_ang_raw_dict = { f'trans_err_ang_deg_raw': (trans_err_ang_raw*180/torch.pi).tolist() }
        rot_err_raw_dict = { f'rot_err_deg_raw': (rot_err_raw*180/torch.pi).tolist() }
        return err_dict, trans_err_ang_raw_dict, rot_err_raw_dict
    else:
        return err_dict


def compute_pose_metrics(inputs, outputs, frame_ids, ret_raw=False):
    """
    Compute pose metrics for a validation batch.
    
    Args:
        inputs: Input batch dictionary (should contain ("gt_c2w_poses", frame_id) if GT poses are available)
        outputs: Output batch dictionary (should contain ("cam_T_cam", 0, frame_id) for predicted poses)
        frame_ids: List of frame IDs to process (e.g., [0, -1, 1])
    
    Returns:
        Dictionary of pose metrics (empty if GT poses not available)
    """
    metrics_dict = {}
    metrics_trans_ang_raw_dict = {}
    metrics_rot_err_raw_dict = {}

    # Check if GT poses are available
    if ("gt_c2w_poses", 0) not in inputs:
        if ret_raw:
            return metrics_dict, metrics_trans_ang_raw_dict, metrics_rot_err_raw_dict
        return metrics_dict
    
    # Get GT absolute poses for target frame (frame 0)
    gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
    
    # Compute metrics for each source frame
    for frame_id in frame_ids[1:]:
        if frame_id == "s":
            continue  # Skip stereo frames
        
        # Get GT absolute poses for source frame
        if ("gt_c2w_poses", frame_id) not in inputs:
            continue
        
        gt_src_abs_poses = inputs[("gt_c2w_poses", frame_id)]  # (B, 4, 4)
        
        # Get predicted relative poses
        if ("cam_T_cam", 0, frame_id) not in outputs:
            continue
        
        pred_rel_poses_batch = outputs[("cam_T_cam", 0, frame_id)].detach()  # (B, 4, 4)
        
        # Compute GT relative poses: T_target_to_source = inv(T_source) @ T_target
        gt_tgt2src_rel_poses = torch.linalg.inv(gt_src_abs_poses) @ gt_tgt_abs_poses
        
        assert gt_tgt2src_rel_poses.shape == pred_rel_poses_batch.shape, \
            f'gt_tgt2src_rel_poses.shape: {gt_tgt2src_rel_poses.shape}, pred_rel_poses_batch.shape: {pred_rel_poses_batch.shape}'
        
        # Compute pose errors
        if ret_raw:
            err_dict, trans_err_ang_raw_dict, rot_err_raw_dict = compute_pose_error_v2(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach(), ret_raw=True)
        else:
            err_dict = compute_pose_error_v2(gt_tgt2src_rel_poses, pred_rel_poses_batch.detach())

        # Accumulate metrics (average across frames)
        # def accumulate_metrics_across_frames(err_dict, metrics_dict):
        #     for k, v in err_dict.items():
        #         key = f"pose_{k}"
        #         if key not in metrics_dict:
        #             metrics_dict[key] = []
        #         metrics_dict[key].append(v.item())
        #     return metrics_dict
        def _accum(acc, new_metrics, key_prefix="pose_"):
            for k, v in new_metrics.items():
                acc.setdefault(f"{key_prefix}{k}", []).append(float(v))
        def _accum_raw(acc, new_metrics, key_prefix="pose_"):
            for k, v in new_metrics.items():
                # acc.setdefault(f"{key_prefix}{k}", []).append(v)
                acc.setdefault(f"{key_prefix}{k}", []).extend(v)

        # update metrics_dict via appending across frames
        _accum(metrics_dict, err_dict)
        if ret_raw:
            _accum_raw(metrics_trans_ang_raw_dict, trans_err_ang_raw_dict)
            _accum_raw(metrics_rot_err_raw_dict, rot_err_raw_dict)
        
        # manully extend some metrics
        # Log scale of estimated translation
        pred_rel_trans_scale = pred_rel_poses_batch[:, :3, 3].norm(dim=1).mean()
        if "pose_pred_rel_trans_scale" not in metrics_dict:
            metrics_dict["pose_pred_rel_trans_scale"] = []
        metrics_dict["pose_pred_rel_trans_scale"].append(pred_rel_trans_scale.item())
        
        # Log scale of depth
        if ("depth", 0, 0) in outputs:
            pred_f0_depth_scale = outputs[("depth", 0, 0)].mean()
            if "pose_pred_f0_depth_scale" not in metrics_dict:
                metrics_dict["pose_pred_f0_depth_scale"] = []
            metrics_dict["pose_pred_f0_depth_scale"].append(pred_f0_depth_scale.item())
    
    # Average metrics across all frames within the batch
    def _avg_metrics_across_frames(metrics_dict):
        for k in list(metrics_dict.keys()):
            if len(metrics_dict[k]) > 0:
                metrics_dict[k] = sum(metrics_dict[k]) / len(metrics_dict[k])
            else:
                del metrics_dict[k]
        return metrics_dict

    if not ret_raw:
        return _avg_metrics_across_frames(metrics_dict)
    else:
        return _avg_metrics_across_frames(metrics_dict), metrics_trans_ang_raw_dict, metrics_rot_err_raw_dict