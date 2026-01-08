from __future__ import absolute_import, division, print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_smooth_light(light, img):
    """Computes the smoothness loss for a light image
    """
    grad_light_x = torch.abs(light[:, :, :, :-1] - light[:, :, :, 1:])
    grad_light_y = torch.abs(light[:, :, :-1, :] - light[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    epsilon_x = 0.01 * torch.ones_like(grad_img_x)
    Denominator_x = torch.max(grad_img_x, epsilon_x)
    x_loss = torch.abs(torch.div(grad_light_x, Denominator_x))

    epsilon_y = 0.01 * torch.ones_like(grad_img_y)
    Denominator_y = torch.max(grad_img_y, epsilon_y)
    y_loss = torch.abs(torch.div(grad_light_y, Denominator_y))
    
    return x_loss.mean() + y_loss.mean()


def get_smooth_bright(transform, target, pred, occu_mask):
    
    """Computes the smoothness loss for a appearance flow
    """
    grad_transform_x = torch.mean(torch.abs(transform[:, :, :, :-1] - transform[:, :, :, 1:]), 1, keepdim=True)
    grad_transform_y = torch.mean(torch.abs(transform[:, :, :-1, :] - transform[:, :, 1:, :]), 1, keepdim=True)
     
    residue = (target - pred)
    
    grad_residue_x = torch.mean(torch.abs(residue[:, :, :, :-1] - residue[:, :, :, 1:]), 1, keepdim=True)
    grad_residue_y = torch.mean(torch.abs(residue[:, :, :-1, :] - residue[:, :, 1:, :]), 1, keepdim=True)

    mask_x = occu_mask[:, :, :, :-1]
    mask_y = occu_mask[:, :, :-1, :]

    # grad_residue_x = grad_residue_x * mask_x / (mask_x.mean() + 1e-7)
    # grad_residue_y = grad_residue_y * mask_y / (mask_y.mean() + 1e-7)
    
    grad_transform_x *= torch.exp(-grad_residue_x)
    grad_transform_y *= torch.exp(-grad_residue_y)

    grad_transform_x *= mask_x
    grad_transform_y *= mask_y
    
    return (grad_transform_x.sum() / mask_x.sum() + grad_transform_y.sum() / mask_y.sum())

def compute_flow_huber_loss(pred,        # rigid flow: [B, 2, H, W]
                            target,      # optical flow: [B, 2, H, W]
                            occu_mask,   # [B, 1, H, W] or [B, H, W]
                            delta=1.0,
                            eps=1e-7):
    """
    Huber (SmoothL1) loss for optical flow with occlusion mask.
    Small residual -> L2 (rotation-friendly)
    Large residual -> L1 (robust to outliers)
    """

    if occu_mask.dim() == 3:
        occu_mask = occu_mask.unsqueeze(1)  # [B,1,H,W]

    # flow residual
    residue = pred - target                       # [B,2,H,W]
    residue_norm = torch.norm(residue, dim=1, keepdim=True)  # [B,1,H,W]

    # apply occlusion mask
    residue_norm = residue_norm * occu_mask

    # Huber formulation
    abs_res = residue_norm
    quadratic = torch.clamp(abs_res, max=delta)
    linear = abs_res - quadratic

    huber_loss = 0.5 * quadratic ** 2 / delta + linear

    # normalize by valid pixels
    loss = huber_loss.sum() / (occu_mask.sum() + eps)

    return loss


def compute_flow_berhu_loss(pred,        # rigid flow: [B, 2, H, W]
                            target,      # optical flow: [B, 2, H, W]
                            occu_mask,   # [B, 1, H, W] or [B, H, W]
                            c_ratio=0.2,
                            eps=1e-7):
    """
    BerHu loss for optical flow with occlusion mask.
    """

    if occu_mask.dim() == 3:
        occu_mask = occu_mask.unsqueeze(1)  # [B,1,H,W]

    # flow residual
    residue = pred - target                      # [B,2,H,W]
    residue_norm = torch.norm(residue, dim=1, keepdim=True)  # [B,1,H,W]

    # apply occlusion mask
    residue_norm = residue_norm * occu_mask

    # berHu threshold (detach!)
    max_res = residue_norm.max()
    c = c_ratio * max_res.detach() + eps

    # berHu formulation
    l1_part = residue_norm
    l2_part = (residue_norm ** 2 + c ** 2) / (2.0 * c)

    berhu_loss = torch.where(residue_norm <= c, l1_part, l2_part)

    # normalize by valid pixels (same spirit as your smooth loss)
    loss = berhu_loss.sum() / (occu_mask.sum() + eps)

    return loss


def compute_local_sums(I, J, filt, stride, padding, win):

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ncc_loss(I, J, win=None):
    """
    calculate the normalize local cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [5] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    # return -1 * torch.mean(cc)
    return -1 * cc


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(pred, target, ssim):
    """Computes reprojection loss between predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def compute_hf_distillation_loss(student_hf, teacher_hf, loss_type='mse'):
    """
    Compute high-frequency distillation loss.
    
    Args:
        student_hf: Student high-frequency map (B, C, H, W)
        teacher_hf: Teacher high-frequency map (B, C, H, W)
        loss_type: 'mse' or 'l1' (default: 'mse')
    
    Returns:
        Scalar loss value
    """
    if loss_type == 'mse':
        return F.mse_loss(student_hf, teacher_hf)
    elif loss_type == 'l1':
        return F.l1_loss(student_hf, teacher_hf)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'mse' or 'l1'.")


def compute_topology_loss(student_hf, teacher_hf):
    """
    Compute topology loss using Wasserstein distance between persistent homology diagrams.
    Placeholder implementation using simple MSE for now.
    Can be replaced with perslay/gudhi for actual persistent homology.
    
    Args:
        student_hf: Student high-frequency map (B, C, H, W)
        teacher_hf: Teacher high-frequency map (B, C, H, W)
    
    Returns:
        Scalar loss value
    """
    # Placeholder: simple MSE approximation
    # TODO: Replace with actual persistent homology computation using perslay/gudhi
    # For now, use a simple approximation that captures topological differences
    return torch.mean((student_hf - teacher_hf)**2)


def compute_reconstruction_loss_hfd(student_depth_norm, target_depth_norm, loss_type='l1'):
    """
    Compute reconstruction loss on normalized depth for HFD training.
    
    Args:
        student_depth_norm: Normalized student depth (B, C, H, W)
        target_depth_norm: Normalized target depth (B, C, H, W)
        loss_type: 'l1' or 'l2' (default: 'l1')
    
    Returns:
        Scalar loss value
    """
    if loss_type == 'l1':
        return F.l1_loss(student_depth_norm, target_depth_norm)
    elif loss_type == 'l2':
        return F.mse_loss(student_depth_norm, target_depth_norm)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'l1' or 'l2'.")


def compute_hfd_losses(student_depth, teacher_depth, gt_depth=None,
                       hf_loss_weight=0.4, topo_loss_weight=0.15, recon_loss_weight=1.0,
                       hf_loss_type='mse', recon_loss_type='l1', grad_threshold=0.01):
    """
    Compute all HFD losses: HF distillation, topology, and reconstruction.
    Reuses functions from utils.hfd for HF map extraction.
    
    Args:
        student_depth: Student predicted depth (B, C, H, W) or (B, H, W)
        teacher_depth: Teacher depth from DepthAnything (B, C, H, W) or (B, H, W)
        gt_depth: Optional ground truth depth (B, C, H, W) or (B, H, W)
        hf_loss_weight: Weight for HF distillation loss (default: 0.4)
        topo_loss_weight: Weight for topology loss (default: 0.15)
        recon_loss_weight: Weight for reconstruction loss (default: 1.0)
        hf_loss_type: 'mse' or 'l1' for HF loss (default: 'mse')
        recon_loss_type: 'l1' or 'l2' for reconstruction loss (default: 'l1')
        grad_threshold: Gradient threshold for HF mask (default: 0.01)
    
    Returns:
        Dictionary with individual losses and total loss
    """
    assert student_depth.shape == teacher_depth.shape, "Student and teacher depth must have the same shape"
    assert student_depth.dim() == 4, "Student depth must be 4D"
    from utils.hfd import dwt_n_layer, scale_invariant_normalize, hf_mask_map
    # Normalize depths for scale-invariant comparison
    student_norm = scale_invariant_normalize(student_depth)
    teacher_norm = scale_invariant_normalize(teacher_depth)
    
    # Extract high-frequency maps using (2-layer DWT)
    #
    student_hf = dwt_n_layer(student_norm, 2, wave='db2', use_torch_wavelets=True)
    teacher_hf = dwt_n_layer(teacher_norm, 2, wave='db2', use_torch_wavelets=True)
    
    # Apply HF mask to remove low-gradient regions and noise
    student_hf_masked, _ = hf_mask_map(student_hf, grad_threshold=grad_threshold)
    teacher_hf_masked, mask = hf_mask_map(teacher_hf, grad_threshold=grad_threshold)
    
    # Compute HF distillation loss (only on masked regions)
    if mask.sum() > 0:
        loss_hf = compute_hf_distillation_loss(student_hf_masked, teacher_hf_masked, loss_type=hf_loss_type)
    else:
        loss_hf = torch.tensor(0.0, device=student_depth.device)
    
    # Compute topology loss
    loss_topo = compute_topology_loss(student_hf_masked, teacher_hf_masked)
    
    # Compute reconstruction loss
    if gt_depth is not None:
        gt_norm = scale_invariant_normalize(gt_depth)
        loss_recon = compute_reconstruction_loss_hfd(student_norm, gt_norm, loss_type=recon_loss_type)
    else:
        # Use teacher as target if GT not available
        loss_recon = compute_reconstruction_loss_hfd(student_norm, teacher_norm, loss_type=recon_loss_type)
    
    # Total loss
    total_loss = (recon_loss_weight * loss_recon + 
                  hf_loss_weight * loss_hf + 
                  topo_loss_weight * loss_topo)
    
    return {
        'loss': total_loss,
        'loss_recon': loss_recon,
        'loss_hf': loss_hf,
        'loss_topo': loss_topo
    }


def compute_losses(inputs, outputs, opt, ssim):
    """Computes all losses for the model
    
    Returns:
        losses: dict containing 'loss' (total weighted loss) and all sub-losses (raw, unweighted)
    """
    # Determine supervision mode
    # is_decompose_mode = opt.reproj_supervise_type != "paba_color_warp"
    is_decompose_mode = opt.reproj_supervise_type == "reprojection_color_warp"
    
    # Initialize all losses
    loss_reconstruction = 0
    loss_reflec = 0
    loss_reprojection = 0
    loss_disp_smooth = 0
    
    # Count dividends
    num_frame_ids = len(opt.frame_ids)
    num_pose_frames = len(opt.frame_ids[1:])
    
    # ============================================================
    # Decompose-based losses (only for decompose methods)
    # ============================================================
    if is_decompose_mode:
        # Reconstruction loss (averaged over all frame_ids)
        for frame_id in opt.frame_ids:
            loss_reconstruction += (compute_reprojection_loss(
                inputs[("color_aug", frame_id, 0)], 
                outputs[("reprojection_color", 0, frame_id)],
                ssim
            )).mean()
        loss_reconstruction = loss_reconstruction / num_frame_ids

    # ============================================================
    # Reprojection and reflectance losses (for all source frames)
    # ============================================================
    for frame_id in opt.frame_ids[1:]: 
        mask = outputs[("valid_mask", 0, frame_id)]
        
        # Reflectance loss (only for decompose-based methods)
        if is_decompose_mode:
            loss_reflec += (torch.abs(
                outputs[("reflectance", 0, 0)] - outputs[("reflectance_warp", 0, frame_id)]
            ).mean(1, True) * mask).sum() / mask.sum()
        
        # Reprojection loss (works for all supervision types)
        supervise_which = outputs[(opt.reproj_supervise_type, 0, frame_id)]

        # debug_af_ori_correct_tgt = True
        # if debug_af_ori_correct_tgt and opt.reproj_supervise_type == "afstyle_color_warp":
        #     # apply the learned delta on gt_tgt img; while warped_src is the raw one.
        #     loss_reprojection += (compute_reprojection_loss(
        #         inputs[("color_aug", 0, 0)] - outputs[("transform", 0, frame_id)], 
        #         outputs[("color_warp", 0, frame_id)],
        #         ssim
        #     ) * mask).sum() / mask.sum()
        #     continue

        loss_reprojection += (compute_reprojection_loss(
            inputs[("color_aug", 0, 0)], 
            supervise_which,
            ssim
        ) * mask).sum() / mask.sum()
    
    # Normalize losses by number of frames
    if num_pose_frames > 0:
        if is_decompose_mode:
            loss_reflec = loss_reflec / num_pose_frames
        loss_reprojection = loss_reprojection / num_pose_frames
            
    # Disparity smoothness loss
    disp = outputs[("disp", 0)]
    color = inputs[("color_aug", 0, 0)]
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    loss_disp_smooth = get_smooth_loss(norm_disp, color)
 
    # Total weighted loss
    total_loss = (opt.reprojection_constraint * loss_reprojection + 
                  opt.reflec_constraint * loss_reflec + 
                  opt.disparity_smoothness * loss_disp_smooth + 
                  opt.reconstruction_constraint * loss_reconstruction)
    
    # Return all losses
    losses = {
        "loss": total_loss,
        "loss_reconstruction": loss_reconstruction,
        "loss_reflec": loss_reflec,
        "loss_reprojection": loss_reprojection,
        "loss_disp_smooth": loss_disp_smooth
    }
    
    return losses

