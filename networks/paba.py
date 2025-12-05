from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffineAlignment(nn.Module):
    """Physics-Aware Brightness Alignment (PABA) Module
    
    Corrects illumination differences between target and warped source images
    using local affine transformations (scale α and shift β) computed per patch.
    
    Args:
        patch_size: Size of non-overlapping patches (default: 8)
        eps: Small epsilon for numerical stability (default: 1e-6)
        min_valid_ratio: Minimum ratio of valid pixels in a patch to compute transformation (default: 0.1)
    """
    
    def __init__(self, patch_size=8, eps=1e-6, min_valid_ratio=0.1):
        super(LocalAffineAlignment, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.min_valid_ratio = min_valid_ratio
    
    def forward(self, target_img, warped_source_img, valid_mask=None):
        """
        Args:
            target_img: (B, C, H, W) Target image
            warped_source_img: (B, C, H, W) Warped source image (geometrically aligned)
            valid_mask: (B, 1, H, W) Optional mask for valid pixels (1=valid, 0=invalid)
        
        Returns:
            aligned_img: (B, C, H, W) Aligned source image
            alpha_map: (B, 1, H, W) Scale factor map for visualization
            beta_map: (B, 1, H, W) Shift factor map for visualization
        """
        B, C, H, W = target_img.shape
        
        # Ensure dimensions are multiples of patch_size (pad if needed)
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            target_img = F.pad(target_img, (0, pad_w, 0, pad_h), mode='reflect')
            warped_source_img = F.pad(warped_source_img, (0, pad_w, 0, pad_h), mode='reflect')
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        H_padded, W_padded = target_img.shape[2], target_img.shape[3]
        
        # Reshape to patches: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        num_patches_h = H_padded // self.patch_size
        num_patches_w = W_padded // self.patch_size
        
        target_patches = target_img.view(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        source_patches = warped_source_img.view(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        
        # Flatten patches: (B, C, num_patches_h, num_patches_w, patch_size * patch_size)
        target_patches_flat = target_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        target_patches_flat = target_patches_flat.view(B, C, num_patches_h, num_patches_w, self.patch_size * self.patch_size)
        
        source_patches_flat = source_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        source_patches_flat = source_patches_flat.view(B, C, num_patches_h, num_patches_w, self.patch_size * self.patch_size)
        
        # Handle valid mask if provided
        if valid_mask is not None:
            valid_patches = valid_mask.view(B, 1, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
            valid_patches = valid_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
            valid_patches = valid_patches.view(B, 1, num_patches_h, num_patches_w, self.patch_size * self.patch_size)
            # Count valid pixels per patch
            num_valid = valid_patches.sum(dim=-1, keepdim=True)  # (B, 1, num_patches_h, num_patches_w, 1)
            min_valid = int(self.min_valid_ratio * self.patch_size * self.patch_size)
            patch_valid = (num_valid >= min_valid).float()  # (B, 1, num_patches_h, num_patches_w, 1)
        else:
            patch_valid = torch.ones(B, 1, num_patches_h, num_patches_w, 1, 
                                   device=target_img.device, dtype=target_img.dtype)
            num_valid = torch.ones_like(patch_valid) * (self.patch_size * self.patch_size)
        
        # Compute statistics per patch per channel
        # For each patch, we solve: target = alpha * source + beta
        # Using least squares: alpha = cov(target, source) / var(source)
        #                      beta = mean(target) - alpha * mean(source)
        
        # Compute means
        if valid_mask is not None:
            # Weighted mean using valid mask
            target_mean = (target_patches_flat * valid_patches).sum(dim=-1, keepdim=True) / (num_valid + self.eps)
            source_mean = (source_patches_flat * valid_patches).sum(dim=-1, keepdim=True) / (num_valid + self.eps)
        else:
            target_mean = target_patches_flat.mean(dim=-1, keepdim=True)  # (B, C, num_patches_h, num_patches_w, 1)
            source_mean = source_patches_flat.mean(dim=-1, keepdim=True)  # (B, C, num_patches_h, num_patches_w, 1)
        
        # Center the patches
        target_centered = target_patches_flat - target_mean
        source_centered = source_patches_flat - source_mean
        
        # Compute variance and covariance
        if valid_mask is not None:
            # Weighted variance and covariance
            source_var = ((source_centered ** 2) * valid_patches).sum(dim=-1, keepdim=True) / (num_valid + self.eps)
            target_source_cov = ((target_centered * source_centered) * valid_patches).sum(dim=-1, keepdim=True) / (num_valid + self.eps)
        else:
            source_var = source_centered.pow(2).mean(dim=-1, keepdim=True)  # (B, C, num_patches_h, num_patches_w, 1)
            target_source_cov = (target_centered * source_centered).mean(dim=-1, keepdim=True)  # (B, C, num_patches_h, num_patches_w, 1)
        
        # Compute alpha and beta (closed-form least squares solution)
        # alpha = cov(target, source) / var(source)
        # Add epsilon to denominator for numerical stability
        alpha = target_source_cov / (source_var + self.eps)
        beta = target_mean - alpha * source_mean
        
        # If patch is invalid (too few valid pixels), set alpha=1, beta=0
        alpha = alpha * patch_valid + (1 - patch_valid)  # alpha = 1 when invalid
        beta = beta * patch_valid  # beta = 0 when invalid
        
        # Average alpha and beta across channels (or keep per-channel)
        # For simplicity, average across channels
        alpha_map = alpha.mean(dim=1, keepdim=True)  # (B, 1, num_patches_h, num_patches_w, 1)
        beta_map = beta.mean(dim=1, keepdim=True)  # (B, 1, num_patches_h, num_patches_w, 1)
        
        # Remove the last dimension
        alpha_map = alpha_map.squeeze(-1)  # (B, 1, num_patches_h, num_patches_w)
        beta_map = beta_map.squeeze(-1)  # (B, 1, num_patches_h, num_patches_w)
        
        # Upsample alpha and beta maps to full resolution using nearest neighbor
        alpha_map_full = F.interpolate(
            alpha_map, size=(H_padded, W_padded), mode='nearest'
        )  # (B, 1, H_padded, W_padded)
        beta_map_full = F.interpolate(
            beta_map, size=(H_padded, W_padded), mode='nearest'
        )  # (B, 1, H_padded, W_padded)
        
        # Apply transformation: aligned = alpha * source + beta
        aligned_img = alpha_map_full * warped_source_img + beta_map_full
        
        # Crop back to original size if we padded
        if pad_h > 0 or pad_w > 0:
            aligned_img = aligned_img[:, :, :H, :W]
            alpha_map_full = alpha_map_full[:, :, :H, :W]
            beta_map_full = beta_map_full[:, :, :H, :W]
        
        return aligned_img, alpha_map_full, beta_map_full

