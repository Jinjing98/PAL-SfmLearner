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
    
    def __init__(self, patch_size=8, eps=1e-6, min_valid_ratio=0.1, interp_mode='nearest'):
        super(LocalAffineAlignment, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.min_valid_ratio = min_valid_ratio
        self.interp_mode = interp_mode
        assert self.interp_mode in ['nearest', 'bilinear'], f"Invalid interpolation mode: {self.interp_mode}"

        # additional ablation designs
        self.constrain_alpha = True
        self.constrain_type = 'soft_clamp' #relu 
        # 在暗部噪声区域，G 和 B 通道的信息量很小（信噪比低）; 内窥镜图像主要是红色
        self.alpha_beta_computed_shared_cross_channel = True #强制所有通道共享同一个alpha beta. 
    
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
        
        kernel_pixels = self.patch_size * self.patch_size
        
        if self.alpha_beta_computed_shared_cross_channel:
            # -----------------------------------------------------------
            # 【Shared Cross-Channel Mode】: 强制所有通道共享同一个alpha beta
            # 策略: 把 C 个通道的像素展平到 "像素数" 维度 (K*K) 中
            # 将所有RGB像素视为该Patch内的样本点，计算统一的alpha和beta
            # -----------------------------------------------------------
            # (B, C, num_patches_h, num_patches_w, K*K) -> (B, 1, num_patches_h, num_patches_w, C*K*K)
            # 将 RGB 像素全部视为该 Patch 内的样本点
            # target_patches_flat = target_patches_flat.reshape(B, 1, num_patches_h, num_patches_w, C * kernel_pixels)
            # source_patches_flat = source_patches_flat.reshape(B, 1, num_patches_h, num_patches_w, C * kernel_pixels)

# [BUG FIX START] 
            # Original Shape: (B, C, num_patches_h, num_patches_w, K*K)
            # We must move C to the end BEFORE flattening.
            
            # 1. Permute to: (B, num_patches_h, num_patches_w, C, K*K)
            target_patches_flat = target_patches_flat.permute(0, 2, 3, 1, 4).contiguous()
            source_patches_flat = source_patches_flat.permute(0, 2, 3, 1, 4).contiguous()
            
            # 2. Now C and K*K are adjacent, we can safely flatten them
            # New Shape: (B, 1, num_patches_h, num_patches_w, C * K*K)
            target_patches_flat = target_patches_flat.view(B, 1, num_patches_h, num_patches_w, C * kernel_pixels)
            source_patches_flat = source_patches_flat.view(B, 1, num_patches_h, num_patches_w, C * kernel_pixels)
            # [BUG FIX END]

            # Handle valid mask if provided
            if valid_mask is not None:
                # Mask 通常是 (B, 1, H, W)，reshape 后是 (B, 1, num_patches_h, num_patches_w, K*K)
                valid_patches = valid_mask.view(B, 1, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
                valid_patches = valid_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
                valid_patches = valid_patches.view(B, 1, num_patches_h, num_patches_w, kernel_pixels)
                # 重复 C 次以匹配 RGB 通道数
                valid_patches = valid_patches.repeat(1, 1, 1, 1, C)  # (B, 1, num_patches_h, num_patches_w, C*K*K)
                
                # Count valid pixels per patch (now includes all channels)
                num_valid = valid_patches.sum(dim=-1, keepdim=True)  # (B, 1, num_patches_h, num_patches_w, 1)
                min_valid = int(self.min_valid_ratio * C * kernel_pixels)
                patch_valid = (num_valid >= min_valid).float()  # (B, 1, num_patches_h, num_patches_w, 1)
            else:
                patch_valid = torch.ones(B, 1, num_patches_h, num_patches_w, 1, 
                                       device=target_img.device, dtype=target_img.dtype)
                num_valid = torch.ones_like(patch_valid) * (C * kernel_pixels)
            
            # Compute statistics (now based on all RGB pixels combined)
            patch_valid_sum = num_valid + self.eps
            
            # Mean
            target_mean = (target_patches_flat * valid_patches).sum(dim=-1, keepdim=True) / patch_valid_sum
            source_mean = (source_patches_flat * valid_patches).sum(dim=-1, keepdim=True) / patch_valid_sum
            
            # Center the patches
            target_centered = (target_patches_flat - target_mean) * valid_patches
            source_centered = (source_patches_flat - source_mean) * valid_patches
            
            # Covariance & Variance
            target_source_cov = (target_centered * source_centered).sum(dim=-1, keepdim=True) / patch_valid_sum
            source_var = (source_centered ** 2).sum(dim=-1, keepdim=True) / patch_valid_sum
            
            # Compute alpha and beta (closed-form least squares solution)
            alpha_raw = target_source_cov / (source_var + self.eps)
            
            # Constrain alpha to avoid negative values
            if self.constrain_alpha:
                if self.constrain_type == 'soft_clamp':
                    alpha = torch.clamp(alpha_raw, min=0.01)
                elif self.constrain_type == 'relu':
                    alpha = F.relu(alpha_raw)
                else:
                    raise ValueError(f"Invalid constrain type: {self.constrain_type}")
            else:
                alpha = alpha_raw
            
            beta = target_mean - alpha * source_mean
            
            # If patch is invalid (too few valid pixels), set alpha=1, beta=0
            alpha = alpha * patch_valid + (1 - patch_valid)  # alpha = 1 when invalid
            beta = beta * patch_valid  # beta = 0 when invalid
            
            # Alpha and beta are (B, 1, num_patches_h, num_patches_w, 1)
            # Squeeze the last dimension to get (B, 1, num_patches_h, num_patches_w)
            # This is the correct shape for upsampling
            alpha_map = alpha.squeeze(-1)  # (B, 1, num_patches_h, num_patches_w)
            beta_map = beta.squeeze(-1)    # (B, 1, num_patches_h, num_patches_w)
            
        else:
            # -----------------------------------------------------------
            # 【Per-Channel Mode】: 原始实现，每个通道独立计算alpha和beta
            # -----------------------------------------------------------
            # Handle valid mask if provided
            if valid_mask is not None:
                valid_patches = valid_mask.view(B, 1, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
                valid_patches = valid_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
                valid_patches = valid_patches.view(B, 1, num_patches_h, num_patches_w, kernel_pixels)
                # Expand to match channel dimension
                valid_patches = valid_patches.expand(B, C, num_patches_h, num_patches_w, kernel_pixels)
                
                # Count valid pixels per patch
                num_valid = valid_patches.sum(dim=-1, keepdim=True)  # (B, C, num_patches_h, num_patches_w, 1)
                min_valid = int(self.min_valid_ratio * kernel_pixels)
                patch_valid = (num_valid >= min_valid).float()  # (B, C, num_patches_h, num_patches_w, 1)
            else:
                patch_valid = torch.ones(B, C, num_patches_h, num_patches_w, 1, 
                                       device=target_img.device, dtype=target_img.dtype)
                num_valid = torch.ones_like(patch_valid) * kernel_pixels
            
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

            # further constrain the alpha to avoid negative values
            if self.constrain_alpha:
                if self.constrain_type == 'soft_clamp':
                    # Soft Clamp (防止纯黑死区，给一点点底数)
                    # 允许光照变暗，但不允许反转，且保留微弱的正梯度    
                    alpha = torch.clamp(alpha, min=0.01)
                elif self.constrain_type == 'relu':
                    alpha = F.relu(alpha)
                else:
                    raise ValueError(f"Invalid constrain type: {self.constrain_type}")

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
        
        # Verify alpha_map and beta_map shapes before upsampling
        # They should be (B, 1, num_patches_h, num_patches_w)
        assert len(alpha_map.shape) == 4, f"alpha_map should be 4D, got shape {alpha_map.shape}"
        assert len(beta_map.shape) == 4, f"beta_map should be 4D, got shape {beta_map.shape}"
        assert alpha_map.shape[1] == 1, f"alpha_map channel dim should be 1, got {alpha_map.shape[1]}"
        
        # Upsample alpha and beta maps to full resolution using nearest neighbor
        alpha_map_full = F.interpolate(
            alpha_map, size=(H_padded, W_padded), 
            mode=self.interp_mode,
        )  # (B, 1, H_padded, W_padded)
        beta_map_full = F.interpolate(
            beta_map, size=(H_padded, W_padded), 
            mode=self.interp_mode,
        )  # (B, 1, H_padded, W_padded)
        
        # Verify shapes after upsampling
        assert alpha_map_full.shape == (B, 1, H_padded, W_padded), \
            f"alpha_map_full shape mismatch: {alpha_map_full.shape} vs expected {(B, 1, H_padded, W_padded)}"
        assert beta_map_full.shape == (B, 1, H_padded, W_padded), \
            f"beta_map_full shape mismatch: {beta_map_full.shape} vs expected {(B, 1, H_padded, W_padded)}"
        assert warped_source_img.shape == (B, C, H_padded, W_padded), \
            f"warped_source_img shape mismatch: {warped_source_img.shape} vs expected {(B, C, H_padded, W_padded)}"
        
        # Apply transformation: aligned = alpha * source + beta
        # Broadcasting: alpha_map_full (B, 1, H, W) * warped_source_img (B, C, H, W) -> (B, C, H, W)
        # This works for both shared and per-channel modes since alpha_map_full is (B, 1, H, W)
        aligned_img = alpha_map_full * warped_source_img + beta_map_full
        
        # Verify output shape
        assert aligned_img.shape == (B, C, H_padded, W_padded), \
            f"aligned_img shape mismatch: {aligned_img.shape} vs expected {(B, C, H_padded, W_padded)}"
        
        # Crop back to original size if we padded
        if pad_h > 0 or pad_w > 0:
            aligned_img = aligned_img[:, :, :H, :W]
            alpha_map_full = alpha_map_full[:, :, :H, :W]
            beta_map_full = beta_map_full[:, :, :H, :W]
        
        return aligned_img, alpha_map_full, beta_map_full

