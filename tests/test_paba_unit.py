from __future__ import absolute_import, division, print_function

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.paba import LocalAffineAlignment
from loss import SSIM


def synthesize_lighting_degradation(img_tensor, degradation_type='gradient'):
    """
    Synthesize lighting degradation to simulate illumination differences.
    
    Args:
        img_tensor: (B, C, H, W) Target image tensor
        degradation_type: Type of degradation ('gradient', 'contrast', 'spotlight', 'combined')
    
    Returns:
        degraded_img: (B, C, H, W) Degraded image with lighting changes
    """
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    if degradation_type == 'gradient':
        # Create a gradient mask (bright on left, dark on right)
        x = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        gradient = 0.3 + 0.7 * x  # Scale from 0.3 to 1.0
        degraded = img_tensor * gradient
    
    elif degradation_type == 'contrast':
        # Reduce contrast globally
        mean = img_tensor.mean()
        degraded = 0.5 * mean + 0.5 * img_tensor  # Reduce contrast by 50%
    
    elif degradation_type == 'spotlight':
        # Create a spotlight effect (bright center, dark edges)
        center_h, center_w = H // 2, W // 2
        y = torch.arange(H, device=device).float().view(H, 1)
        x = torch.arange(W, device=device).float().view(1, W)
        dist_sq = (y - center_h) ** 2 + (x - center_w) ** 2
        max_dist_sq = (H/2) ** 2 + (W/2) ** 2
        spotlight = 0.4 + 0.6 * (1 - dist_sq / max_dist_sq).clamp(0, 1)
        spotlight = spotlight.view(1, 1, H, W)
        degraded = img_tensor * spotlight
    
    elif degradation_type == 'combined':
        # Combine multiple degradations
        # Gradient
        x = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        gradient = 0.4 + 0.6 * x
        # Spotlight
        center_h, center_w = H // 2, W // 2
        y = torch.arange(H, device=device).float().view(H, 1)
        x_spot = torch.arange(W, device=device).float().view(1, W)
        dist_sq = (y - center_h) ** 2 + (x_spot - center_w) ** 2
        max_dist_sq = (H/2) ** 2 + (W/2) ** 2
        spotlight = 0.5 + 0.5 * (1 - dist_sq / max_dist_sq).clamp(0, 1)
        spotlight = spotlight.view(1, 1, H, W)
        # Combine
        combined_mask = gradient * spotlight
        degraded = img_tensor * combined_mask
    
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")
    
    # Clamp to valid range
    degraded = degraded.clamp(0, 1)
    
    return degraded


def compute_metrics(img1, img2, ssim_module):
    """Compute L1 loss and SSIM between two images.
    
    Args:
        img1: (B, C, H, W) First image
        img2: (B, C, H, W) Second image
        ssim_module: SSIM module instance (reused for efficiency)
    
    Returns:
        l1_loss: Scalar L1 loss
        ssim_val: Scalar SSIM value (averaged over spatial dimensions)
    """
    l1_loss = F.l1_loss(img1, img2).item()
    
    # SSIM expects (B, C, H, W) format
    # SSIM returns (B, C, H, W) - this is actually the SSIM loss (1 - SSIM) / 2
    # So we need to convert back: SSIM = 1 - 2 * ssim_loss
    ssim_loss_map = ssim_module(img1, img2)  # Returns (B, C, H, W) with values in [0, 1]
    ssim_loss = ssim_loss_map.mean().item()  # Average SSIM loss
    ssim_val = 1.0 - 2.0 * ssim_loss  # Convert back to SSIM (ranges from -1 to 1, typically 0 to 1)
    
    return l1_loss, ssim_val


def visualize_results(target, corrupted, aligned, alpha_map, save_path):
    """Create a visualization grid showing all results."""
    # Convert tensors to numpy for visualization
    def tensor_to_np_img(t):
        """Convert image tensor to numpy array in (H, W, C) format, uint8 [0, 255]"""
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach()
            if len(t.shape) == 4:
                t = t[0]  # Take first batch: (B, C, H, W) -> (C, H, W)
            
            # Handle (C, H, W) -> (H, W, C)
            if len(t.shape) == 3:
                if t.shape[0] in [1, 3]:
                    t = t.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    if t.shape[2] == 1:
                        t = t.squeeze(2)  # (H, W, 1) -> (H, W)
            
            t = t.numpy()
            
            # Normalize to [0, 255]
            if t.dtype != np.uint8:
                # Clamp to [0, 1] if needed, then scale to [0, 255]
                if t.max() > 1.0 or t.min() < 0.0:
                    t = np.clip(t, 0, 1)
                t = (t * 255).astype(np.uint8)
        return t
    
    def tensor_to_np_alpha(t):
        """Convert alpha map tensor to numpy array in (H, W) format"""
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach()
            if len(t.shape) == 4:
                t = t[0]  # (B, 1, H, W) -> (1, H, W)
            if len(t.shape) == 3:
                if t.shape[0] == 1:
                    t = t[0]  # (1, H, W) -> (H, W)
                elif t.shape[2] == 1:
                    t = t[:, :, 0]  # (H, W, 1) -> (H, W)
            t = t.numpy()
        return t
    
    # Convert images to numpy
    target_np = tensor_to_np_img(target)
    corrupted_np = tensor_to_np_img(corrupted)
    aligned_np = tensor_to_np_img(aligned)
    alpha_np = tensor_to_np_alpha(alpha_map)
    
    # Ensure all images are RGB (H, W, 3)
    if len(target_np.shape) == 2:
        target_np = cv2.cvtColor(target_np, cv2.COLOR_GRAY2RGB)
    elif len(target_np.shape) == 3 and target_np.shape[2] == 1:
        target_np = cv2.cvtColor(target_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if len(corrupted_np.shape) == 2:
        corrupted_np = cv2.cvtColor(corrupted_np, cv2.COLOR_GRAY2RGB)
    elif len(corrupted_np.shape) == 3 and corrupted_np.shape[2] == 1:
        corrupted_np = cv2.cvtColor(corrupted_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if len(aligned_np.shape) == 2:
        aligned_np = cv2.cvtColor(aligned_np, cv2.COLOR_GRAY2RGB)
    elif len(aligned_np.shape) == 3 and aligned_np.shape[2] == 1:
        aligned_np = cv2.cvtColor(aligned_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Ensure all images have same shape
    H, W = target_np.shape[:2]
    if corrupted_np.shape[:2] != (H, W):
        corrupted_np = cv2.resize(corrupted_np, (W, H))
    if aligned_np.shape[:2] != (H, W):
        aligned_np = cv2.resize(aligned_np, (W, H))
    if alpha_np.shape != (H, W):
        alpha_np = cv2.resize(alpha_np, (W, H))
    
    # Normalize alpha map for visualization
    alpha_min, alpha_max = alpha_np.min(), alpha_np.max()
    print(f"Debug - Alpha map range: [{alpha_min:.4f}, {alpha_max:.4f}]")
    if alpha_max > alpha_min:
        alpha_vis = ((alpha_np - alpha_min) / (alpha_max - alpha_min) * 255).astype(np.uint8)
    else:
        alpha_vis = np.zeros_like(alpha_np, dtype=np.uint8)
    # Apply viridis colormap (purple -> blue -> green -> yellow)
    alpha_vis = cv2.applyColorMap(alpha_vis, cv2.COLORMAP_VIRIDIS)
    # Convert BGR to RGB
    alpha_vis = cv2.cvtColor(alpha_vis, cv2.COLOR_BGR2RGB)
    
    # Compute error maps: |Target - Other|
    # Convert to float and normalize to [0, 1] for proper error computation
    target_float = target_np.astype(np.float32) / 255.0
    corrupted_float = corrupted_np.astype(np.float32) / 255.0
    aligned_float = aligned_np.astype(np.float32) / 255.0
    
    # Error Before: |Target - Corrupted Source| (per pixel, per channel)
    error_before = np.abs(target_float - corrupted_float)  # (H, W, 3)
    # Error After: |Target - Aligned Output| (per pixel, per channel)
    error_after = np.abs(target_float - aligned_float)  # (H, W, 3)
    
    # Average across RGB channels to get single channel error
    error_before = error_before.mean(axis=2)  # (H, W) in [0, 1]
    error_after = error_after.mean(axis=2)  # (H, W) in [0, 1]
    
    print(f"Debug - Error stats:")
    print(f"  Error Before: min={error_before.min():.4f}, max={error_before.max():.4f}, mean={error_before.mean():.4f}, 95th={np.percentile(error_before, 95):.4f}")
    print(f"  Error After:  min={error_after.min():.4f}, max={error_after.max():.4f}, mean={error_after.mean():.4f}, 95th={np.percentile(error_after, 95):.4f}")
    
    # Normalize error maps using same scale for fair comparison
    # Use 95th percentile to avoid outliers affecting visualization
    max_error = max(np.percentile(error_before, 95), np.percentile(error_after, 95))
    print(f"  Using max_error (95th percentile) = {max_error:.4f} for normalization")
    
    if max_error > 0:
        # Normalize to [0, 1] then scale to [0, 255] for colormap
        # Clip values above max_error to 1.0 for visualization
        error_before_norm = np.clip(error_before / max_error, 0, 1) * 255
        error_after_norm = np.clip(error_after / max_error, 0, 1) * 255
        error_before_norm = error_before_norm.astype(np.uint8)
        error_after_norm = error_after_norm.astype(np.uint8)
        
        print(f"  After normalization: before range=[{error_before_norm.min()}, {error_before_norm.max()}], after range=[{error_after_norm.min()}, {error_after_norm.max()}]")
    else:
        error_before_norm = np.zeros_like(error_before, dtype=np.uint8)
        error_after_norm = np.zeros_like(error_after, dtype=np.uint8)
        print("  Warning: max_error is 0, error maps will be all zeros")
    
    # Apply colormap to error maps (HOT: dark=low error, bright=high error)
    # HOT colormap: black -> red -> yellow -> white (low to high error)
    # Ensure we have 2D arrays before applying colormap
    assert len(error_before_norm.shape) == 2, f"Error before shape: {error_before_norm.shape}, expected 2D"
    assert len(error_after_norm.shape) == 2, f"Error after shape: {error_after_norm.shape}, expected 2D"
    
    # Sanity check: Verify error maps are actually errors, not original images
    # Error maps should have different statistics than original images
    print(f"  Sanity check - Error before should be different from images:")
    print(f"    Error mean={error_before.mean():.4f}, Target mean={target_float.mean():.4f}, Corrupted mean={corrupted_float.mean():.4f}")
    
    # Apply HOT colormap (expects uint8 single channel, returns BGR 3-channel)
    # HOT: black (0) -> red -> yellow -> white (255) for high error
    error_before_norm = cv2.applyColorMap(error_before_norm, cv2.COLORMAP_HOT)
    error_after_norm = cv2.applyColorMap(error_after_norm, cv2.COLORMAP_HOT)
    
    # Convert BGR to RGB for display (OpenCV uses BGR, PIL uses RGB)
    error_before_norm = cv2.cvtColor(error_before_norm, cv2.COLOR_BGR2RGB)
    error_after_norm = cv2.cvtColor(error_after_norm, cv2.COLOR_BGR2RGB)
    
    # Final shape check
    print(f"  Final shapes: error_before_norm={error_before_norm.shape}, error_after_norm={error_after_norm.shape}")
    print(f"  Expected: (H={H}, W={W}, 3)")
    
    # Create grid: 2 rows x 3 columns
    # Row 1: Target, Corrupted Source, Aligned Output
    # Row 2: Alpha Map, Error Before (Target vs Corrupted), Error After (Target vs Aligned)
    H, W = target_np.shape[:2]
    grid = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)
    
    # Row 1: Input images
    grid[0:H, 0:W] = target_np
    grid[0:H, W:2*W] = corrupted_np
    grid[0:H, 2*W:3*W] = aligned_np
    
    # Row 2: Analysis
    grid[H:2*H, 0:W] = alpha_vis
    grid[H:2*H, W:2*W] = error_before_norm  # Error: Target vs Corrupted Source
    grid[H:2*H, 2*W:3*W] = error_after_norm  # Error: Target vs Aligned Output
    
    # Save
    Image.fromarray(grid).save(save_path)
    print(f"Saved visualization to {save_path}")


def test_paba_unit():
    """Unit test for PABA module."""
    print("=" * 60)
    print("PABA Unit Test: Physics-Aware Brightness Alignment")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize modules
    paba = LocalAffineAlignment(
        # patch_size=64, 
        patch_size=8, 
        # patch_size=1, 
        eps=1e-6, 
        min_valid_ratio=0.1,
        interp_mode='nearest',
        # interp_mode='bilinear',
    )
    paba = paba.to(device)
    paba.eval()
    
    # Initialize SSIM module (reuse for robustness and efficiency)
    ssim_module = SSIM()
    ssim_module = ssim_module.to(device)
    ssim_module.eval()
    
    # Load a real endoscopic image (or use a test image)
    # For now, create a synthetic test image if no real image is available
    test_image_path = None
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    
    if test_image_path and os.path.exists(test_image_path):
        # Load real image
        print(f"Loading image from: {test_image_path}")
        img = Image.open(test_image_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    else:
        # Create synthetic test image
        print("No image provided, creating synthetic test image...")
        H, W = 256, 320
        # Create a textured image with some structure
        y = np.linspace(0, 1, H)
        x = np.linspace(0, 1, W)
        X, Y = np.meshgrid(x, y)
        # Create some texture
        texture = np.sin(X * 10) * np.cos(Y * 10) * 0.3 + 0.5
        img = np.stack([texture, texture * 0.9, texture * 0.8], axis=0)  # (3, H, W)
        img = torch.from_numpy(img).float().unsqueeze(0)  # (1, 3, H, W)
        img = img.clamp(0, 1)
    
    img = img.to(device)
    B, C, H, W = img.shape
    print(f"Image shape: {B} x {C} x {H} x {W}")
    
    # Create target image (original)
    target_img = img.clone()
    
    # Synthesize corrupted source image with lighting degradation
    print("\nSynthesizing lighting degradation...")
    corrupted_img = synthesize_lighting_degradation(target_img, degradation_type='combined')
    
    # Create optional valid mask (all valid for this test)
    valid_mask = torch.ones(B, 1, H, W, device=device)
    
    # Test the module
    print("\nRunning PABA alignment...")
    with torch.no_grad():
        aligned_img, alpha_map, beta_map = paba(target_img, corrupted_img, valid_mask)
    
    # Debug: Check value ranges
    print(f"Target range: [{target_img.min().item():.4f}, {target_img.max().item():.4f}]")
    print(f"Corrupted range: [{corrupted_img.min().item():.4f}, {corrupted_img.max().item():.4f}]")
    print(f"Aligned range: [{aligned_img.min().item():.4f}, {aligned_img.max().item():.4f}]")
    print(f"Alpha range: [{alpha_map.min().item():.4f}, {alpha_map.max().item():.4f}]")
    
    # Compute metrics
    print("\nComputing metrics...")
    l1_before, ssim_before = compute_metrics(target_img, corrupted_img, ssim_module)
    l1_after, ssim_after = compute_metrics(target_img, aligned_img, ssim_module)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Before alignment:")
    print(f"  L1 Loss: {l1_before:.6f}")
    print(f"  SSIM:    {ssim_before:.6f}")
    print(f"\nAfter alignment:")
    print(f"  L1 Loss: {l1_after:.6f}")
    print(f"  SSIM:    {ssim_after:.6f}")
    print(f"\nImprovement:")
    print(f"  L1 Reduction: {((l1_before - l1_after) / l1_before * 100):.2f}%")
    print(f"  SSIM Gain:    {((ssim_after - ssim_before) / (1 - ssim_before + 1e-6) * 100):.2f}%")
    print("=" * 60)
    
    # Visualize results
    output_dir = "tests/outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "paba_test_results.png")
    
    print(f"\nGenerating visualization...")
    visualize_results(target_img, corrupted_img, aligned_img, alpha_map, save_path)
    
    # Check if alignment improved the metrics
    if l1_after < l1_before and ssim_after > ssim_before:
        print("\n✓ Test PASSED: Alignment improved both L1 and SSIM!")
        return True
    else:
        print("\n✗ Test FAILED: Alignment did not improve metrics.")
        return False


if __name__ == "__main__":
    success = test_paba_unit()
    sys.exit(0 if success else 1)

