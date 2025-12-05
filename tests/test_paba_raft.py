from __future__ import absolute_import, division, print_function

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.paba import LocalAffineAlignment
from loss import SSIM
import datasets


def resize_for_raft(img, target_size=None):
    """Resize image to be divisible by 8 for RAFT.
    
    Args:
        img: PIL Image or numpy array
        target_size: (H, W) target size, or None to auto-resize
    
    Returns:
        resized_img: Resized image
        original_size: (H, W) original size
    """
    if isinstance(img, Image.Image):
        original_size = img.size  # (W, H)
        if target_size is None:
            # Auto-resize to nearest multiple of 8
            W, H = original_size
            W_new = (W // 8) * 8
            H_new = (H // 8) * 8
            target_size = (H_new, W_new)
        else:
            H_new, W_new = target_size
        
        resized = img.resize((W_new, H_new), Image.LANCZOS)
        return resized, original_size
    else:
        # numpy array
        H, W = img.shape[:2]
        original_size = (H, W)
        if target_size is None:
            H_new = (H // 8) * 8
            W_new = (W // 8) * 8
            target_size = (H_new, W_new)
        else:
            H_new, W_new = target_size
        
        resized = cv2.resize(img, (W_new, H_new))
        return resized, original_size


def compute_optical_flow_raft(img1, img2, device):
    """Compute optical flow from img1 to img2 using RAFT.
    
    Args:
        img1: (1, 3, H, W) tensor, target image
        img2: (1, 3, H, W) tensor, source image
        device: torch device
    
    Returns:
        flow: (1, 2, H, W) tensor, optical flow
    """
    try:
        from torchvision.models.optical_flow import raft_large
        from torchvision.models.optical_flow import Raft_Large_Weights
        from torchvision.utils import flow_to_image
    except ImportError:
        raise ImportError("torchvision.models.optical_flow requires torchvision >= 0.13.0")
    
    # Load pretrained RAFT model
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False)
    model = model.to(device)
    model.eval()
    
    # Normalize images for RAFT (expects [0, 1] range)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    with torch.no_grad():
        # RAFT expects images in [0, 1] range
        flow_list = model(img1, img2)
        flow = flow_list[-1]  # Use the final flow prediction
    
    return flow


def warp_image_with_flow(img, flow):
    """Warp image using optical flow.
    
    RAFT flow convention: flow[i,j] = (u, v) means pixel (i,j) in img1 
    corresponds to pixel (i+v, j+u) in img2.
    
    To warp img2 to match img1, we need to sample img2 at locations 
    (i - v, j - u) for each pixel (i,j) in img1.
    
    Args:
        img: (B, C, H, W) tensor, image to warp (img2, source)
        flow: (B, 2, H, W) tensor, optical flow from img1 to img2
              flow[:, 0, :, :] is horizontal (u), flow[:, 1, :, :] is vertical (v)
    
    Returns:
        warped_img: (B, C, H, W) tensor, warped image (img2 warped to match img1)
    """
    B, C, H, W = img.shape
    device = img.device
    
    # Create coordinate grid for img1 (target)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=device),
        torch.arange(0, W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    # grid_x: (H, W) with values [0, 1, ..., W-1]
    # grid_y: (H, W) with values [0, 1, ..., H-1]
    
    # Expand to batch dimension
    grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
    
    # Extract flow components: flow is (B, 2, H, W) where [:, 0, :, :] is u (horizontal) and [:, 1, :, :] is v (vertical)
    flow_u = flow[:, 0, :, :]  # (B, H, W) - horizontal displacement
    flow_v = flow[:, 1, :, :]  # (B, H, W) - vertical displacement
    
    # To warp img2 to img1: sample img2 at (x - u, y - v) for each pixel (x, y) in img1
    sample_x = grid_x - flow_u  # (B, H, W)
    sample_y = grid_y - flow_v  # (B, H, W)
    
    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects coordinates in [-1, 1] where (-1, -1) is top-left and (1, 1) is bottom-right
    sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0  # (B, H, W)
    sample_y_norm = 2.0 * sample_y / (H - 1) - 1.0  # (B, H, W)
    
    # Stack to (B, H, W, 2) for grid_sample
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)  # (B, H, W, 2)
    
    # Warp image: sample img at the computed grid locations
    warped_img = F.grid_sample(
        img, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    
    return warped_img


def compute_metrics(img1, img2, ssim_module):
    """Compute L1 loss and SSIM between two images."""
    l1_loss = F.l1_loss(img1, img2).item()
    
    # SSIM expects (B, C, H, W) format
    ssim_loss_map = ssim_module(img1, img2)
    ssim_loss = ssim_loss_map.mean().item()
    ssim_val = 1.0 - 2.0 * ssim_loss  # Convert back to SSIM
    
    return l1_loss, ssim_val


def visualize_results_with_colorbar(target, warped, aligned, alpha_map, save_path, use_simple_grid=False):
    """Create visualization grid with colorbar for alpha map.
    
    Args:
        target: Target image tensor
        warped: Warped source image tensor
        aligned: Aligned output image tensor
        alpha_map: Alpha map tensor
        save_path: Path to save the visualization
        use_simple_grid: If True, use 2x3 grid layout (like test_paba_unit.py).
                         If False, use 2x4 matplotlib subplots with histogram and stats.
    """
    # Convert tensors to numpy
    def tensor_to_np_v0(t):
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach()
            if len(t.shape) == 4:
                t = t[0]  # Take first batch
            if len(t.shape) == 3 and t.shape[0] == 3:
                t = t.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            t = t.numpy()
            
            # Handle different value ranges
            # If values are in [0, 1], scale to [0, 255]
            if t.max() <= 1.0 and t.min() >= 0.0:
                t = (t * 255).astype(np.uint8)
            # If values are already in [0, 255], just clip and convert
            elif t.max() <= 255.0 and t.min() >= 0.0:
                t = np.clip(t, 0, 255).astype(np.uint8)
            # If values are outside [0, 1] or [0, 255], normalize first
            else:
                # Normalize to [0, 1] then scale to [0, 255]
                t_min, t_max = t.min(), t.max()
                if t_max > t_min:
                    t = (t - t_min) / (t_max - t_min)
                else:
                    t = np.zeros_like(t)
                t = (t * 255).astype(np.uint8)
        return t
    

    def tensor_to_np(t):
        """
        Robustly convert tensor to numpy image for visualization.
        Handles dimension reordering and strictly clips values to [0, 1].
        """
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach()
            
            # Handle Batch Dim: (B, C, H, W) -> (C, H, W)
            if len(t.shape) == 4:
                t = t[0]  
            
            # Handle Channel Dim: (C, H, W) -> (H, W, C) for RGB/Gray
            if len(t.shape) == 3:
                # Check if it's (C, H, W) where C is 1 or 3
                if t.shape[0] == 1 or t.shape[0] == 3:
                    t = t.permute(1, 2, 0)
                    # If grayscale (H, W, 1), squeeze to (H, W) for cleaner processing
                    if t.shape[2] == 1:
                        t = t.squeeze(2)

            t = t.numpy()

        # ROBUST CONVERSION
        # 1. Clip first: Force mathematically valid HDR values into displayable LDR range
        t = np.clip(t, 0, 1)
        
        # 2. Scale to 8-bit integer
        t = (t * 255).astype(np.uint8)
        
        return t
    target_np = tensor_to_np(target)
    warped_np = tensor_to_np(warped)
    aligned_np = tensor_to_np(aligned)
    
    # Handle alpha map
    if isinstance(alpha_map, torch.Tensor):
        alpha_np = alpha_map.cpu().detach()
        if len(alpha_map.shape) == 4:
            alpha_np = alpha_np[0]  # (B, 1, H, W) -> (1, H, W)
        if len(alpha_np.shape) == 3 and alpha_np.shape[0] == 1:
            alpha_np = alpha_np[0]  # (1, H, W) -> (H, W)
        alpha_np = alpha_np.numpy()
    else:
        alpha_np = np.array(alpha_map)
        if len(alpha_np.shape) == 3 and alpha_np.shape[0] == 1:
            alpha_np = alpha_np[0]
    
    # Ensure all images are RGB
    if len(target_np.shape) == 2:
        target_np = cv2.cvtColor(target_np, cv2.COLOR_GRAY2RGB)
    elif len(target_np.shape) == 3 and target_np.shape[2] == 1:
        target_np = cv2.cvtColor(target_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if len(warped_np.shape) == 2:
        warped_np = cv2.cvtColor(warped_np, cv2.COLOR_GRAY2RGB)
    elif len(warped_np.shape) == 3 and warped_np.shape[2] == 1:
        warped_np = cv2.cvtColor(warped_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if len(aligned_np.shape) == 2:
        aligned_np = cv2.cvtColor(aligned_np, cv2.COLOR_GRAY2RGB)
    elif len(aligned_np.shape) == 3 and aligned_np.shape[2] == 1:
        aligned_np = cv2.cvtColor(aligned_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Resize to same size if needed
    H, W = target_np.shape[:2]
    if warped_np.shape[:2] != (H, W):
        warped_np = cv2.resize(warped_np, (W, H))
    if aligned_np.shape[:2] != (H, W):
        aligned_np = cv2.resize(aligned_np, (W, H))
    if alpha_np.shape != (H, W):
        alpha_np = cv2.resize(alpha_np, (W, H))
    
    if use_simple_grid:
        # Use simple 2x3 grid layout (like test_paba_unit.py)
        # Row 1: Target, Warped Source, Aligned Output
        # Row 2: Alpha Map, Error Before, Error After
        
        # Normalize alpha map for visualization
        alpha_min, alpha_max = alpha_np.min(), alpha_np.max()
        if alpha_max > alpha_min:
            alpha_vis = ((alpha_np - alpha_min) / (alpha_max - alpha_min) * 255).astype(np.uint8)
        else:
            alpha_vis = np.zeros_like(alpha_np, dtype=np.uint8)
        alpha_vis = cv2.applyColorMap(alpha_vis, cv2.COLORMAP_VIRIDIS)
        alpha_vis = cv2.cvtColor(alpha_vis, cv2.COLOR_BGR2RGB)
        
        # Compute error maps
        error_warped = np.abs(target_np.astype(float) - warped_np.astype(float))
        error_warped = error_warped.mean(axis=2)  # Average across channels
        error_aligned = np.abs(target_np.astype(float) - aligned_np.astype(float))
        error_aligned = error_aligned.mean(axis=2)
        
        # Normalize error maps using same scale
        max_error = max(np.percentile(error_warped, 95), np.percentile(error_aligned, 95))
        if max_error > 0:
            error_warped_norm = np.clip(error_warped / max_error, 0, 1) * 255
            error_aligned_norm = np.clip(error_aligned / max_error, 0, 1) * 255
            error_warped_norm = error_warped_norm.astype(np.uint8)
            error_aligned_norm = error_aligned_norm.astype(np.uint8)
        else:
            error_warped_norm = np.zeros_like(error_warped, dtype=np.uint8)
            error_aligned_norm = np.zeros_like(error_aligned, dtype=np.uint8)
        
        # Apply HOT colormap
        error_warped_colored = cv2.applyColorMap(error_warped_norm, cv2.COLORMAP_HOT)
        error_warped_colored = cv2.cvtColor(error_warped_colored, cv2.COLOR_BGR2RGB)
        error_aligned_colored = cv2.applyColorMap(error_aligned_norm, cv2.COLORMAP_HOT)
        error_aligned_colored = cv2.cvtColor(error_aligned_colored, cv2.COLOR_BGR2RGB)
        
        # Create grid: 2 rows x 3 columns
        grid = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)
        
        # Row 1: Input images
        grid[0:H, 0:W] = target_np
        grid[0:H, W:2*W] = warped_np
        grid[0:H, 2*W:3*W] = aligned_np
        
        # Row 2: Analysis
        grid[H:2*H, 0:W] = alpha_vis
        grid[H:2*H, W:2*W] = error_warped_colored  # Error: Target vs Warped
        grid[H:2*H, 2*W:3*W] = error_aligned_colored  # Error: Target vs Aligned
        
        # Save
        Image.fromarray(grid).save(save_path)
        print(f"Saved visualization to {save_path}")
        return
    
    # Create figure with subplots (original 2x4 layout)
    fig = plt.figure(figsize=(16, 8))
    
    # Row 1: Images
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(target_np)
    ax1.set_title('Target (Frame 1)', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(warped_np)
    ax2.set_title('Warped Source (Frame 2)', fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(aligned_np)
    ax3.set_title('Aligned Output', fontsize=12)
    ax3.axis('off')
    
    # Alpha map with colorbar
    ax4 = plt.subplot(2, 4, 4)
    im = ax4.imshow(alpha_np, cmap='viridis', interpolation='nearest')
    ax4.set_title('Alpha Map (Correction Strength)', fontsize=12)
    ax4.axis('off')
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Alpha Value', rotation=270, labelpad=15)
    
    # Row 2: Error maps
    # Error: Target vs Warped
    error_warped = np.abs(target_np.astype(float) - warped_np.astype(float))
    error_warped = error_warped.mean(axis=2)  # Average across channels
    max_error = error_warped.max()
    if max_error > 0:
        error_warped_norm = (error_warped / max_error * 255).astype(np.uint8)
    else:
        error_warped_norm = np.zeros_like(error_warped, dtype=np.uint8)
    error_warped_colored = cv2.applyColorMap(error_warped_norm, cv2.COLORMAP_HOT)
    error_warped_colored = cv2.cvtColor(error_warped_colored, cv2.COLOR_BGR2RGB)
    
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(error_warped_colored)
    ax5.set_title('Error: Target vs Warped', fontsize=12)
    ax5.axis('off')
    
    # Error: Target vs Aligned
    error_aligned = np.abs(target_np.astype(float) - aligned_np.astype(float))
    error_aligned = error_aligned.mean(axis=2)
    if max_error > 0:
        error_aligned_norm = (error_aligned / max_error * 255).astype(np.uint8)
    else:
        error_aligned_norm = np.zeros_like(error_aligned, dtype=np.uint8)
    error_aligned_colored = cv2.applyColorMap(error_aligned_norm, cv2.COLORMAP_HOT)
    error_aligned_colored = cv2.cvtColor(error_aligned_colored, cv2.COLOR_BGR2RGB)
    
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(error_aligned_colored)
    ax6.set_title('Error: Target vs Aligned', fontsize=12)
    ax6.axis('off')
    
    # Alpha map statistics
    ax7 = plt.subplot(2, 4, 7)
    ax7.hist(alpha_np.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax7.set_title('Alpha Map Histogram', fontsize=12)
    ax7.set_xlabel('Alpha Value')
    ax7.set_ylabel('Frequency')
    ax7.grid(True, alpha=0.3)
    
    # Text summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    alpha_mean = alpha_np.mean()
    alpha_std = alpha_np.std()
    alpha_min = alpha_np.min()
    alpha_max = alpha_np.max()
    
    text_str = f"Alpha Map Statistics:\n"
    text_str += f"Mean: {alpha_mean:.4f}\n"
    text_str += f"Std:  {alpha_std:.4f}\n"
    text_str += f"Min:  {alpha_min:.4f}\n"
    text_str += f"Max:  {alpha_max:.4f}\n\n"
    text_str += f"Interpretation:\n"
    if alpha_std < 0.1:
        text_str += "Low variation →\nLikely correcting\nillumination (good)"
    else:
        text_str += "High variation →\nMay be correcting\ntexture (check!)"
    
    ax8.text(0.1, 0.5, text_str, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def test_paba_raft():
    """Real-world validation test using RAFT optical flow."""
    print("=" * 60)
    print("PABA Real-World Validation: RAFT Optical Flow")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize PABA module
    paba = LocalAffineAlignment(
        # patch_size=24, 
        patch_size=12, 
        # patch_size=8, 
        # patch_size=4, 
        # patch_size=1, 
        eps=1e-6, 
        min_valid_ratio=0.1,
        # interp_mode='nearest',
        interp_mode='bilinear',
    )
    paba = paba.to(device)
    paba.eval()
    
    # Initialize SSIM module
    ssim_module = SSIM()
    ssim_module = ssim_module.to(device)
    ssim_module.eval()
    
    # Load two consecutive frames
    print("\nLoading consecutive frames...")
    
    # Option 1: Load from dataset
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # Load from file paths
        frame_0_path = sys.argv[1]
        if len(sys.argv) > 2:
            frame_1_path = sys.argv[2]
        else:
            # Assume consecutive frame (increment number in filename)
            import re
            base_path = os.path.dirname(frame_0_path)
            frame_name = os.path.basename(frame_0_path)
            # Try to extract frame number
            match = re.search(r'(\d+)(\D*\.\w+)$', frame_name)
            if match:
                frame_num = int(match.group(1))
                suffix = match.group(2)
                frame_1_name = f"{frame_num + 1:06d}{suffix}"
                frame_1_path = os.path.join(base_path, frame_1_name)
                if not os.path.exists(frame_1_path):
                    print(f"Warning: {frame_1_path} not found, trying alternative...")
                    frame_1_path = frame_0_path  # Fallback
            else:
                frame_1_path = frame_0_path
        
        print(f"Loading Frame 0: {frame_0_path}")
        print(f"Loading Frame 1: {frame_1_path}")
        
        frame_0_pil = Image.open(frame_0_path).convert('RGB')
        frame_1_pil = Image.open(frame_1_path).convert('RGB')
    else:
        # Option 2: Load from dataset
        from options import MonodepthOptions
        opt = MonodepthOptions()
        opt.frame_ids = [0, 1]  # Consecutive frames
        opt.height = 256
        opt.width = 320
        
        if len(sys.argv) > 1:
            opt.data_path = sys.argv[1]
        else:
            opt.data_path = "/mnt/cluster/datasets/SCARED/training/"
        
        opt.dataset = "endovis"
        opt.is_train = False
        
        # Create dataset
        dataset = datasets.SCAREDRAWDataset(
            opt.data_path, opt.filenames, opt.height, opt.width,
            opt.frame_ids, 4, is_train=opt.is_train, img_ext=opt.img_ext
        )
        
        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            return False
        
        print(f"Dataset size: {len(dataset)}")
        sample_idx = 0
        inputs = dataset[sample_idx]
        
        # Extract frames
        frame_0 = inputs[("color", 0, 0)]  # Target frame
        frame_1 = inputs[("color", 1, 0)]  # Source frame (next frame)
        
        # Convert to PIL
        def tensor_to_pil(t):
            t = t.squeeze().cpu()
            if len(t.shape) == 3:
                t = t.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            t = t.numpy()
            if t.max() <= 1.0:
                t = (t * 255).astype(np.uint8)
            else:
                t = np.clip(t, 0, 255).astype(np.uint8)
            return Image.fromarray(t)
        
        frame_0_pil = tensor_to_pil(frame_0)
        frame_1_pil = tensor_to_pil(frame_1)
    
    # Resize for RAFT (divisible by 8)
    print("\nResizing images for RAFT (must be divisible by 8)...")
    frame_0_resized, orig_size_0 = resize_for_raft(frame_0_pil, target_size=(256, 320))
    frame_1_resized, orig_size_1 = resize_for_raft(frame_1_pil, target_size=(256, 320))
    
    print(f"Resized to: {frame_0_resized.size} (W x H)")
    
    # Convert to tensors
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    frame_0_tensor = to_tensor(frame_0_resized).unsqueeze(0).to(device)  # (1, 3, H, W)
    frame_1_tensor = to_tensor(frame_1_resized).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    print(f"Frame 0 shape: {frame_0_tensor.shape}")
    print(f"Frame 1 shape: {frame_1_tensor.shape}")
    
    # Compute optical flow
    print("\nComputing optical flow with RAFT...")
    try:
        flow = compute_optical_flow_raft(frame_0_tensor, frame_1_tensor, device)
        print(f"Flow shape: {flow.shape}")
    except Exception as e:
        print(f"Error computing optical flow: {e}")
        print("Falling back to identity flow (no warping)")
        B, C, H, W = frame_1_tensor.shape
        flow = torch.zeros(B, 2, H, W, device=device)
    
    # Warp source frame to target
    print("\nWarping source frame...")
    frame_1_warped = warp_image_with_flow(frame_1_tensor, flow)
    
    # Create valid mask (all valid for now, can be improved to mask black borders)
    B, C, H, W = frame_0_tensor.shape
    valid_mask = torch.ones(B, 1, H, W, device=device)
    
    # Apply PABA
    print("\nApplying PABA alignment...")
    print(f"  Target range: [{frame_0_tensor.min():.4f}, {frame_0_tensor.max():.4f}]")
    print(f"  Warped range: [{frame_1_warped.min():.4f}, {frame_1_warped.max():.4f}]")
    
    with torch.no_grad():
        aligned_img, alpha_map, beta_map = paba(frame_0_tensor, frame_1_warped, valid_mask)
    
    print(f"  Aligned range: [{aligned_img.min():.4f}, {aligned_img.max():.4f}]")
    print(f"  Alpha range: [{alpha_map.min():.4f}, {alpha_map.max():.4f}]")
    print(f"  Beta range: [{beta_map.min():.4f}, {beta_map.max():.4f}]")
    
    # Clamp aligned image to valid range [0, 1] for visualization
    # (PABA can produce values outside [0, 1] which is fine for training but needs clamping for vis)
    aligned_img_vis = torch.clamp(aligned_img, 0, 1)
    
    # Compute metrics (use clamped version for fair comparison)
    print("\nComputing metrics...")
    l1_warped, ssim_warped = compute_metrics(frame_0_tensor, frame_1_warped, ssim_module)
    l1_aligned, ssim_aligned = compute_metrics(frame_0_tensor, aligned_img_vis, ssim_module)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Target vs Warped Source:")
    print(f"  L1 Loss: {l1_warped:.6f}")
    print(f"  SSIM:    {ssim_warped:.6f}")
    print(f"\nTarget vs Aligned Output:")
    print(f"  L1 Loss: {l1_aligned:.6f}")
    print(f"  SSIM:    {ssim_aligned:.6f}")
    print(f"\nImprovement:")
    print(f"  L1 Reduction: {((l1_warped - l1_aligned) / l1_warped * 100):.2f}%")
    print(f"  SSIM Gain:    {((ssim_aligned - ssim_warped) / (1 - ssim_warped + 1e-6) * 100):.2f}%")
    print("=" * 60)
    
    # Visualize results
    output_dir = "tests/outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "paba_raft_results.png")
    
    print(f"\nGenerating visualization...")
    # Use simple 2x3 grid layout (like test_paba_unit.py)
    # Set to False to use the detailed 2x4 matplotlib layout with histogram and stats
    use_simple_grid = True
    visualize_results_with_colorbar(
        frame_0_tensor, frame_1_warped, aligned_img_vis, alpha_map, save_path,
        use_simple_grid=use_simple_grid
    )
    
    # Check if alignment improved
    if l1_aligned < l1_warped and ssim_aligned > ssim_warped:
        print("\n✓ Test PASSED: PABA improved alignment after optical flow warping!")
        return True
    else:
        print("\n✗ Test FAILED: PABA did not improve metrics.")
        return False


if __name__ == "__main__":
    success = test_paba_raft()
    sys.exit(0 if success else 1)

