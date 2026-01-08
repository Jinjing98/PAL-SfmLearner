import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import sys
import os
import random
from pathlib import Path
from glob import glob
from pytorch_wavelets import DWTForward, DWTInverse

# Setup path for depth_anything_3 imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))
from depth_anything_3.api import DepthAnything3

# -------------------- 工具函数 --------------------
def load_rgb_image(path, size=(256,256)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32)/255.0
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # Bx3xHxW

# Global model cache
_depth_model = None

def _get_depth_model(device):
    """Lazy initialization of DepthAnything3 (ViT base) with pretrained weights."""
    global _depth_model
    if _depth_model is None:
        # Load pretrained weights from Hugging Face Hub
        _depth_model = DepthAnything3.from_pretrained("depth-anything/da3-base").to(device).eval()
    return _depth_model.to(device)

def pseudo_depth(rgb_tensor):
    """Estimate depth using DepthAnything3 (ViT base) official API."""
    device = rgb_tensor.device
    model = _get_depth_model(device)
    B, _, H, W = rgb_tensor.shape
    
    # Convert tensor to numpy arrays: (B, 3, H, W) -> list of (H, W, 3) uint8
    images = [(rgb_tensor[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) 
              for b in range(B)]
    
    # Use official inference API (handles preprocessing automatically)
    prediction = model.inference(
        images,
        process_res=max(H, W),  # Preserve original resolution
        process_res_method="upper_bound_resize"
    )
    
    # Convert depth from numpy to tensor: (N, H, W) -> (B, 1, H, W)
    depth_np = prediction.depth  # (N, H, W)
    depth = torch.from_numpy(depth_np).to(device).unsqueeze(1)  # (N, 1, H, W)
    
    # # Resize if needed to match exact input resolution
    # if depth.shape[2:] != (H, W):
    #     depth = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)
    
    return depth

# DWT / iDWT
def dwt2d(x, wave='bior3.3',):
    coeffs = pywt.dwt2(x.squeeze(0).squeeze(0).cpu().numpy(), wave)
    LL, (LH, HL, HH) = coeffs
    return {'LL': torch.from_numpy(LL).unsqueeze(0).unsqueeze(0).to(x.device),
            'LH': torch.from_numpy(LH).unsqueeze(0).unsqueeze(0).to(x.device),
            'HL': torch.from_numpy(HL).unsqueeze(0).unsqueeze(0).to(x.device),
            'HH': torch.from_numpy(HH).unsqueeze(0).unsqueeze(0).to(x.device)}

def idwt2d(coeffs, wave='bior3.3'):
    """Inverse DWT with size matching to ensure all coefficients have the same size."""
    LL = coeffs['LL'].squeeze(0).squeeze(0).cpu().numpy()
    LH = coeffs['LH'].squeeze(0).squeeze(0).cpu().numpy()
    HL = coeffs['HL'].squeeze(0).squeeze(0).cpu().numpy()
    HH = coeffs['HH'].squeeze(0).squeeze(0).cpu().numpy()
    
    # Ensure all coefficients have the same size (required by pywt.idwt2)
    target_h, target_w = LL.shape
    LH = LH[:target_h, :target_w] if LH.shape != (target_h, target_w) else LH
    HL = HL[:target_h, :target_w] if HL.shape != (target_h, target_w) else HL
    HH = HH[:target_h, :target_w] if HH.shape != (target_h, target_w) else HH
    
    arr = pywt.idwt2((LL, (LH, HL, HH)), wave)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(coeffs['LL'].device)

# from torch_wavelets import DWTForward, DWTInverse
def idwt2d_torch(coeffs, wave='bior3.3'):
    ifm = DWTInverse(wave=wave, mode='zero').to(coeffs['LL'].device)

    LL = coeffs['LL']
    LH, HL, HH = coeffs['LH'], coeffs['HL'], coeffs['HH']

    # Repack as separable HF: (B, C, 3, H, W)
    hf = torch.stack([LH, HL, HH], dim=2)

    recon = ifm((LL, [hf]))
    return recon


def dwt2d_torch(x, wave='bior3.3'):
    xfm = DWTForward(J=1, wave=wave, mode='zero').to(x.device)
    LL, high = xfm(x)

    hf = high[0]

    # Case 1: separable HF -> (B, C, 3, H, W)
    if isinstance(hf, torch.Tensor) and hf.dim() == 5:
        LH = hf[:, :, 0]
        HL = hf[:, :, 1]
        HH = hf[:, :, 2]

    # Case 2: channel-packed -> (B, 3C, H, W)
    elif isinstance(hf, torch.Tensor) and hf.dim() == 4:
        assert 0, "Invalid hf type"
        B, C3, H, W = hf.shape
        C = C3 // 3
        LH = hf[:, 0*C:1*C]
        HL = hf[:, 1*C:2*C]
        HH = hf[:, 2*C:3*C]

    # Case 3: tuple -> (LH, HL, HH)
    else:
        assert 0, "Invalid hf type"
        LH, HL, HH = hf

    return {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}

 
def dwt_n_layer(depth_img, n_layers, wave='bior3.3', use_torch_wavelets=False):
    """Apply N-layer DWT, zero out the deepest LL, and reconstruct."""
    stack = []
    cur = depth_img
    
    dwt2d_func = dwt2d_torch if use_torch_wavelets else dwt2d
    idwt2d_func = idwt2d_torch if use_torch_wavelets else idwt2d
    
    # Forward: decompose N layers
    for _ in range(n_layers):
        c = dwt2d_func(cur, wave)
        stack.append(c)
        cur = c['LL']
    
    # Zero out the deepest LL
    stack[-1]['LL'] = torch.zeros_like(stack[-1]['LL'])
    
    # Backward: reconstruct from deepest to shallowest
    rec = None
    for c in reversed(stack):
        if rec is None:
            rec = idwt2d_func(c)
        else:
            # Ensure rec matches the expected size for this level
            target_h, target_w = c['LL'].shape[2], c['LL'].shape[3]
            if rec.shape[2:] != (target_h, target_w):
                rec = F.interpolate(rec, size=(target_h, target_w), mode='bilinear', align_corners=False)
            c_up = {'LL': rec, 'LH': c['LH'], 'HL': c['HL'], 'HH': c['HH']}
            rec = idwt2d_func(c_up)
    
    # Ensure output matches input size
    target_h, target_w = depth_img.shape[2], depth_img.shape[3]
    if rec.shape[2:] != (target_h, target_w):
        rec = F.interpolate(rec, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    return rec


def scale_invariant_normalize(depth):
    """
    Scale-invariant normalization for depth maps.
    Ensures high-frequency distillation works with scale-ambiguous monocular depth.
    
    Args:
        depth: Depth tensor of shape (B, C, H, W) or (B, H, W)
    
    Returns:
        Normalized depth tensor of same shape
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    
    # Compute mean and std across spatial dimensions
    depth_mean = depth.mean(dim=[2, 3], keepdim=True)
    depth_std = depth.std(dim=[2, 3], keepdim=True)
    
    # Normalize
    depth_norm = (depth - depth_mean) / (depth_std + 1e-6)
    
    return depth_norm


def hf_mask_map(hf_map, grad_threshold=0.01):
    """
    Create mask for high-frequency map to remove low-gradient regions and reflection noise.
    
    Args:
        hf_map: High-frequency map tensor of shape (B, C, H, W) or (B, H, W)
        grad_threshold: Gradient threshold (default: 0.01)
    
    Returns:
        Tuple of (masked high-frequency map, mask) of same shape
    """
    if hf_map.dim() == 3:
        hf_map = hf_map.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    
    # Compute gradients
    dy = hf_map[:, :, 1:, :] - hf_map[:, :, :-1, :]
    dx = hf_map[:, :, :, 1:] - hf_map[:, :, :, :-1]
    grad = torch.sqrt(F.pad(dx, (0, 1, 0, 0))**2 + F.pad(dy, (0, 0, 0, 1))**2 + 1e-8)
    
    # Create mask: keep regions with gradient > threshold
    mask = (grad > grad_threshold).float()
    
    return hf_map * mask, mask

# -------------------- Case 1: Top - N层 DWT --------------------
def top_case(depth_img, wave='bior3.3'):
    """2-layer DWT (for backward compatibility)."""
    return dwt_n_layer(depth_img, 2, wave)#, use_torch_wavelets=True)

# -------------------- Case 2: Middle - 深层 DWT (3层) --------------------
def middle_case(depth_img):
    c1 = dwt2d(depth_img)
    c2 = dwt2d(c1['LL'])
    c3 = dwt2d(c2['LL'])
    # 最底层 LL3置0
    c3_hf = {'LL': torch.zeros_like(c3['LL']), 'LH': c3['LH'], 'HL': c3['HL'], 'HH': c3['HH']}
    recon3 = idwt2d(c3_hf)
    
    # Ensure recon3 matches c2['LL'] size
    target_h, target_w = c2['LL'].shape[2], c2['LL'].shape[3]
    if recon3.shape[2:] != (target_h, target_w):
        recon3 = F.interpolate(recon3, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # layer2重构
    c2_up = {'LL': recon3, 'LH': c2['LH'], 'HL': c2['HL'], 'HH': c2['HH']}
    recon2 = idwt2d(c2_up)
    
    # Ensure recon2 matches c1['LL'] size
    target_h, target_w = c1['LL'].shape[2], c1['LL'].shape[3]
    if recon2.shape[2:] != (target_h, target_w):
        recon2 = F.interpolate(recon2, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # layer1重构
    c1_up = {'LL': recon2, 'LH': c1['LH'], 'HL': c1['HL'], 'HH': c1['HH']}
    out = idwt2d(c1_up)
    
    # Ensure output matches input size
    target_h, target_w = depth_img.shape[2], depth_img.shape[3]
    if out.shape[2:] != (target_h, target_w):
        out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    return out

# -------------------- Case 3: Bottom - 局部自适应 DWT --------------------
def bottom_case(depth_img, blk=64, T_sigma=0.01, T_grad=0.02):
    B, C, H, W = depth_img.shape
    out = torch.zeros_like(depth_img)
    
    # gradient
    dy = depth_img[:, :, 1:, :] - depth_img[:, :, :-1, :]
    dx = depth_img[:, :, :, 1:] - depth_img[:, :, :, :-1]
    grad = torch.sqrt(F.pad(dx,(0,1,0,0))**2 + F.pad(dy,(0,0,0,1))**2 + 1e-8)
    
    for y in range(0, H, blk):
        for x in range(0, W, blk):
            # Handle boundary cases
            y_end = min(y + blk, H)
            x_end = min(x + blk, W)
            patch = depth_img[:, :, y:y_end, x:x_end]
            if patch.numel() == 0:
                continue
            # sigma = torch.var(patch)
            sigma = torch.var(patch, dim=[2,3])
            # Handle gradient calculation for boundary patches
            grad_patch = grad[:, :, y:y_end, x:x_end]
            g = torch.mean(grad_patch) if grad_patch.numel() > 0 else 0.0
            if sigma>T_sigma and g>T_grad:
                lvl = 3
            elif sigma>T_sigma or g>T_grad:
                lvl = 2
            else:
                lvl = 1
            cur = patch
            stack=[]
            for _ in range(lvl):
                c=dwt2d(cur)
                stack.append(c)
                cur=c['LL']
            stack[-1]['LL']=torch.zeros_like(stack[-1]['LL'])
            rec=None
            for c in reversed(stack):
                if rec is None:
                    rec=idwt2d(c)
                else:
                    # Ensure rec matches the expected size for this level
                    target_h, target_w = c['LL'].shape[2], c['LL'].shape[3]
                    if rec.shape[2:] != (target_h, target_w):
                        rec = F.interpolate(rec, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    c_up={'LL':rec,'LH':c['LH'],'HL':c['HL'],'HH':c['HH']}
                    rec=idwt2d(c_up)
            
            # Ensure rec matches patch size
            patch_h, patch_w = patch.shape[2], patch.shape[3]
            if rec.shape[2:] != (patch_h, patch_w):
                rec = F.interpolate(rec, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            
            # Handle boundary cases when assigning back
            out[:, :, y:y_end, x:x_end] = rec[:, :, :patch_h, :patch_w]
    return out

# -------------------- 可视化 --------------------
def visualize_cases(image_path, output_path=None):
    """Visualize depth estimation and DWT/ALWT results.
    
    Args:
        image_path: Path to input RGB image
        output_path: Optional output path for saving the figure. If None, saves to "hfd.png"
    """
    rgb = load_rgb_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = rgb.to(device)
    depth = pseudo_depth(rgb)  # 使用DepthAnything3 (ViT base)估计深度

    # Convert RGB to numpy for display
    rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    depth_np = depth.squeeze().cpu().numpy()
    
    # Compute DWT layers 1-5
    dwt_results = []
    for n in range(1, 6):
        dwt_result = dwt_n_layer(depth, n)
        dwt_results.append(dwt_result.squeeze().cpu().numpy())
    
    # Compute ALWT with different patch sizes
    patch_sizes = [32, 48, 64, 96, 128]
    alwt_results = []
    for patch_size in patch_sizes:
        alwt_result = bottom_case(depth, blk=patch_size)
        alwt_results.append(alwt_result.squeeze().cpu().numpy())

    # Create figure with 2 rows
    # Row 1: RGB, Depth, DWT layers 1-5
    # Row 2: ALWT with different patch sizes
    fig = plt.figure(figsize=(20, 8))
    
    # Row 1: RGB + Depth + DWT layers 1-5 (7 plots)
    plt.subplot(2, 7, 1)
    plt.imshow(rgb_np)
    plt.title('RGB Image')
    plt.axis('off')
    
    plt.subplot(2, 7, 2)
    plt.imshow(depth_np, cmap='gray')
    plt.title('Depth (DepthAnything3)')
    plt.axis('off')
    
    for i, (n, dwt_np) in enumerate(zip(range(1, 6), dwt_results)):
        plt.subplot(2, 7, 3 + i)
        plt.imshow(dwt_np, cmap='gray')
        plt.title(f'DWT {n}-layer')
        plt.axis('off')
    
    # Row 2: ALWT with different patch sizes (7 plots, first 2 empty)
    plt.subplot(2, 7, 8)
    plt.axis('off')  # Empty
    
    plt.subplot(2, 7, 9)
    plt.axis('off')  # Empty
    
    for i, (patch_size, alwt_np) in enumerate(zip(patch_sizes, alwt_results)):
        plt.subplot(2, 7, 10 + i)
        plt.imshow(alwt_np, cmap='gray')
        plt.title(f'ALWT (blk={patch_size})')
        plt.axis('off')

    plt.tight_layout()
    if output_path is None:
        output_path = "hfd.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

# -------------------- 运行示例 --------------------
if __name__=="__main__":
    # Base directory containing images
    base_dir = "/mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized"
    
    # Find all PNG images in the directory tree
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob(os.path.join(base_dir, '**', ext), recursive=True))
        all_images.extend(glob(os.path.join(base_dir, '**', ext.upper()), recursive=True))
    
    if not all_images:
        print(f"No images found in {base_dir}")
        # Fallback to specific path
        image_path = "/mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/dataset1/keyframe3/image_02/data/0000000001.png"
        if os.path.exists(image_path):
            visualize_cases(image_path)
    else:
        # Randomly select images (default: 3 images)
        num_images = 3
        selected_images = random.sample(all_images, min(num_images, len(all_images)))
        
        print(f"Found {len(all_images)} images. Randomly selected {len(selected_images)} images:")
        for img_path in selected_images:
            print(f"  - {img_path}")
        
        # Visualize each selected image
        for i, image_path in enumerate(selected_images):
            print(f"\nProcessing image {i+1}/{len(selected_images)}: {os.path.basename(image_path)}")
            try:
                # Save with unique filename
                output_path = f"hfd_{i+1}_{os.path.basename(image_path).rsplit('.', 1)[0]}.png"
                visualize_cases(image_path, output_path=output_path)
                print(f"Saved to {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue