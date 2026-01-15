# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from utils import pose_encoding_to_extri_intri_v2
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

from utils import load_pretrained_weights

# Import LoRA modules
from third_party.EndoDAC.models.backbones.mylora import Linear as LoraLinear
from third_party.EndoDAC.models.backbones.mylora import DVLinear as DVLinear
from third_party.EndoDAC.models.backbones.galora import LoRALayer
import torch.nn.functional as F
import math

def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class LoraLinear_QV(LoraLinear):
    """
    LoRA Linear layer that only applies LoRA to Q and V projections in QKV.
    Assumes out_features = 3 * hidden_dim, where the output is organized as [Q, K, V].
    LoRA is only applied to Q and V parts, K remains unchanged.
    
    Inherits from LoraLinear (mylora.Linear).
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs
    ):
        # out_features must be divisible by 3 (Q, K, V)
        assert out_features % 3 == 0, f"out_features must be divisible by 3 for QKV, got {out_features}"
        
        # Initialize parent with full out_features to create the base Linear layer
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        self.hidden_dim = out_features // 3
        
        # LoRA parameters only for Q and V (2/3 of out_features)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # Only 2 * hidden_dim for Q and V
            self.lora_B = nn.Parameter(self.weight.new_zeros((2 * self.hidden_dim, r)))
            self.scaling = lora_alpha / r
            # Freeze the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            # Base forward: x @ W.T
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # Compute LoRA output for Q and V only: x @ A.T @ B.T
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            
            # Split result into Q, K, V
            # result shape: (..., 3 * hidden_dim)
            q = result[..., :self.hidden_dim]
            k = result[..., self.hidden_dim:2*self.hidden_dim]
            v = result[..., 2*self.hidden_dim:]
            
            # Split lora_out into Q and V parts
            # lora_out shape: (..., 2 * hidden_dim)
            lora_q = lora_out[..., :self.hidden_dim]
            lora_v = lora_out[..., self.hidden_dim:]
            
            # Apply LoRA to Q and V, keep K unchanged
            q = q + lora_q
            v = v + lora_v
            
            # Concatenate back to [Q, K, V]
            result = torch.cat([q, k, v], dim=-1)
            
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class DVLinear_QV(DVLinear):
    """
    DVLoRA Linear layer that only applies LoRA to Q and V projections in QKV.
    Assumes out_features = 3 * hidden_dim, where the output is organized as [Q, K, V].
    LoRA is only applied to Q and V parts, K remains unchanged.
    
    Inherits from DVLinear (mylora.DVLinear).
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs
    ):
        # out_features must be divisible by 3 (Q, K, V)
        assert out_features % 3 == 0, f"out_features must be divisible by 3 for QKV, got {out_features}"
        
        # Initialize parent with full out_features to create the base Linear layer
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        self.hidden_dim = out_features // 3
        
        # DVLoRA parameters only for Q and V (2/3 of out_features)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # Only 2 * hidden_dim for Q and V
            self.lora_B = nn.Parameter(self.weight.new_zeros((2 * self.hidden_dim, r)))
            self.lora_U = nn.Parameter(self.weight.new_zeros(r, 1))
            self.lora_V = nn.Parameter(self.weight.new_zeros(2 * self.hidden_dim, 1))
            self.scaling = lora_alpha / r
            # Freeze the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize A, U, V same way as DVLinear
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_V, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            # Base forward: x @ W.T
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # Compute DVLoRA output for Q and V only: x @ (A*U).T @ (B*V).T
            lora_out = (self.lora_dropout(x) @ (self.lora_A * self.lora_U).T @ (self.lora_B * self.lora_V).T) * self.scaling
            
            # Split result into Q, K, V
            # result shape: (..., 3 * hidden_dim)
            q = result[..., :self.hidden_dim]
            k = result[..., self.hidden_dim:2*self.hidden_dim]
            v = result[..., 2*self.hidden_dim:]
            
            # Split lora_out into Q and V parts
            # lora_out shape: (..., 2 * hidden_dim)
            lora_q = lora_out[..., :self.hidden_dim]
            lora_v = lora_out[..., self.hidden_dim:]
            
            # Apply LoRA to Q and V, keep K unchanged
            q = q + lora_q
            v = v + lora_v
            
            # Concatenate back to [Q, K, V]
            result = torch.cat([q, k, v], dim=-1)
            
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def mark_only_part_as_trainable_v2(
    model: nn.Module, 
    bias: str = 'none', 
    warm_up: bool = True,
    other_trainable: list = None
) -> None:
    """
    Mark only LoRA parameters and other specified parameters as trainable.
    
    Args:
        model: The model to mark parameters for
        bias: Bias handling mode ('none', 'all', 'lora_only')
        warm_up: If True, mark lora_A and lora_B as trainable (for warm-up phase).
                 If False, mark lora_U and lora_V as trainable (for fine-tuning phase).
        other_trainable: List of parameter name substrings that should remain trainable.
                        Default: ['residual_', 'conv_depth_']
    """
    if other_trainable is None:
        other_trainable = ['residual_', 'conv_depth_']
    
    for n, p in model.named_parameters():
        # Check if parameter name contains any of the trainable substrings
        is_trainable = False
        
        if warm_up:
            # Warm-up phase: lora_A and lora_B are trainable
            if 'lora_A' in n or 'lora_B' in n:
                is_trainable = True
        else:
            # Fine-tuning phase: lora_U and lora_V are trainable
            if 'lora_U' in n or 'lora_V' in n:
                is_trainable = True
        
        # Check if parameter name contains any of the other_trainable substrings
        for trainable_substring in other_trainable:
            if trainable_substring in n:
                is_trainable = True
                break
        
        if not is_trainable:
            p.requires_grad = False
    
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError




class EndoDepthAnything3Net(nn.Module):
    """
    EndoDepthAnything3Net is a wrapper for the Depth Anything 3 network. With extended 
    control on dino_resize_hw and ref_view_strategy.

    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None,
                 ref_view_strategy="saddle_balanced",
                 dino_resize_hw=None,
                 lora_type="none",
                 lora_r=4,
                 lora_apply_to_attn=False):
        """
        Initialize EndoDepthAnything3Net with given yaml-initialized configuration.
        
        Args:
            dino_resize_hw: Tuple of (height, width) to resize input images to before backbone.
                           If None, no resizing is performed. If not None, both h and w must be
                           divisible by PATCH_SIZE (14).
            lora_type: Type of LoRA to apply. Options: "none", "lora", "dvlora". Default: "none"
            lora_r: Rank of LoRA. Default: 4
            lora_apply_to_attn: If True, also apply LoRA to attention projection layers (attn.proj).
                               Default: False (only applies to MLP feed-forward layers)
        """
        super().__init__()

        assert ref_view_strategy in ["saddle_balanced", "saddle_sim_range", "first", "middle"], "Invalid reference view strategy"
        self.ref_view_strategy = ref_view_strategy
        
        # Sanity check: if dino_resize_hw is provided, ensure dimensions are divisible by patch size
        if dino_resize_hw is not None:
            h, w = dino_resize_hw
            assert h % self.PATCH_SIZE == 0, \
                f"dino_resize_hw height ({h}) must be divisible by PATCH_SIZE ({self.PATCH_SIZE})"
            assert w % self.PATCH_SIZE == 0, \
                f"dino_resize_hw width ({w}) must be divisible by PATCH_SIZE ({self.PATCH_SIZE})"
        
        self.dino_resize_hw = dino_resize_hw
        
        # LoRA configuration
        assert lora_type in ["none", "lora", "dvlora"], f"Invalid lora_type: {lora_type}. Must be 'none', 'lora', or 'dvlora'"
        assert lora_r > 0, "lora_r must be greater than 0"
        self.lora_type = lora_type
        self.lora_r = lora_r
        self.lora_apply_to_attn = lora_apply_to_attn

        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        
        # Apply LoRA to backbone if specified
        if self.lora_type != "none":
            self._apply_lora_to_backbone()
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))
        
        # # Mark only LoRA parameters as trainable if LoRA is enabled
        # if self.lora_type != "none":
        #     # only mark the LoRA parameters as trainable
        #     # dvlora: warm up, lora_A, lora_B, lora_U, lora_V, residual_, conv_depth_
        #     # lora: warm up, lora_A, lora_B, residual_, conv_depth_
        #     mark_only_part_as_trainable_v2(self.backbone)

    def _apply_lora_to_backbone(self):
        """
        Apply LoRA to the MLP layers in the backbone transformer blocks.
        Optionally also applies LoRA to attention projection layers if lora_apply_to_attn is True.
        Following the pattern from endodac.py
        """
        if not hasattr(self.backbone, 'blocks'):
            assert False, "Backbone doesn't have blocks attribute"
        else:
            # Standard case: backbone has blocks attribute
            for layer_idx, blk in enumerate(self.backbone.blocks):
                if (hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc1') and hasattr(blk.mlp, 'fc2')) \
                    or (hasattr(blk.mlp, 'w12') and hasattr(blk.mlp, 'w3')):
                    if hasattr(blk.mlp, 'fc1') and hasattr(blk.mlp, 'fc2'):
                        # we only apply on the qkv proj layer rather the attn.proj
                        # Only apply attention LoRA to layers 6-11 (0-indexed: 6, 7, 8, 9, 10, 11)
                        self._apply_lora_to_block(blk, layer_idx=layer_idx, 
                                                # lora_on_qkv=self.lora_apply_to_attn and (6 <= layer_idx <= 11), 
                                                
                                                # HARD CODE TO APPLY LORA ON ALL 12 LAYERRS and QV_Only
                                                lora_on_qkv=False,
                                                lora_on_qv=self.lora_apply_to_attn,
                                                
                                                lora_on_proj=False)
                    elif hasattr(blk.mlp, 'w12') and hasattr(blk.mlp, 'w3'):
                        # print(f"Applying LoRA to SwiGLU block of giant model")
                        self._apply_lora_to_block_swiglufused(blk, layer_idx=layer_idx)
                    else:
                        assert False, "blk is not attn blk"
                
                else:
                    assert False, "blk has no mlp or swiglufused"
    
    def _apply_lora_to_block(self, blk, layer_idx=None, lora_on_qkv=False, lora_on_qv=False, lora_on_proj=False):
        """
        Apply LoRA to a single transformer block's MLP layers and optionally attention projection layer.
        
        Args:
            blk: Transformer block with mlp attribute containing fc1 and fc2, and optionally attn.proj
            layer_idx: Layer index (0-indexed) for selective attention LoRA application
            lora_on_qkv: Whether to apply LoRA to attention qkv layer (full QKV)
            lora_on_qv: Whether to apply LoRA only to Q and V in attention qkv layer (partial, K unchanged)
            lora_on_proj: Whether to apply LoRA to attention projection layer
        """
        # Apply LoRA to MLP feed-forward layers
        mlp_in_features = blk.mlp.fc1.in_features
        mlp_hidden_features = blk.mlp.fc1.out_features
        mlp_out_features = blk.mlp.fc2.out_features
        
        if self.lora_type == "dvlora":
            blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.lora_r, lora_alpha=self.lora_r)
            blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.lora_r, lora_alpha=self.lora_r)
        elif self.lora_type == "lora":
            blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.lora_r)
            blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.lora_r)
        
        # Optionally apply LoRA to attention projection layer (only for layers 6-11)
        if lora_on_proj:
            assert 0, 'temporal disabled'
            attn_proj_in_features = blk.attn.proj.in_features
            attn_proj_out_features = blk.attn.proj.out_features
            
            if self.lora_type == "dvlora":
                blk.attn.proj = DVLinear(attn_proj_in_features, attn_proj_out_features, r=self.lora_r, lora_alpha=self.lora_r)
            elif self.lora_type == "lora":
                blk.attn.proj = LoraLinear(attn_proj_in_features, attn_proj_out_features, r=self.lora_r)
        
        if lora_on_qkv:
            assert not lora_on_qv, "lora_on_qkv and lora_on_qv cannot be True at the same time"
            # optionally apply on attn.qkv (full QKV)
            qkv_in = blk.attn.qkv.in_features
            qkv_out = blk.attn.qkv.out_features
            if self.lora_type == "dvlora":
                blk.attn.qkv = DVLinear(qkv_in, qkv_out, r=self.lora_r, lora_alpha=self.lora_r)
            elif self.lora_type == "lora":
                blk.attn.qkv = LoraLinear(qkv_in, qkv_out, r=self.lora_r)
        
        if lora_on_qv:
            assert not lora_on_qkv, "lora_on_qv and lora_on_qkv cannot be True at the same time"
            # optionally apply on attn.qkv, but only on Q and V (K unchanged)
            qkv_in = blk.attn.qkv.in_features
            qkv_out = blk.attn.qkv.out_features
            if self.lora_type == "dvlora":
                blk.attn.qkv = DVLinear_QV(qkv_in, qkv_out, r=self.lora_r, lora_alpha=self.lora_r)
            elif self.lora_type == "lora":
                blk.attn.qkv = LoraLinear_QV(qkv_in, qkv_out, r=self.lora_r)

        print(f"Applied LoRA to attention projection layer{layer_idx} with lora_on_qkv: {lora_on_qkv}, lora_on_qv: {lora_on_qv}, and lora_on_proj: {lora_on_proj}")


    def _apply_lora_to_block_swiglufused(self, blk, layer_idx=None):
        """
        Apply LoRA to a single transformer block's SwiGLU feed-forward layers and optionally attention projection layer.
        
        Args:
            blk: Transformer block with swiglufused attribute containing w12 and w3
            layer_idx: Layer index (0-indexed) for selective attention LoRA application
            apply_attn_lora: Whether to apply LoRA to attention layers (overrides self.lora_apply_to_attn if provided)
        """
        # Apply LoRA to SwiGLU feed-forward layers
        # In SwiGLU: w12 has shape [in_features, 2 * hidden_features], w3 has shape [hidden_features, out_features]
        swiglufused_in_features = blk.mlp.w12.in_features
        swiglufused_w12_out_features = blk.mlp.w12.out_features  # This is 2 * hidden_features (e.g., 8192 for pretrained)
        swiglufused_hidden_features = blk.mlp.w3.in_features  # This is the actual hidden_features (e.g., 4096 for pretrained)
        swiglufused_out_features = blk.mlp.w3.out_features
        if self.lora_type == "dvlora":
            blk.mlp.w12 = DVLinear(swiglufused_in_features, swiglufused_w12_out_features, r=self.lora_r, lora_alpha=self.lora_r)
            blk.mlp.w3 = DVLinear(swiglufused_hidden_features, swiglufused_out_features, r=self.lora_r, lora_alpha=self.lora_r)
        elif self.lora_type == "lora":
            blk.mlp.w12 = LoraLinear(swiglufused_in_features, swiglufused_w12_out_features, r=self.lora_r)
            blk.mlp.w3 = LoraLinear(swiglufused_hidden_features, swiglufused_out_features, r=self.lora_r)
        

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        # ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) 
            intrinsics: Camera intrinsics (B, N, 3, 3) 
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        B, S, C, H_raw, W_raw = x.shape
        
        # Resize input if dino_resize_hw is specified
        if self.dino_resize_hw is not None:
            if H_raw != self.dino_resize_hw[0] or W_raw != self.dino_resize_hw[1]:
                print(f"Resizing input from {H_raw}x{W_raw} to {self.dino_resize_hw[0]}x{self.dino_resize_hw[1]}")
                # resize B S C H W to B S C H_new W_new
                x = torch.nn.functional.interpolate(x.view(B*S, C, H_raw, W_raw), size=self.dino_resize_hw, mode="bilinear", align_corners=True)
                x = x.view(B, S, C, self.dino_resize_hw[0], self.dino_resize_hw[1])

        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=self.ref_view_strategy
        )
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            # H W control the depth from DPT is with spatial dim dino_h_w
            output = self._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)
        
        output = self._process_mono_sky_estimation(output)    

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        return output

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process mono sky estimation."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output
        
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        # DPT has no idea regarding the target size, it is all controlled here.
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            # c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            rot_representation = self.cam_dec.rot_representation if hasattr(self.cam_dec, 'rot_representation') else "quat_xyzw"
            c2w, ixt = pose_encoding_to_extri_intri_v2(pose_enc, (H, W), 
                                                        rot_representation=rot_representation)
            output.extrinsics = affine_inverse(c2w) # c2w -> w2c
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


if __name__ == "__main__":
    # net = EndoDepthAnything3Net(net="endo-da3-base.yaml", head="endo-da3-base.yaml", cam_dec="endo-da3-base.yaml", cam_enc="endo-da3-base.yaml", gs_head="endo-da3-base.yaml", gs_adapter="endo-da3-base.yaml")
    # print(net)

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.utils.io.input_processor import InputProcessor  
    from depth_anything_3.api import DepthAnything3
    from utils.util import set_seed

    # Set seed for reproducible initialization (if not loading pretrained weights)
    set_seed(42)
    
    # Model = create_object(load_config("networks/configs/endo-da3-all.yaml"))

    Model_with_wrapper = create_object(load_config("networks/configs/endo-da3-depth-default.yaml"))
    Model_with_wrapper = create_object(load_config("networks/configs/endo-da3-all-default.yaml"))
    Model_with_wrapper = create_object(load_config("networks/configs/endo-da3-all-wowrapper.yaml"))
    Model_with_wrapper.eval()
    Model_with_wrapper.to("cuda")

    # Set seed again before creating second model to ensure same initialization
    set_seed(42)
    
    # Model = create_object(load_config("networks/configs/endo-da3-all-wowrapper.yaml"))
    # Model = create_object(load_config("networks/configs/endo-da3-all-wowrapper-giant.yaml"))
    Model = create_object(load_config("networks/configs/endo-da3-depth-wowrapper.yaml"))
    Model.eval()
    Model.to("cuda")

    # Load pretrained weights
    print("\n" + "="*60)
    print("Loading pretrained weights from DepthAnything3")
    print("="*60)
    
    model_pretrained = DepthAnything3.from_pretrained("depth-anything/da3-base")
    # model_pretrained = DepthAnything3.from_pretrained("depth-anything/da3-giant")
    model_pretrained = model_pretrained#.to(device="cuda")

    load_pretrained = False
    # load_pretrained = True
    load_infer_wrapper = False
    load_infer_wrapper = True
    if load_pretrained:

        # DPT DEPTH: B S H W 1; 
        # DPT depth_Conf: B S H W 

        # DualDPT DEPTH: B S H W;
        # DualDPT depth_Conf: B S H W;
        # DualDPT RAY: B S H W 6;
        # DualDPT RAY_CONF: B S H W ;
        # extrinsics: B S 3 4 ;
        # intrinsics: B S 3 3 ;

        # Load weights into Model (without wrapper - needs to remove both prefixes)
        # disable_modules = []
        # if hasattr(Model, 'cam_dec') and Model.cam_dec is not None:
            # disable_modules = ["cam_dec"] if Model.cam_dec.rot_representation!="quat_xyzw" else []
        
        # Determine if cam_dec should be disabled based on rotation representation
        disable_modules = []
        if hasattr(Model, 'cam_dec') and Model.cam_dec is not None:
            if hasattr(Model.cam_dec, 'rot_representation'):
                if Model.cam_dec.rot_representation != "quat_xyzw":
                    # disable_modules = ["cam_dec"] # old models before 01.01.2025
                    disable_modules.append("cam_dec.fc_qvec")
                    disable_modules.append("cam_dec.fc_t")
            if hasattr(Model.cam_dec, 'fc_fov_arch'):
                if Model.cam_dec.fc_fov_arch != "linear_relu":
                    disable_modules.append("cam_dec.fc_fov")

        # load_pretrained_weights(
        #     model=Model,
        #     pretrained_model=model_pretrained,
        #     model_name="Model",
        #     remove_prefixes=["model.", "pretrained."],
        #     disable_modules=disable_modules,
        #     strict=False,
        #     max_levels=3,
        #     # max_levels=6,# show lora param
        #     verbose=False
        # )
        if load_infer_wrapper:
            # Load weights into Model_with_wrapper (only needs to remove model. prefix)
            load_pretrained_weights(
                model=Model_with_wrapper,
                pretrained_model=model_pretrained,
                model_name="Model_with_wrapper",
                remove_prefixes=["model."],
                strict=False,
                max_levels=3,
                verbose=False
            )
    

    # Test with same input
    set_seed(42)  # Set seed for input tensor too
    input_imgs_tensor = torch.randn(2, 3, 3, 336, 504).to("cuda")
    input_imgs_tensor = torch.randn(2, 3, 3, 256, 320).to("cuda")
    input_intrinsics_tensor = torch.randn(2, 3, 3, 3).to("cuda")
    input_extrinsics_tensor = torch.randn(2, 3, 4, 4).to("cuda")

    input_intrinsics_tensor = None
    input_extrinsics_tensor = None
    export_feat_layers = []
    infer_gs = False
    use_ray_pose = False

    with torch.no_grad():
        # output = Model.forward(input_imgs_tensor, 
        #         intrinsics=input_intrinsics_tensor, 
        #         extrinsics=input_extrinsics_tensor,
        #         export_feat_layers=export_feat_layers,
        #         infer_gs=infer_gs,
        #         use_ray_pose=use_ray_pose)
        if load_infer_wrapper:
            output_with_wrapper = Model_with_wrapper.forward(input_imgs_tensor, 
                    intrinsics=input_intrinsics_tensor, 
                    extrinsics=input_extrinsics_tensor,
                    export_feat_layers=export_feat_layers,
                    infer_gs=infer_gs,
                    use_ray_pose=use_ray_pose)
    
    # for key, value in output.items():
    #     print(f"Output {key}: {value.shape}")
    #     if isinstance(value, torch.Tensor):
    #         print(value.min(), value.max(), value.mean())
    #     else:
    #         print(value)
    #     print("-"*60)
    # print('Intrisics')
    # print(output.intrinsics[0, 0])
    # print('Extrinsics')
    # print(output.extrinsics[0, 0])

    if load_infer_wrapper:
        for key, value in output_with_wrapper.items():
            print(f"Output_with_wrapper {key}: {value.shape}")
            # if isinstance(value, torch.Tensor):
            #     print(value.min(), value.max(), value.mean())
            # else:
            #     print(value)
            print("-"*60)


    # print("\n" + "="*60)
    # print("Output Comparison:")
    # print("="*60)

    # # check identical outputs
    # for key in output.keys():
    #     if key in output_with_wrapper:
    #         if output[key].shape == output_with_wrapper[key].shape:
    #             if isinstance(output[key]==output_with_wrapper[key], bool) or (output[key] == output_with_wrapper[key]).all().item():
    #                 print(f"Output {key} is identical: {output[key].shape}")
    #             else:
    #                 print(f"Output {key} is not identical:  {output[key].shape}")
    #         else:
    #             print(f"Output {key} shapes don't match: {output[key].shape} and {output_with_wrapper[key].shape}")
    #     else:
    #         print(f"Output {key} not in output_with_wrapper")








