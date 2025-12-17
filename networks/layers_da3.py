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

from typing import Callable, Tuple
import torch
import torch.nn as nn

from depth_anything_3.model.dinov2.layers import Block
from depth_anything_3.model.dinov2.layers.attention import Attention
from depth_anything_3.model.dinov2.layers.mlp import Mlp
from third_party.EndoDAC.models.backbones.layers.utils import ResBottleneckBlock


class BlockWithResidual(Block):
    """
    Wrapper around DA3 Block that adds residual block (conv layers) support.
    Follows the pattern from EndoDAC Block implementation.
    
    The only difference from the base Block is the residual block extension:
    - use_residual_block: whether to use residual conv blocks
    - res_conv_kernel_size: kernel size for residual conv layers
    - res_conv_padding: padding for residual conv layers
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
        use_residual_block: bool = False,
        res_conv_kernel_size: int = 3,
        res_conv_padding: int = 1,
        patch_size: int = 14,
        input_img_size: Tuple[int, int] = (518, 518),
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            qk_norm=qk_norm,
            rope=rope,
            ln_eps=ln_eps,
        )
        self.use_residual_block = use_residual_block
        self.patch_size = patch_size
        self.input_img_size = input_img_size
        # Compute patch dimensions in __init__ following EndoDAC Block pattern
        self.patch_h = input_img_size[0] // patch_size
        self.patch_w = input_img_size[1] // patch_size
        # Always include cls_token (as per user requirement)
        self.include_cls_token = 1
        
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 8
            self.residual_ = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 8,
                act_layer=act_layer,
                conv_kernels=res_conv_kernel_size,
                conv_paddings=res_conv_padding,
            )
    
    def forward(self, x: torch.Tensor, pos=None, attn_mask=None) -> torch.Tensor:
        # x comes from process_attention which reshapes (B, S, N, C) -> (B*S, N, C) for local attention
        # For residual blocks, we need (B*S, N, C) format 
        # If it's global attention (B, S*N, C), we can't extract spatial patches
        # So we assume it's local attention format (B*S, N, C)
        x_shape = x.shape
        x_ndim = len(x_shape)
        
        if x_ndim == 3:
            # Normal case: x has shape (B*S, N, C) for local 
            B_S, N, C = x_shape
            x = super().forward(x, pos=pos, attn_mask=attn_mask)
            
            # Apply residual block if enabled
            if self.use_residual_block:

                num_patches = self.patch_h * self.patch_w
                
                # Extract patch embeddings from the end (last num_patches tokens)
                patch_embed = x[:, self.include_cls_token:, :].reshape(B_S, self.patch_h, self.patch_w, C).clone()
                # Apply conv residual block
                patch_embed = self.residual_(patch_embed.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                # Add residual back to patch embeddings (only the patch tokens)
                x[:, self.include_cls_token:, :] = x[:, self.include_cls_token:, :] + patch_embed.reshape(B_S, num_patches, C)

        else:
            raise ValueError(f"Unexpected input shape: {x_shape}, expected 3D or 4D tensor")
        
        return x

