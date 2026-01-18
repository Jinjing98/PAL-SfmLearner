"""
Multi-scale DPT and DualDPT decoders for Depth Anything 3.
Extends the base DPT and DualDPT classes to support multi-scale output similar to endodac.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from addict import Dict as AddictDict

# Import base classes
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))
from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.utils.head_utils import custom_interpolate


class Interpolate(nn.Module):
    """Interpolation module for upsampling."""
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class HeadDepth(nn.Module):
    """Depth head for multi-scale output, similar to endodac."""
    def __init__(self, features, output_dim=1):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.head(x)


class DPTMultiScale(DPT):
    """
    Multi-scale DPT decoder that outputs predictions at multiple scales.
    Inherits from DPT and adds enable_multi_scale option.
    """
    def __init__(self, *args, enable_multi_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_multi_scale = enable_multi_scale
        
        if self.enable_multi_scale:
            # Create separate output heads for each scale (similar to endodac)
            # Scales correspond to: path_4 (scale 3), path_3 (scale 2), path_2 (scale 1), path_1 (scale 0)
            features = kwargs.get('features', 256)
            output_dim = self.out_dim
            
            # Create depth heads for each scale
            self.conv_depth_4 = HeadDepth(features, output_dim)
            self.conv_depth_3 = HeadDepth(features, output_dim)
            self.conv_depth_2 = HeadDepth(features, output_dim)
            self.conv_depth_1 = HeadDepth(features, output_dim)

    def _fuse_multi_scale(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-scale fusion that returns intermediate paths at different scales.
        Returns: path_4, path_3, path_2, path_1 (corresponding to scales 3, 2, 1, 0)
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # 4 -> 3 -> 2 -> 1 (similar to endodac)
        path_4 = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, l3_rn, size=l2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, l2_rn, size=l1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, l1_rn)
        
        return path_4, path_3, path_2, path_1

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            # permute -> contiguous before reshape to keep conv input contiguous
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # Align scale
            resized_feats.append(x)

        if self.enable_multi_scale:
            # Multi-scale fusion and output
            path_4, path_3, path_2, path_1 = self._fuse_multi_scale(resized_feats)
            
            # Compute target output resolution
            h_out = int(ph * self.patch_size / self.down_ratio)
            w_out = int(pw * self.patch_size / self.down_ratio)
            
            # Apply output heads at each scale
            # Note: HeadDepth already includes upsampling, so we apply it directly
            outs: Dict[str, torch.Tensor] = {}
            
            # Scale 3 (path_4) - coarsest
            logits_4 = self.conv_depth_4(path_4)
            if self.has_conf:
                fmap_4 = logits_4.permute(0, 2, 3, 1)
                pred_4 = self._apply_activation_single(fmap_4[..., :-1], self.activation)
                conf_4 = self._apply_activation_single(fmap_4[..., -1], self.conf_activation)
                outs[f"{self.head_main}_3"] = pred_4.squeeze(1)
                outs[f"{self.head_main}_conf_3"] = conf_4.squeeze(1)
            else:
                outs[f"{self.head_main}_3"] = self._apply_activation_single(logits_4, self.activation).squeeze(1)
            
            # Scale 2 (path_3)
            logits_3 = self.conv_depth_3(path_3)
            if self.has_conf:
                fmap_3 = logits_3.permute(0, 2, 3, 1)
                pred_3 = self._apply_activation_single(fmap_3[..., :-1], self.activation)
                conf_3 = self._apply_activation_single(fmap_3[..., -1], self.conf_activation)
                outs[f"{self.head_main}_2"] = pred_3.squeeze(1)
                outs[f"{self.head_main}_conf_2"] = conf_3.squeeze(1)
            else:
                outs[f"{self.head_main}_2"] = self._apply_activation_single(logits_3, self.activation).squeeze(1)
            
            # Scale 1 (path_2)
            logits_2 = self.conv_depth_2(path_2)
            if self.has_conf:
                fmap_2 = logits_2.permute(0, 2, 3, 1)
                pred_2 = self._apply_activation_single(fmap_2[..., :-1], self.activation)
                conf_2 = self._apply_activation_single(fmap_2[..., -1], self.conf_activation)
                outs[f"{self.head_main}_1"] = pred_2.squeeze(1)
                outs[f"{self.head_main}_conf_1"] = conf_2.squeeze(1)
            else:
                outs[f"{self.head_main}_1"] = self._apply_activation_single(logits_2, self.activation).squeeze(1)
            
            # Scale 0 (path_1) - finest
            logits_1 = self.conv_depth_1(path_1)
            if self.has_conf:
                fmap_1 = logits_1.permute(0, 2, 3, 1)
                pred_1 = self._apply_activation_single(fmap_1[..., :-1], self.activation)
                conf_1 = self._apply_activation_single(fmap_1[..., -1], self.conf_activation)
                outs[f"{self.head_main}_0"] = pred_1.squeeze(1)
                outs[f"{self.head_main}_conf_0"] = conf_1.squeeze(1)
            else:
                outs[f"{self.head_main}_0"] = self._apply_activation_single(logits_1, self.activation).squeeze(1)
            
            # Also output sky head if enabled (at finest scale only)
            if self.use_sky_head:
                # Use path_1 for sky head
                sky_logits = self.scratch.sky_output_conv2(path_1)
                outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)
            
            return outs
        else:
            # Original single-scale behavior
            return super()._forward_impl(feats, H, W, patch_start_idx)


class DualDPTMultiScale(DualDPT):
    """
    Multi-scale DualDPT decoder that outputs predictions at multiple scales.
    Inherits from DualDPT and adds enable_multi_scale option.
    """
    def __init__(self, *args, enable_multi_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_multi_scale = enable_multi_scale
        
        if self.enable_multi_scale:
            # Create separate output heads for each scale
            features = kwargs.get('features', 256)
            output_dim = kwargs.get('output_dim', 2)
            
            # Main head outputs for each scale
            self.conv_depth_main_4 = HeadDepth(features, output_dim)
            self.conv_depth_main_3 = HeadDepth(features, output_dim)
            self.conv_depth_main_2 = HeadDepth(features, output_dim)
            self.conv_depth_main_1 = HeadDepth(features, output_dim)
            
            # Aux head outputs for each scale (7 channels: 6 for ray + 1 for conf)
            self.conv_depth_aux_4 = HeadDepth(features, 7)
            self.conv_depth_aux_3 = HeadDepth(features, 7)
            self.conv_depth_aux_2 = HeadDepth(features, 7)
            self.conv_depth_aux_1 = HeadDepth(features, 7)

    def _fuse_multi_scale(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Multi-scale fusion that returns intermediate paths at different scales for both main and aux.
        Returns: path_main_4, path_main_3, path_main_2, path_main_1, path_aux_4, path_aux_3, path_aux_2, path_aux_1
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # Main branch: 4 -> 3 -> 2 -> 1
        path_main_4 = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        path_main_3 = self.scratch.refinenet3(path_main_4, l3_rn, size=l2_rn.shape[2:])
        path_main_2 = self.scratch.refinenet2(path_main_3, l2_rn, size=l1_rn.shape[2:])
        path_main_1 = self.scratch.refinenet1(path_main_2, l1_rn)
        
        # Aux branch: 4 -> 3 -> 2 -> 1
        path_aux_4 = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        path_aux_3 = self.scratch.refinenet3_aux(path_aux_4, l3_rn, size=l2_rn.shape[2:])
        path_aux_2 = self.scratch.refinenet2_aux(path_aux_3, l2_rn, size=l1_rn.shape[2:])
        path_aux_1 = self.scratch.refinenet1_aux(path_aux_2, l1_rn)
        
        return path_main_4, path_main_3, path_main_2, path_main_1, path_aux_4, path_aux_3, path_aux_2, path_aux_1

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,) -> Dict[str, torch.Tensor]:
        if self.enable_multi_scale:
            B, _, C = feats[0].shape
            ph, pw = H // self.patch_size, W // self.patch_size
            resized_feats = []
            for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
                x = feats[take_idx][:, patch_start_idx:]
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

                x = self.projects[stage_idx](x)
                if self.pos_embed:
                    x = self._add_pos_embed(x, W, H)
                x = self.resize_layers[stage_idx](x)  # align scales
                resized_feats.append(x)

            # Multi-scale fusion
            path_main_4, path_main_3, path_main_2, path_main_1, path_aux_4, path_aux_3, path_aux_2, path_aux_1 = self._fuse_multi_scale(resized_feats)
            
            # Compute target output resolution
            h_out = int(ph * self.patch_size / self.down_ratio)
            w_out = int(pw * self.patch_size / self.down_ratio)
            
            outs: Dict[str, torch.Tensor] = {}
            

            # enforce shows better in exp even on dino 256 320: to be fixed
            enforce_256_320_explicitly = True # necessary if we want more than dino_size_224_280
            enforce_256_320_explicitly = False # necessary if we want more than dino_size_224_280
            
            # Main head outputs at each scale
            for scale_idx, (path_main, path_aux) in enumerate([
                (path_main_4, path_aux_4),
                (path_main_3, path_aux_3),
                (path_main_2, path_aux_2),
                (path_main_1, path_aux_1),
            ]):
                scale_name = str(3 - scale_idx)  # 3, 2, 1, 0

               
                if enforce_256_320_explicitly:
                # --- [Change 1] 计算当前 scale 的目标分辨率 ---
                # h_out_layer = int((ph * self.patch_size / self.down_ratio) / (2**(3-scale_idx)))
                # w_out_layer = int((pw * self.patch_size / self.down_ratio) / (2**(3-scale_idx)))
                    h_out_layer = int((256 / (2**(3-scale_idx))))
                    w_out_layer = int((320 / (2**(3-scale_idx))))


                # Main head
                main_logits = getattr(self, f'conv_depth_main_{scale_idx + 1}')(path_main)

                if enforce_256_320_explicitly:
                    # --- [Change 3] 强制对齐到 target layer resolution ---
                    # print('DPTMulti before interpolate', main_logits.shape)
                    main_logits = custom_interpolate(main_logits, (h_out_layer, w_out_layer), mode="bilinear", align_corners=True)
                    # print('DPTMulti after interpolate', main_logits.shape)

                fmap_main = main_logits.permute(0, 2, 3, 1)
                main_pred = self._apply_activation_single(fmap_main[..., :-1], self.activation)
                main_conf = self._apply_activation_single(fmap_main[..., -1], self.conf_activation)
                outs[f"{self.head_main}_{scale_name}"] = main_pred.squeeze(-1)
                outs[f"{self.head_main}_conf_{scale_name}"] = main_conf.squeeze(-1)
                
                # Aux head
                aux_logits = getattr(self, f'conv_depth_aux_{scale_idx + 1}')(path_aux)

                if enforce_256_320_explicitly:
                    # --- [Change 4] 强制对齐到 target layer resolution ---
                    aux_logits = custom_interpolate(aux_logits, (h_out_layer, w_out_layer), mode="bilinear", align_corners=True)

                fmap_aux = aux_logits.permute(0, 2, 3, 1)
                aux_pred = self._apply_activation_single(fmap_aux[..., :-1], "linear")
                aux_conf = self._apply_activation_single(fmap_aux[..., -1], self.conf_activation)
                outs[f"{self.head_aux}_{scale_name}"] = aux_pred
                # outs[f"{self.head_aux}_conf_{scale_name}"] = aux_conf
            
            return outs
        else:
            # Original single-scale behavior
            return super()._forward_impl(feats, H, W, patch_start_idx)

