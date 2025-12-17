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

def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)




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
                 dino_resize_hw=None):
        """
        Initialize EndoDepthAnything3Net with given yaml-initialized configuration.
        
        Args:
            dino_resize_hw: Tuple of (height, width) to resize input images to before backbone.
                           If None, no resizing is performed. If not None, both h and w must be
                           divisible by PATCH_SIZE (14).
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

        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
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
            output.extrinsics = affine_inverse(c2w)
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


class EndoCameraDec(nn.Module):
    def __init__(self, dim_in=1536, rot_representation="quat_xyzw", 
                 rot_scale_factor=1.0, trans_scale_factor=1.0,
                 explicit_bias_init_6d9d=True):
        super().__init__()
        # Rotation dimension mapping
        rot_dims = {
            "angle_axis": 3, "euler": 3, "quat": 3,
            "quat_xyzw": 4, "quat_wxyz": 4,
            "6D": 6, "9D": 9
        }
        
        if rot_representation not in rot_dims:
            raise ValueError(
                f"Unsupported rotation representation: {rot_representation}. "
                f"Supported: {', '.join(rot_dims.keys())}"
            )
        
        self.rot_representation = rot_representation
        self.rot_scale_factor = rot_scale_factor
        self.trans_scale_factor = trans_scale_factor
        self.explicit_bias_init_6d9d = explicit_bias_init_6d9d
        rot_dim = rot_dims[rot_representation]
        
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, rot_dim)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())
        
        # Apply Special Init on self.fc_qvec if 6D/9D and flag is enabled
        if self.explicit_bias_init_6d9d and self.rot_representation in ["6D", "9D"]:
            # Initialize bias with Identity rotation matrix
            if self.rot_representation == "6D":
                # 6D representation: Identity matrix [1,0,0, 0,1,0] flattened
                bias_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            elif self.rot_representation == "9D":
                # 9D representation: Identity matrix [1,0,0, 0,1,0, 0,0,1] flattened
                bias_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            # Initialize bias (note: device will be set when model is moved to device)
            with torch.no_grad():
                self.fc_qvec.bias.copy_(bias_rot)

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        
        # Apply scale factors to translation and rotation
        out_t = self.trans_scale_factor * self.fc_t(feat.float()).reshape(B, N, 3)
        
        if camera_encoding is None:
            out_rot_raw = self.fc_qvec(feat.float()).reshape(B, N, -1)
            
            # Apply special scaling for 6D/9D with explicit_bias_init_6d9d
            if self.explicit_bias_init_6d9d and self.rot_representation == "6D":
                out_rot = out_rot_raw.clone()
                # scale the r2,r3,r4,r6 by rot_scale_factor
                out_rot[..., 1:4] *= self.rot_scale_factor
                out_rot[..., 5] *= self.rot_scale_factor
            elif self.explicit_bias_init_6d9d and self.rot_representation == "9D":
                out_rot = out_rot_raw.clone()
                # scale the r2,r3,r4, r6,r7,r8 by rot_scale_factor
                out_rot[..., 1:4] *= self.rot_scale_factor
                out_rot[..., 5:8] *= self.rot_scale_factor
            else:
                # naive mul: scale all rotation components by rot_scale_factor
                out_rot = self.rot_scale_factor * out_rot_raw
            
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            # Extract rotation and fov from camera_encoding
            # Format: [T(3), rotation(M), fov_h(1), fov_w(1)]
            # Get rotation dimension from the model's expected dimension
            rot_dim = self.fc_qvec.out_features
            out_rot = camera_encoding[..., 3:3+rot_dim]
            out_fov = camera_encoding[..., -2:]
        
        pose_enc = torch.cat([out_t, out_rot, out_fov], dim=-1)
        return pose_enc


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
    Model_with_wrapper.eval()
    Model_with_wrapper.to("cuda")

    # Set seed again before creating second model to ensure same initialization
    set_seed(42)
    
    Model = create_object(load_config("networks/configs/endo-da3-depth-wowrapper.yaml"))
    Model = create_object(load_config("networks/configs/endo-da3-all-wowrapper.yaml"))
    Model.eval()
    Model.to("cuda")

    # Load pretrained weights
    print("\n" + "="*60)
    print("Loading pretrained weights from DepthAnything3")
    print("="*60)
    
    model_pretrained = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model_pretrained = model_pretrained.to(device="cuda")

    load_pretrained = False
    load_pretrained = True
    load_infer_wrapper = False
    if load_pretrained:
        # Load weights into Model (without wrapper - needs to remove both prefixes)
        load_pretrained_weights(
            model=Model,
            pretrained_model=model_pretrained,
            model_name="Model",
            # remove_prefixes=["model.", "pretrained."],
            disable_modules=["cam_dec", "cam_enc"],
            strict=False,
            max_levels=3,
            verbose=False
        )
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
    input_imgs_tensor = torch.randn(1, 3, 3, 336, 504).to("cuda")
    input_intrinsics_tensor = torch.randn(1, 3, 3, 3).to("cuda")
    input_extrinsics_tensor = torch.randn(1, 3, 4, 4).to("cuda")

    input_intrinsics_tensor = None
    input_extrinsics_tensor = None
    export_feat_layers = []
    infer_gs = False
    use_ray_pose = False

    with torch.no_grad():
        output = Model.forward(input_imgs_tensor, 
                intrinsics=input_intrinsics_tensor, 
                extrinsics=input_extrinsics_tensor,
                export_feat_layers=export_feat_layers,
                infer_gs=infer_gs,
                use_ray_pose=use_ray_pose)
        if load_infer_wrapper:
            output_with_wrapper = Model_with_wrapper.forward(input_imgs_tensor, 
                    intrinsics=input_intrinsics_tensor, 
                    extrinsics=input_extrinsics_tensor,
                    export_feat_layers=export_feat_layers,
                    infer_gs=infer_gs,
                    use_ray_pose=use_ray_pose)
    
    for key, value in output.items():
        print(f"Output {key}: {value.shape}")
        if isinstance(value, torch.Tensor):
            print(value.min(), value.max(), value.mean())
        else:
            print(value)
        print("-"*60)
    print('Intrisics')
    print(output.intrinsics[0, 0])
    print('Extrinsics')
    print(output.extrinsics[0, 0])

    if load_infer_wrapper:
        for key, value in output_with_wrapper.items():
            print(f"Output_with_wrapper {key}: {value.shape}")
            if isinstance(value, torch.Tensor):
                print(value.min(), value.max(), value.mean())
            else:
                print(value)
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








