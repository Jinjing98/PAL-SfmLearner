from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import sys
from pathlib import Path

# Setup path for depth_anything_3 imports
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))

from third_party.EndoDAC.models.endodac import endodac, mark_only_part_as_trainable
from networks.endo_da3 import mark_only_part_as_trainable_v2, EndoDepthAnything3Net
from depth_anything_3.cfg import create_object, load_config
from omegaconf import OmegaConf
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.dpt import DPT
from utils import load_pretrained_weights
from third_party.EndoDAC.models.encoders import ResnetEncoder
from third_party.EndoDAC.models.decoders import PositionDecoder, TransformDecoder, DepthDecoder
from third_party.EndoDAC.models.decoders import IntrinsicsHead, PoseCNN
from networks.raft import RAFT

from utils.utils_optic_flow import get_occu_mask_backward, get_occu_mask_bidirection, optical_flow
from utils.metrics import compute_depth_metrics, compute_pose_metrics, compute_depth_errors
from utils.util import set_seed, readlines, normalize_image, sec_to_hm_str, disp_to_depth_v2, create_dataset_from_file_or_list
from utils.warping import (
    transformation_from_parameters,
    transformation_from_parameters_6D,
    transformation_from_parameters_9D,
    transformation_from_parameters_quat,
    transformation_from_parameters_euler,
    BackprojectDepth,
    Project3D,
    SpatialTransformer
)
from loss import get_smooth_loss, get_smooth_bright, ncc_loss, SSIM
from networks.pose_decoder import PoseDecoder
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
import torch

# Depth clamping constants (consistent with evaluation)
MIN_DEPTH = 1e-3
MAX_DEPTH = 150

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


class EndoDepthAnything3NetWrapper(torch.nn.Module):
    """
    Wrapper for EndoDepthAnything3Net to adapt interface for trainer_endoda3.
    Converts between (B, 3, H, W) input/output format and EndoDepthAnything3Net's (B, N, 3, H, W) format.
    Also converts depth to disparity and creates multi-scale outputs.
    Formats intrinsics and relative poses if cam_dec outputs are available.
    """
    def __init__(self, model, min_depth=0.1, max_depth=150.0, scales=[0, 1, 2, 3], 
                 rot_representation='angle_axis',
                 da3_depth_regression_target='depth2disp'):
        super().__init__()
        self.model = model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scales = scales
        self.rot_representation = rot_representation
        self.da3_depth_regression_target = da3_depth_regression_target
        # Expose lora_type from wrapped model for compatibility
        self.lora_type = getattr(model, 'lora_type', 'none')
        
    def forward(self, x, frame_id=None, index_in_spatial_S=None, raw_model_output=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) or (B, S, 3, H, W) for multi-frame
            frame_id: Optional frame_id for relative pose outputs (required if index_in_spatial_S is provided)
            index_in_spatial_S: Optional index in spatial dimension S to extract pose for specific frame_id
            raw_model_output: Optional pre-computed raw model output (for caching, avoids re-calling model)
            
        Returns:
            Dictionary with keys:
            - ("disp", scale) for each scale
            - ("K", 0), ("inv_K", 0) if intrinsics available (from frame 0)
            - ("translation", 0, frame_id), rotation outputs if multi-frame and extrinsics available
        """
        single_frame_input = True
        if x.dim() == 4:
            # Convert single frame input (B, 3, H, W) to (B, S, 3, H, W) for EndoDepthAnything3Net
            B, C, H, W = x.shape
            x_mv = x.unsqueeze(1)  # (B, S, 3, H, W), S==1
        else:
            single_frame_input = False
            # multi-frame input (B, S, 3, H, W)
            assert x.dim() == 5, f"x shape: {x.shape}"
            B, S, C, H, W = x.shape
            x_mv = x
            # Validate input ordering: frame 0 must be first (for ref_view_strategy='first')
            # We can infer this from the fact that we're using enable_seq_inputs with frame_ids starting with 0
        assert C == 3, f"C shape: {C}"
        
        # Forward through EndoDepthAnything3Net
        # Use cached raw_model_output if provided (for enable_seq_inputs caching)
        if raw_model_output is not None:
            output = raw_model_output
        else:
            # Call model normally
            output = self.model(x_mv, extrinsics=None, intrinsics=None, 
                               export_feat_layers=[], infer_gs=False, use_ray_pose=False)
        # Depth output: (B, S, H, W) if  DualDPT Head
        # Depth output: (B, S, H, W, 1) if DPT Head
        head = self.model.head
        if isinstance(self.model.head, DPT):
            if head.head_main == "depth":
                output.depth = output.depth.squeeze(-1)
            elif head.head_main == "disp":
                output.disp = output.disp.squeeze(-1)
        if self.da3_depth_regression_target == "depth2disp":
            assert head.head_main == "depth", f"Expected head_main='depth' for depth2disp regression, but got head_main='{head.head_main}'"
            assert head.activation == "sigmoid", f"Expected activation='sigmoid' for depth2disp regression, but got activation='{head.activation}'"
            
            depth = output.depth
            assert depth.dim() == 4, f"depth shape: {depth.shape}"
            if single_frame_input:
                assert depth.shape[1] == 1, f"depth shape: {depth.shape}"
            
            # For multi-frame input with ref_view_strategy='first', extract depth for target frame (frame 0)
            if not single_frame_input:
                # Ensure ref_view_strategy is 'first' for deterministic frame 0 extraction
                assert self.model.ref_view_strategy == "first", \
                    f"ref_view_strategy must be 'first' for multi-frame depth extraction, got {self.model.ref_view_strategy}"
                # Extract depth for frame 0 (first frame in spatial dimension)
                depth = depth[:, 0:1, :, :]  # (B, 1, H, W) - keep dim for consistency

            # Exp before 19.Dec improperly use self.min_depth
            # depth_clamped = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
            depth_clamped = torch.clamp(depth, min=MIN_DEPTH, max=MAX_DEPTH)
            disp = 1.0 / depth_clamped # (B, 1, H, W) for multi-frame, (B, S, H, W) for single-frame
        
            # Interpolate to match input image size if needed
            if disp.shape[-2:] != (256, 320):
                disp = F.interpolate(disp, size=(256, 320), mode="bilinear", align_corners=True)

            # Create multi-scale disp outputs
            outputs = {}
            for scale in self.scales:
                if scale == 0:
                    outputs[("disp", scale)] = disp
                else:
                    h_scale = H // (2 ** scale)
                    w_scale = W // (2 ** scale)
                    disp_scale = F.interpolate(disp, size=(h_scale, w_scale), 
                                            mode="bilinear", align_corners=True)
                    outputs[("disp", scale)] = disp_scale
        elif self.da3_depth_regression_target == "disp":
            # Sanity checks: verify head configuration matches disp regression
            assert head.head_main == "disp", f"Expected head_main='disp' for disp regression, but got head_main='{head.head_main}'"
            assert head.activation == "sigmoid", f"Expected activation='sigmoid' for disp regression, but got activation='{head.activation}'"

            disp = output.disp
            
            assert disp.dim() == 4, f"Expected disp shape (B, S, H, W), but got shape: {disp.shape}"

            if single_frame_input:
                assert disp.shape[1] == 1, f"disp shape: {disp.shape}"
            
            # For multi-frame input with ref_view_strategy='first', extract disp for target frame (frame 0)
            if not single_frame_input:
                # Ensure ref_view_strategy is 'first' for deterministic frame 0 extraction
                assert self.model.ref_view_strategy == "first", \
                    f"ref_view_strategy must be 'first' for multi-frame disp extraction, got {self.model.ref_view_strategy}"
                # Extract disp for frame 0 (first frame in spatial dimension)
                disp = disp[:, 0:1, :, :]  # (B, 1, H, W) - keep dim for consistency
            
            # Interpolate to match input image size if needed
            if disp.shape[-2:] != (256, 320):
                disp = F.interpolate(disp, size=(256, 320), mode="bilinear", align_corners=True)
            
            # Create multi-scale disp outputs
            outputs = {}
            for scale in self.scales:
                if scale == 0:
                    outputs[("disp", scale)] = disp
                else:
                    h_scale = H // (2 ** scale)
                    w_scale = W // (2 ** scale)
                    disp_scale = F.interpolate(disp, size=(h_scale, w_scale), 
                                              mode="bilinear", align_corners=True)
                    outputs[("disp", scale)] = disp_scale
        
        # Format intrinsics and pose outputs if available
        if hasattr(output, 'intrinsics') and output.intrinsics is not None:
            intrinsics = output.intrinsics
            # Ensure intrinsics is a tensor
            if isinstance(intrinsics, torch.Tensor):
                # Use intrinsics from frame 0 (reference frame)
                # make sure ref_view_strategy is "first"
                assert self.model.ref_view_strategy == "first", f"ref_view_strategy must be 'first' for intrinsics estimates"
                cam_K = intrinsics[:, 0]  # (B, 3, 3)
                inv_K = torch.inverse(cam_K)
                outputs[('K', 0)] = cam_K
                outputs[('inv_K', 0)] = inv_K
        
        # Format relative pose if multi-frame and extrinsics available
        # Skip pose extraction if index_in_spatial_S is None (will be handled later in predict_poses)
        if not single_frame_input and hasattr(output, 'extrinsics') and output.extrinsics is not None:
            extrinsics = output.extrinsics  # (B, S, 3, 4) - w2c format
            # Ensure extrinsics is a tensor
            if isinstance(extrinsics, torch.Tensor) and extrinsics.shape[1] >= 2:
                # Only extract pose if index_in_spatial_S is provided (explicit pose extraction)
                # Otherwise skip - pose will be extracted later in predict_poses() with proper index_in_spatial_S
                if index_in_spatial_S is not None:
                    # frame_id must be provided when extracting pose
                    assert frame_id is not None, "frame_id must be provided when index_in_spatial_S is not None"
                    
                    # Make sure ref_view_strategy is "first" for deterministic parsing
                    assert self.model.ref_view_strategy == "first", \
                        f"ref_view_strategy must be 'first' for relative pose estimates, got {self.model.ref_view_strategy}"
                    
                    # Use explicit index_in_spatial_S
                    assert 0 <= index_in_spatial_S < extrinsics.shape[1], \
                        f"index_in_spatial_S ({index_in_spatial_S}) out of range [0, {extrinsics.shape[1]})"
                    # Frame 0 is always at index 0 (ref_view_strategy='first')
                    idx_0 = 0
                    idx_fi = index_in_spatial_S
                    
                    # Compute relative pose from frame 0 to frame idx_fi
                    R_0, t_0 = extrinsics[:, idx_0, :, :3], extrinsics[:, idx_0, :, 3:4]  # (B, 3, 3), (B, 3, 1)
                    R_fi, t_fi = extrinsics[:, idx_fi, :, :3], extrinsics[:, idx_fi, :, 3:4]  # (B, 3, 3), (B, 3, 1)
                    
                    # Build 4x4 matrices and compute relative: T_rel = T_fi @ inv(T_0) (for w2c extrinsics)
                    ones_row = torch.zeros(B, 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
                    ones_row[:, 0, 3] = 1.0
                    T_0 = torch.cat([torch.cat([R_0, t_0], dim=2), ones_row], dim=1)  # (B, 4, 4)
                    T_fi = torch.cat([torch.cat([R_fi, t_fi], dim=2), ones_row], dim=1)  # (B, 4, 4)
                    T_rel = T_fi @ torch.inverse(T_0)  # (B, 4, 4) - relative pose from frame 0 to frame idx_fi
                    
                    R_rel = T_rel[:, :3, :3]  # (B, 3, 3)
                    t_rel = T_rel[:, :3, 3:4]  # (B, 3, 1)
                    
                    # Convert rotation matrix to required representation
                    rot_output = self._rot_matrix_to_representation(R_rel, self.rot_representation)
                    rot_output = rot_output.unsqueeze(1)  # (B, 1, M)
                    translation = t_rel.squeeze(-1).unsqueeze(1)  # (B, 1, 3)
                    
                    rot_output = rot_output.unsqueeze(1)
                    translation = translation.unsqueeze(-1)

                    # Store outputs
                    outputs[("translation", 0, frame_id)] = translation
                    self._store_pose_outputs_wrapper(outputs, rot_output, translation, self.rot_representation, frame_id)
                # else: Skip pose extraction - will be handled later in predict_poses() with proper index_in_spatial_S
        
        return outputs
    
    def _rot_matrix_to_representation(self, R, rot_representation):
        """Convert rotation matrix to required representation.
        Reuses existing functions from utils.warping where possible.
        """
        if rot_representation == "6D":
            return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)  # (B, 6)
        elif rot_representation == "9D":
            return R.reshape(R.shape[0], -1)  # (B, 9)
        elif rot_representation == "quat":
            from depth_anything_3.model.utils.transform import mat_to_quat
            quat_xyzw = mat_to_quat(R)  # (B, 4) in XYZW format
            return quat_xyzw[:, :3]  # (B, 3)
        elif rot_representation == "angle_axis":
            trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
            axis = torch.stack([
                R[:, 2, 1] - R[:, 1, 2],
                R[:, 0, 2] - R[:, 2, 0],
                R[:, 1, 0] - R[:, 0, 1]
            ], dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)
            axis = axis / (axis_norm + 1e-8)
            return axis * angle.unsqueeze(-1)  # (B, 3)
        elif rot_representation == "euler":
            sy = torch.sqrt(torch.clamp(R[:, 0, 0]**2 + R[:, 1, 0]**2, min=1e-8))
            singular = sy < 1e-6
            x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
            y = torch.atan2(-R[:, 2, 0], sy)
            z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
            x[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 1, 1])
            y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
            z[singular] = 0
            return torch.stack([x, y, z], dim=1)  # (B, 3)
        else:
            raise ValueError(f"Unsupported rotation representation: {rot_representation}")
    
    def extract_pose_from_output(self, raw_model_output, frame_id, index_in_spatial_S, B, H, W):
        """
        Extract pose from already-computed raw model output without re-calling the model.
        
        Args:
            raw_model_output: Raw output from EndoDepthAnything3Net model
            frame_id: Frame ID for pose output key
            index_in_spatial_S: Index in spatial dimension S for frame_id (frame 0 is at index 0)
            B, H, W: Batch size, height, width
            
        Returns:
            Dictionary with pose outputs for the specified frame_id
        """
        outputs = {}
        
        # Extract extrinsics if available
        if hasattr(raw_model_output, 'extrinsics') and raw_model_output.extrinsics is not None:
            extrinsics = raw_model_output.extrinsics  # (B, S, 3, 4) - w2c format
            if isinstance(extrinsics, torch.Tensor) and extrinsics.shape[1] >= 2:
                assert self.model.ref_view_strategy == "first", \
                    f"ref_view_strategy must be 'first' for relative pose estimates, got {self.model.ref_view_strategy}"
                
                # Frame 0 is always at index 0 (ref_view_strategy='first')
                idx_0 = 0
                idx_fi = index_in_spatial_S
                
                assert 0 <= idx_fi < extrinsics.shape[1], \
                    f"index_in_spatial_S ({idx_fi}) out of range [0, {extrinsics.shape[1]})"
                
                # Compute relative pose from frame 0 to frame idx_fi
                R_0, t_0 = extrinsics[:, idx_0, :, :3], extrinsics[:, idx_0, :, 3:4]  # (B, 3, 3), (B, 3, 1)
                R_fi, t_fi = extrinsics[:, idx_fi, :, :3], extrinsics[:, idx_fi, :, 3:4]  # (B, 3, 3), (B, 3, 1)
                
                # Build 4x4 matrices and compute relative: T_rel = T_fi @ inv(T_0) (for w2c extrinsics)
                ones_row = torch.zeros(B, 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
                ones_row[:, 0, 3] = 1.0
                T_0 = torch.cat([torch.cat([R_0, t_0], dim=2), ones_row], dim=1)  # (B, 4, 4)
                T_fi = torch.cat([torch.cat([R_fi, t_fi], dim=2), ones_row], dim=1)  # (B, 4, 4)
                T_rel = T_fi @ torch.inverse(T_0)  # (B, 4, 4) - relative pose from frame 0 to frame idx_fi
                
                R_rel = T_rel[:, :3, :3]  # (B, 3, 3)
                t_rel = T_rel[:, :3, 3:4]  # (B, 3, 1)
                
                # Convert rotation matrix to required representation
                rot_output = self._rot_matrix_to_representation(R_rel, self.rot_representation)
                rot_output = rot_output.unsqueeze(1)  # (B, 1, M)
                translation = t_rel.squeeze(-1).unsqueeze(1)  # (B, 1, 3)
                
                rot_output = rot_output.unsqueeze(1)
                translation = translation.unsqueeze(-1)
                
                # Store outputs
                outputs[("translation", 0, frame_id)] = translation
                self._store_pose_outputs_wrapper(outputs, rot_output, translation, self.rot_representation, frame_id)
        
        return outputs
    
    def _store_pose_outputs_wrapper(self, outputs, rot_output, translation, rot_representation, frame_id):
        """Store pose outputs in wrapper (reuses transformation functions from warping)."""
        from utils.warping import (
            transformation_from_parameters,
            transformation_from_parameters_6D,
            transformation_from_parameters_9D,
            transformation_from_parameters_quat,
            transformation_from_parameters_euler
        )
        if rot_representation == "angle_axis":
            outputs[("axisangle", 0, frame_id)] = rot_output
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "6D":
            outputs[("rot6d", 0, frame_id)] = rot_output
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters_6D(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "9D":
            outputs[("rot9d", 0, frame_id)] = rot_output
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters_9D(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "quat":
            outputs[("quat", 0, frame_id)] = rot_output
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters_quat(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "euler":
            outputs[("euler", 0, frame_id)] = rot_output
            outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters_euler(rot_output[:, 0], translation[:, 0])
        else:
            raise ValueError(f"Unsupported rotation representation: {rot_representation}")


class RAFTWrapper(torch.nn.Module):
    """
    Wrapper for RAFT to match PositionDecoder interface.
    Generates multi-scale flows from single RAFT output.
    Supports multi-iteration outputs for different scales.
    """
    def __init__(self, raft_model, scales, max_disp=None, use_multi_iters=False, multi_iters=[2,5,8,11]):
        super().__init__()
        # Register the RAFT model's underlying PyTorch model as a submodule
        # so its parameters are accessible through .parameters()
        if hasattr(raft_model, 'model'):
            # RAFT wrapper contains a .model attribute which is the actual PyTorch model
            self.add_module('raft_pytorch_model', raft_model.model)
        else:
            # If raft_model is already a PyTorch model, register it directly
            self.add_module('raft_pytorch_model', raft_model)
        self.raft_model = raft_model  # Keep reference to RAFT wrapper for forward calls
        self.scales = scales
        self.max_disp = max_disp
        self.use_multi_iters = use_multi_iters
        self.multi_iters = multi_iters 
        
        # Validate multi_iters if enabled
        if self.use_multi_iters:
            assert len(self.multi_iters) == len(self.scales), \
                f"multi_iters length ({len(self.multi_iters)}) must match scales length ({len(self.scales)})"
            # Note: iter_idx can be up to num_flow_updates-1 (0-indexed)
            # The flow_predictions list will have length num_flow_updates
            assert all(0 <= iter_idx < raft_model.num_flow_updates for iter_idx in self.multi_iters), \
                f"All multi_iters must be in range [0, {raft_model.num_flow_updates})"
        
    def _sanitize_flow(self, flow):
        """Sanitize flow: remove NaN and Inf values."""
        flow = torch.where(torch.isfinite(flow), flow, torch.zeros_like(flow))
        return flow
    
    def _clamp_flow(self, flow, scale_factor=1.0):
        """Clamp flow magnitude if max_disp is specified."""
        if self.max_disp is not None:
            max_val = self.max_disp * scale_factor
            flow = torch.clamp(flow, min=-max_val, max=max_val)
        return flow
    
    def _generate_multiscale_flows(self, flow_base, base_h, base_w):
        """
        Generate multi-scale flows from base RAFT output.
        Strategy: downsample flow and scale magnitude proportionally.
        When downsampling by 2^scale, flow magnitude is divided by 2^scale.
        """
        outputs = {}
        for scale in self.scales:
            if scale == 0:
                flow = flow_base
            else:
                h_scale = base_h // (2 ** scale)
                w_scale = base_w // (2 ** scale)
                flow = F.interpolate(flow_base, size=(h_scale, w_scale), 
                                    mode="bilinear", align_corners=True)
                # Scale flow magnitude: when resolution is 1/2^scale, flow is 1/2^scale
                flow = flow / (2 ** scale)
            
            flow = self._sanitize_flow(flow)
            # Clamp uses original max_disp scaled by resolution factor
            flow = self._clamp_flow(flow, scale_factor=1.0 / (2 ** scale) if scale > 0 else 1.0)
            outputs[("position", scale)] = flow
        
        return outputs
    
    def forward(self, framel, framer):
        """
        Forward pass through RAFT.
        
        Args:
            framel: Left frame (B, 3, H, W)
            framer: Right frame (B, 3, H, W)
            
        Returns:
            Dictionary with keys ("position", scale) for each scale
        """
        if self.use_multi_iters:
            # Get all iteration outputs
            flow_predictions = self.raft_model(framel, framer, return_all_iterations=True)
            # flow_predictions is a list of flows from each iteration
            
            outputs = {}
            # Get base resolution from first flow
            if len(flow_predictions) > 0:
                flow_base = flow_predictions[0]
                B, C, base_h, base_w = flow_base.shape
                assert C == 2, f"Expected 2-channel flow, got {C}"
            else:
                raise ValueError("No flow predictions returned from RAFT")
            
            for scale_idx, scale in enumerate(self.scales):
                # Get flow from specified iteration for this scale
                iter_idx = self.multi_iters[scale_idx]
                assert iter_idx < len(flow_predictions), f"iter_idx {iter_idx} is out of range {len(flow_predictions)}"
                flow_at_iter = flow_predictions[iter_idx]  # (B, 2, H, W)
                
                # Scale flow to the appropriate resolution for this scale
                if scale == 0:
                    flow = flow_at_iter
                else:
                    h_scale = base_h // (2 ** scale)
                    w_scale = base_w // (2 ** scale)
                    flow = F.interpolate(flow_at_iter, size=(h_scale, w_scale), 
                                        mode="bilinear", align_corners=True)
                    # Scale flow magnitude: when resolution is 1/2^scale, flow is 1/2^scale
                    flow = flow / (2 ** scale)
                
                flow = self._sanitize_flow(flow)
                # Clamp uses original max_disp scaled by resolution factor
                flow = self._clamp_flow(flow, scale_factor=1.0 / (2 ** scale) if scale > 0 else 1.0)
                outputs[("position", scale)] = flow
        else:
            # Original behavior: use final flow and generate multi-scale
            flow_base = self.raft_model(framel, framer)  # Returns (B, 2, H, W) tensor
            
            if flow_base.dim() == 3:
                flow_base = flow_base.unsqueeze(0)
            B, C, H, W = flow_base.shape
            assert C == 2, f"Expected 2-channel flow, got {C}"
            
            outputs = self._generate_multiscale_flows(flow_base, H, W)
        
        return outputs


class Trainer:
    def __init__(self, options):
        self.opt = options
        # Prepend exp_suffix to model_name if provided
        model_name_with_suffix = self.opt.model_name
        if hasattr(self.opt, 'exp_suffix') and self.opt.exp_suffix:
            model_name_with_suffix = f"{self.opt.exp_suffix}_{self.opt.model_name}"
        self.log_path = os.path.join(self.opt.log_dir, model_name_with_suffix)

        # Set random seed for reproducibility
        if hasattr(self.opt, 'seed'):
            set_seed(self.opt.seed)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # 字典
        self.parameters_to_train = []  # 列表
        self.parameters_to_train_0 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        
        # Validate enable_seq_inputs
        self.enable_seq_inputs = getattr(self.opt, 'enable_seq_inputs', False)
        if self.enable_seq_inputs:
            assert self.opt.depth_model_type == "depthanything3", \
                f"enable_seq_inputs requires depth_model_type='depthanything3', got '{self.opt.depth_model_type}'"
            # When enable_seq_inputs is True, we always use multi-frame input
            # Validate that frame_ids starts with 0
            assert self.opt.frame_ids[0] == 0, \
                f"enable_seq_inputs requires frame_ids to start with 0, got {self.opt.frame_ids}"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Construct models in order: depth -> pose -> k -> (of, af at end)
        self.construct_depth_model()
        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        if self.use_pose_net:
            self.construct_pose_model()
            # Add pose_encoder parameters if it exists (not for da3_internal)
            if "pose_encoder" in self.models:
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            if "pose" in self.models:  # pose model doesn't exist for da3_internal
                self.parameters_to_train += list(self.models["pose"].parameters())
            
            if self.opt.learn_intrinsics:
                self.construct_k_model()
                if 'intrinsics_head' in self.models:  # intrinsics_head doesn't exist for da3_internal
                    self.parameters_to_train += list(self.models['intrinsics_head'].parameters())

        # Construct of and af models at the end
        self.construct_of_model()
        if "position_encoder" in self.models:
            self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())
        self.parameters_to_train_0 += list(self.models["position"].parameters())

        self.construct_af_model()
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())
        self.parameters_to_train += list(self.models["transform"].parameters())

        # Hardcoded flags: enable learnable camera intrinsics and GT rotations
        self.learnable_K = False
        self.replace_with_gt_rel_rotation = False
        # self.learnable_K = True
        # self.replace_with_gt_rel_rotation = True

        # Initialize learnable camera intrinsics (normalized coordinates)
        if self.learnable_K:
            K_init = torch.tensor([
                [0.82, 0.0, 0.5],
                [0.0, 1.02, 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32, device=self.device)
            self.learnable_K_params = torch.nn.Parameter(K_init.clone())
            self.learnable_K_params.to(self.device)
            self.parameters_to_train.append(self.learnable_K_params)
            print("Learnable camera intrinsics enabled (hardcoded flag)")

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        splits_dir = os.path.join(os.path.dirname(__file__), "splits", self.opt.split)
        train_file = getattr(self.opt, 'train_data_file', ['train_files.txt'])
        val_file = getattr(self.opt, 'val_data_file', ['val_files.txt'])
        test_file = getattr(self.opt, 'test_data_file', ['test_files.txt'])
        
        img_ext = '.png'
        
        # Create datasets using shared utility function
        train_dataset = create_dataset_from_file_or_list(
            train_file, splits_dir, self.dataset, self.opt.data_path, 
            self.opt.height, self.opt.width, self.opt.frame_ids, 4, 
            is_train=True, img_ext=img_ext, opt=self.opt, mode='train'
        )
        val_dataset = create_dataset_from_file_or_list(
            val_file, splits_dir, self.dataset, self.opt.data_path,
            self.opt.height, self.opt.width, self.opt.frame_ids, 4,
            is_train=False, img_ext=img_ext, opt=self.opt, mode='val'
        )
        test_dataset = create_dataset_from_file_or_list(
            test_file, splits_dir, self.dataset, self.opt.data_path,
            self.opt.height, self.opt.width, self.opt.frame_ids, 4,
            is_train=False, img_ext=img_ext, opt=self.opt, mode='test'
        )
        
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # is_train = not getattr(self.opt, 'of_samples', False) # can be used for compute depth err
        shuffle = not getattr(self.opt, 'of_samples', False)  # Fixed order for overfitting
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        # test_dataset = self.dataset(
        #     self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
        #     load_gt_poses=os.path.basename(test_fpath) != 'test_files.txt',# there is missing GT for d7k4 where a lot of test samples are
        #     load_gt_depth=True,
        #     )
        # self.test_loader = DataLoader(
        #     test_dataset, 1, False,
        #     num_workers=1, pin_memory=True, drop_last=True,)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rmse", "de/log_rmse", "da/a1", "da/a2", "da/a3"]

        # gt_path = os.path.join(splits_dir, self.opt.eval_split, "gt_depths.npz")
        # self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items, {:d} validation items \n".format(
            len(train_dataset), len(val_dataset)))
        # print("There are {:d} testing items\n".format(len(test_dataset)))

        self.save_opts()
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for name, param in self.models["depth_model"].named_parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad == False:
                NonTrainable_params += mulValue
                # print(name)
            else:
                Trainable_params += mulValue


        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        print(f'Trainable params ratio: {100 * Trainable_params / Total_params}%')

    def construct_depth_model(self):
        """Construct and initialize the depth model from config file.
        Loads pretrained weights if specified in options.
        """
        if self.opt.depth_model_type == "depthanything3":
            # Initialize EndoDepthAnything3Net from config file
            endoda3_model_config_path = self.opt.endoda3_model_config 
            assert os.path.exists(endoda3_model_config_path), f"Config file not found: {endoda3_model_config_path}"
            print(f"Loading depth model setting from config: {endoda3_model_config_path}")
            endoda3_model_config = load_config(endoda3_model_config_path)
            # Store config for saving later
            self.endoda3_model_config = endoda3_model_config
            self.endoda3_model_config_path = endoda3_model_config_path
            depth_model_base = create_object(endoda3_model_config)
            
            # Wrap the model to adapt interface
            self.models["depth_model"] = EndoDepthAnything3NetWrapper(
                depth_model_base,
                min_depth=self.opt.min_depth,
                max_depth=self.opt.max_depth,
                scales=self.opt.scales,
                rot_representation=getattr(self.opt, 'rot_representation', 'angle_axis'),
                da3_depth_regression_target=getattr(self.opt, 'da3_depth_regression_target', 'depth2disp')
            )
            self.models["depth_model"].to(self.device)
            
            # Load pretrained weights if requested
            if self.opt.pretrained_path is not None:
                print("\n" + "="*60)
                print(f"Loading pretrained weights from {self.opt.pretrained_path}")
                print("="*60)
                
                model_pretrained = DepthAnything3.from_pretrained(self.opt.pretrained_path)
                model_pretrained = model_pretrained.to(device=self.device)
                
                # Get the underlying model from the wrapper
                model_to_load = depth_model_base
                
                # Determine if cam_dec should be disabled based on rotation representation
                disable_cam_dec = []
                if hasattr(model_to_load, 'cam_dec') and model_to_load.cam_dec is not None:
                    if hasattr(model_to_load.cam_dec, 'rot_representation'):
                        if model_to_load.cam_dec.rot_representation != "quat_xyzw":
                            disable_cam_dec = ["cam_dec"]
                
                # Load pretrained weights
                load_pretrained_weights(
                    model=model_to_load,
                    pretrained_model=model_pretrained,
                    model_name="depth_model",
                    remove_prefixes=["model.", "pretrained."],
                    disable_modules=disable_cam_dec,
                    strict=False,
                    max_levels=3,
                    verbose=False
                )
                print(f"Successfully loaded pretrained weights from {self.opt.pretrained_path} for depth net.\n")
            else:
                assert False, "scratch training?"
        elif self.opt.depth_model_type == "endodac":
            # Sanity check: endodac model outputs disparity, so da3_depth_regression_target must be "disp"
            da3_depth_regression_target = getattr(self.opt, 'da3_depth_regression_target', 'depth2disp')
            assert da3_depth_regression_target == "disp", \
                f"endodac depth model outputs disparity, so da3_depth_regression_target must be 'disp', " \
                f"but got '{da3_depth_regression_target}'"
            
            # Initialize endodac model (same as in trainer_endodac.py)
            self.models["depth_model"] = endodac(
                backbone_size=getattr(self.opt, 'backbone_size', 'base'), 
                r=getattr(self.opt, 'lora_rank', 4), 
                lora_type=getattr(self.opt, 'lora_type', 'dvlora'),
                image_shape=(224, 280), 
                pretrained_path=self.opt.pretrained_path,
                residual_block_indexes=getattr(self.opt, 'residual_block_indexes', [2, 5, 8, 11]),
                include_cls_token=getattr(self.opt, 'include_cls_token', True))
            self.models["depth_model"].to(self.device)
            
            print(f"Initialized endodac depth model with backbone_size={getattr(self.opt, 'backbone_size', 'base')}, "
                  f"lora_rank={getattr(self.opt, 'lora_rank', 4)}, lora_type={getattr(self.opt, 'lora_type', 'dvlora')}")
        else:
            raise ValueError(f"Unsupported depth_model_type: {self.opt.depth_model_type}")

    def construct_pose_model(self):
        """Construct and initialize the pose model.
        Supports different pose model types: separate_resnet, shared, posecnn, da3_internal.
        """
        if self.opt.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)
            self.models["pose"] = PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
                trans_scale_factor=getattr(self.opt, 'trans_scale_factor', 0.001),
                rot_scale_factor=getattr(self.opt, 'rot_scale_factor', 0.001),
                rot_representation=getattr(self.opt, 'rot_representation', 'angle_axis'),
                explicit_bias_init_6d9d=getattr(self.opt, 'explicit_bias_init_6d9d', False))

        elif self.opt.pose_model_type == "shared":
            self.models["pose"] = PoseDecoder(
                self.models["encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.pose_model_type == "posecnn":
            self.models["pose"] = PoseCNN(
                self.num_input_frames if self.opt.pose_model_input == "all" else 2)

        elif self.opt.pose_model_type == "da3_internal":
            # Sanity check: depth_model must have cam_dec module
            depth_model_base = self.models["depth_model"].model
            if not hasattr(depth_model_base, 'cam_dec') or depth_model_base.cam_dec is None:
                raise ValueError(
                    "pose_model_type 'da3_internal' requires depth model to have cam_dec module. "
                    "Ensure the depth model config includes cam_dec."
                )
            # Verify rotation representation matches
            cam_dec_rot_repr = getattr(depth_model_base.cam_dec, 'rot_representation', 'quat_xyzw')
            opt_rot_repr = getattr(self.opt, 'rot_representation', 'angle_axis')
            if cam_dec_rot_repr != opt_rot_repr and cam_dec_rot_repr != "quat_xyzw":
                print(f"Warning: cam_dec rot_representation ({cam_dec_rot_repr}) != opt rot_representation ({opt_rot_repr}). "
                      f"Will convert from {cam_dec_rot_repr} to {opt_rot_repr}.")
            # No pose model needed - will extract from depth model output

        if self.opt.pose_model_type != "da3_internal":
            self.models["pose"].to(self.device)

    def construct_k_model(self):
        """Construct and initialize the intrinsics (K) model.
        Supports mlp_with_pn_bottleneck_ipt and da3_internal.
        """
        if self.opt.k_model_type == "mlp_with_pn_bottleneck_ipt":
            # Sanity check: pose_encoder must exist for this model type
            if "pose_encoder" not in self.models:
                raise ValueError(
                    "k_model_type 'mlp_with_pn_bottleneck_ipt' requires pose_encoder. "
                    "Ensure pose_model_type is 'separate_resnet'."
                )
            self.models['intrinsics_head'] = IntrinsicsHead(self.models["pose_encoder"].num_ch_enc)
            self.models['intrinsics_head'].to(self.device)
        elif self.opt.k_model_type == "da3_internal":
            # Sanity check: depth_model must have cam_dec module
            depth_model_base = self.models["depth_model"].model
            if not hasattr(depth_model_base, 'cam_dec') or depth_model_base.cam_dec is None:
                raise ValueError(
                    "k_model_type 'da3_internal' requires depth model to have cam_dec module. "
                    "Ensure the depth model config includes cam_dec."
                )
            # No intrinsics_head needed - will extract from depth model output
        else:
            raise ValueError(f"Unsupported k_model_type: {self.opt.k_model_type}")

    def construct_of_model(self):
        """Construct and initialize the optical flow (OF) model.
        Supports separate_resnet and raft types.
        """
        if self.opt.of_model_type == "separate_resnet":
            self.models["position_encoder"] = ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
            self.models["position_encoder"].to(self.device)
            self.models["position"] = PositionDecoder(
                self.models["position_encoder"].num_ch_enc, self.opt.scales)
            self.models["position"].to(self.device)
        elif self.opt.of_model_type == "raft":
            raft_model = RAFT(
                device=self.device,
                weights="Raft_Large_Weights.DEFAULT",
                num_flow_updates=getattr(self.opt, 'raft_num_flow_updates', 12)
            )
            
            # Set trainable modules for RAFT
            raft_trainable_modules = getattr(self.opt, 'raft_trainable_modules', [])
            self._set_raft_trainable_modules(raft_model, raft_trainable_modules)
            
            use_multi_iters = getattr(self.opt, 'use_raft_multi_iters', False)
            multi_iters = getattr(self.opt, 'raft_multi_iters', [2, 5, 8, 11])
            
            self.models["position"] = RAFTWrapper(
                raft_model=raft_model,
                scales=self.opt.scales,
                max_disp=getattr(self.opt, 'raft_max_disp', None),
                use_multi_iters=use_multi_iters,
                multi_iters=multi_iters
            )
            self.models["position"].to(self.device)
        else:
            raise ValueError(f"Unsupported of_model_type: {self.opt.of_model_type}")
    
    def _set_raft_trainable_modules(self, raft_model, trainable_modules):
        """
        Set which RAFT modules are trainable.
        
        Args:
            raft_model: RAFT model instance
            trainable_modules: List of module names to make trainable.
                              Examples: ['convnormrelu', 'layer1', 'layer2_0']
                              Special: 'layer2_0' means first block (index 0) of layer2
        
        Raises:
            AssertionError: If any provided module name cannot be found in the model
        """
        # First, freeze all parameters
        for param in raft_model.model.parameters():
            param.requires_grad = False
        
        if not trainable_modules:
            # No trainable modules specified, RAFT is completely frozen
            return
        
        # Get the underlying PyTorch model
        model = raft_model.model
        
        # Track which modules were successfully found and unfrozen
        found_modules = set()
        
        # Handle special case: "all"
        if "all" in trainable_modules:
            print("Unfreezing all RAFT parameters")
            for param in model.parameters():
                param.requires_grad = True
            return
        
        # Build a map of feature_encoder children for efficient lookup
        feature_encoder_map = {}
        if hasattr(model, 'feature_encoder'):
            feature_encoder_map = {name: module for name, module in model.feature_encoder.named_children()}
        
        # Process each trainable module
        for module_name in trainable_modules:
            # Check if it's a layer_X_Y pattern (e.g., 'layer2_0')
            if "_" in module_name:
                parts = module_name.split("_")
                if len(parts) >= 2:
                    layer_name = parts[0]
                    try:
                        block_idx = int(parts[1])
                        # Check if layer exists in feature_encoder
                        if layer_name in feature_encoder_map:
                            module = feature_encoder_map[layer_name]
                            if block_idx < len(module):
                                block = module[block_idx]
                                print(f"Unfreezing Feature Encoder module: {layer_name}[{block_idx}] ({module_name})")
                                for p in block.parameters():
                                    p.requires_grad = True
                                found_modules.add(module_name)
                            else:
                                raise AssertionError(
                                    f"Block index {block_idx} out of range for '{layer_name}'. "
                                    f"Available blocks: 0-{len(module)-1}"
                                )
                        else:
                            raise AssertionError(
                                f"Layer '{layer_name}' not found in feature_encoder. "
                                f"Available layers: {list(feature_encoder_map.keys())}"
                            )
                    except ValueError:
                        # Not a valid block index pattern, treat as regular module name
                        pass
            
            # Check if it's a full feature_encoder layer name
            if module_name in feature_encoder_map:
                if module_name not in found_modules:  # Avoid double-processing
                    print(f"Unfreezing Feature Encoder module: {module_name}")
                    for p in feature_encoder_map[module_name].parameters():
                        p.requires_grad = True
                    found_modules.add(module_name)
                continue
            
            # Check other top-level modules
            if module_name == "update_block":
                if hasattr(model, 'update_block'):
                    print(f"Unfreezing module: {module_name}")
                    for param in model.update_block.parameters():
                        param.requires_grad = True
                    found_modules.add(module_name)
                else:
                    raise AssertionError(f"Module 'update_block' not found in RAFT model")
            elif module_name == "context_encoder":
                if hasattr(model, 'context_encoder'):
                    print(f"Unfreezing module: {module_name}")
                    for param in model.context_encoder.parameters():
                        param.requires_grad = True
                    found_modules.add(module_name)
                else:
                    raise AssertionError(f"Module 'context_encoder' not found in RAFT model")
            elif module_name not in found_modules and "_" not in module_name:
                # Module not found - check if it was a layer_X_Y pattern that failed
                # (already handled above with assertion)
                # If it's a simple name that wasn't found anywhere, raise error
                raise AssertionError(
                    f"Module '{module_name}' not found in RAFT model. "
                    f"Available feature_encoder modules: {list(feature_encoder_map.keys())}"
                )

    def construct_af_model(self):
        """Construct and initialize the affine transform (AF) model.
        Currently only supports separate_resnet type.
        """
        if self.opt.af_model_type == "separate_resnet":
            self.models["transform_encoder"] = ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
            self.models["transform_encoder"].to(self.device)
            self.models["transform"] = TransformDecoder(
                self.models["transform_encoder"].num_ch_enc, self.opt.scales)
            self.models["transform"].to(self.device)
        else:
            raise ValueError(f"Unsupported af_model_type: {self.opt.af_model_type}")

    def set_train_0(self):
        """Convert all models to training mode
        """
        if "position_encoder" in self.models:
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = True
        
        # Handle position model (RAFT or separate_resnet)
        if self.opt.of_model_type == "raft":
            # For RAFT, only make trainable modules trainable
            raft_trainable_modules = getattr(self.opt, 'raft_trainable_modules', [])
            if raft_trainable_modules:
                # Get the RAFT model from the wrapper
                position_model = self.models["position"]
                if hasattr(position_model, 'raft_model'):
                    raft_model = position_model.raft_model
                    # Re-apply trainable modules setting (in case it was changed)
                    self._set_raft_trainable_modules(raft_model, raft_trainable_modules)
                # Set wrapper to train mode
                self.models["position"].train()
            else:
                # No trainable modules, but still need to set to train mode for forward pass
                # Parameters remain frozen
                self.models["position"].train()
        else:
            # separate_resnet: make all parameters trainable
            for param in self.models["position"].parameters():
                param.requires_grad = True
            self.models["position"].train()

        if "position_encoder" in self.models:
            self.models["position_encoder"].train()

        for param in self.models["depth_model"].parameters():
            param.requires_grad = False
        if "pose_encoder" in self.models:
            for param in self.models["pose_encoder"].parameters():
                param.requires_grad = False
        if "pose" in self.models:
            for param in self.models["pose"].parameters():
                param.requires_grad = False
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["transform"].parameters():
            param.requires_grad = False
        if self.opt.learn_intrinsics and "intrinsics_head" in self.models:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = False

        self.models["depth_model"].eval()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].eval()
        if "pose" in self.models:
            self.models["pose"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        if self.opt.learn_intrinsics and "intrinsics_head" in self.models:
            self.models["intrinsics_head"].eval()

    def set_train(self):
        """Convert all models to training mode
        """
        if "position_encoder" in self.models:
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = False
        
        # Handle position model (RAFT or separate_resnet)
        if self.opt.of_model_type == "raft":
            # For RAFT, freeze all parameters (including trainable modules)
            position_model = self.models["position"]
            if hasattr(position_model, 'raft_model'):
                raft_model = position_model.raft_model
                # Freeze all RAFT parameters
                for param in raft_model.model.parameters():
                    param.requires_grad = False
        else:
            # separate_resnet: freeze all parameters
            for param in self.models["position"].parameters():
                param.requires_grad = False

        for name, param in self.models["depth_model"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = True

        if self.models["depth_model"].lora_type != "none":        
            if self.step < self.opt.warm_up_step:
                warm_up = True
            else:
                warm_up = False
            mark_only_part_as_trainable_v2(self.models["depth_model"], 
                                            warm_up=warm_up,
                                            other_trainable=["residual_", "conv_depth_"])

        if "pose_encoder" in self.models:
            for param in self.models["pose_encoder"].parameters():
                param.requires_grad = True
        if "pose" in self.models:
            for param in self.models["pose"].parameters():
                param.requires_grad = True
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["transform"].parameters():
            param.requires_grad = True
        if self.opt.learn_intrinsics and "intrinsics_head" in self.models:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = True

        if "position_encoder" in self.models:
            self.models["position_encoder"].eval()
        self.models["position"].eval()

        self.models["depth_model"].train()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].train()
        if "pose" in self.models:
            self.models["pose"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
        if self.opt.learn_intrinsics and "intrinsics_head" in self.models:
            self.models["intrinsics_head"].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].eval()
        if "pose" in self.models:
            self.models["pose"].eval()
        if "position_encoder" in self.models:
            self.models["position_encoder"].eval()
        self.models["position"].eval()
        if self.opt.learn_intrinsics and "intrinsics_head" in self.models:
            self.models["intrinsics_head"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        # Get configurable metric names
        self.best_depth_metric = 'rmse'
        self.best_pose_metric = 'pose_rot_err_deg'
        
        # Track best values and epochs
        self.best_depth_value = None
        self.best_depth_epoch = None
        self.best_pose_value = None
        self.best_pose_epoch = None
        
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            
            # Run validation after each epoch
            val_metrics = self.val()
            
            # Save best model based on depth metric
            if val_metrics is not None and self.best_depth_metric in val_metrics:
                current_depth_value = val_metrics[self.best_depth_metric]
                if self.best_depth_value is None or current_depth_value < self.best_depth_value:
                    self.best_depth_value = current_depth_value
                    self.best_depth_epoch = self.epoch
                    print(f"New best {self.best_depth_metric}: {self.best_depth_value:.4f} at epoch {self.epoch}")
                    self.save_model(mode='best_depth')
                    self._save_best_metrics_info()
            
            # Save best model based on pose metric
            if val_metrics is not None and self.best_pose_metric in val_metrics:
                current_pose_value = val_metrics[self.best_pose_metric]
                if self.best_pose_value is None or current_pose_value < self.best_pose_value:
                    self.best_pose_value = current_pose_value
                    self.best_pose_epoch = self.epoch
                    print(f"New best {self.best_pose_metric}: {self.best_pose_value:.4f} at epoch {self.epoch}")
                    self.save_model(mode='best_pose')
                    self._save_best_metrics_info()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model(mode='epoch')
        
        # Save last model
        self.save_model(mode='last')
        
        # Save final best metrics info
        self._save_best_metrics_info()
            
    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # position
            self.set_train_0()
            _, losses_0 = self.process_batch_0(inputs)
            self.model_optimizer_0.zero_grad()
            losses_0["loss"].backward()
            self.model_optimizer_0.step()

            # depth, pose, transform
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                # Compute metrics (depth and pose) if available
                metrics = {}
                # log depth metrics during trn
                if getattr(self.opt, 'compute_depth_metrics', False):
                    depth_metrics = compute_depth_metrics(inputs, outputs)
                    if depth_metrics:
                        metrics.update(depth_metrics)
                
                # log pose metrics during trn
                if getattr(self.opt, 'compute_pose_metrics', False):
                    pose_metrics = compute_pose_metrics(inputs, outputs, self.opt.frame_ids)
                    if pose_metrics:
                        metrics.update(pose_metrics)

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses, metrics=metrics if metrics else None)

            self.step += 1
            
        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    # def run_epoch_eval(self):
    #     """Run a single epoch of evaluation
    #     """

    #     print("Evaluating")
    #     MIN_DEPTH = 1e-3
    #     MAX_DEPTH = 150
        
    #     self.set_eval()
    #     pred_depths = []
    #     for batch_idx, inputs in enumerate(self.test_loader):
    #         input_color = inputs[("color", 0, 0)].cuda()

    #         if self.opt.post_process:
    #             # Post-processed results require each image to have two forward passes
    #             input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

    #         # output = self.models["depth"](self.models["encoder"](input_color))
    #         output = self.models["depth_model"](input_color)
    #         _, pred_depth = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
    #         pred_depth = pred_depth[:, 0].cpu().detach().numpy()
    #         pred_depths.append(pred_depth)
            
    #     pred_depths = np.concatenate(pred_depths)
        
    #     errors = []
    #     ratios = []
        
    #     for i in range(pred_depths.shape[0]):
    #         # gt_depth = self.gt_depths[i]
    #         # obtain gt_depth from inputs
    #         gt_depth = inputs[("depth_gt", 0, 0)].cpu().detach().numpy().squeeze()
    #         gt_height, gt_width = gt_depth.shape[:2]

    #         pred_depth = pred_depths[i]
    #         pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            
    #         mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    #         pred_depth = pred_depth[mask]
    #         gt_depth = gt_depth[mask]

    #         pred_depth *= self.opt.pred_depth_scale_factor
    #         # print(pred_depth.max(), pred_depth.min())
    #         if not self.opt.disable_median_scaling:
    #             ratio = np.median(gt_depth) / np.median(pred_depth)
    #             ratios.append(ratio)
    #             pred_depth *= ratio

    #         pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    #         pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
    #         # errors.append(compute_errors(gt_depth, pred_depth))
    #         errors.append(compute_depth_errors(gt_depth, pred_depth))
    #     if not self.opt.disable_median_scaling:
    #         ratios = np.array(ratios)
    #         med = np.median(ratios)
    #         print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    #     mean_errors = np.array(errors).mean(0)

    #     writer = self.writers["train"]
    #     for i in range(len(mean_errors)):
    #         writer.add_scalar(self.depth_metric_names[i], mean_errors[i], self.epoch)
    #     print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    #     print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        
    #     self.set_train()
        
    #     return mean_errors[2], mean_errors[4]
    def process_batch_0(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs))
        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses

    def predict_poses_0(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":

                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position
                    if self.opt.of_model_type == "raft":
                        # RAFT takes raw frames directly, no encoder
                        outputs_0 = self.models["position"](pose_feats[0], pose_feats[f_i]) # compute tgt2src flow
                        outputs_1 = self.models["position"](pose_feats[f_i], pose_feats[0])
                    else:
                        # separate_resnet: use encoder
                        # has historical order issue
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)],
                                                                                       outputs[(
                                                                                       "position", "high", scale, f_i)])

                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                            outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[
                            ("occu_mask_backward", 0, f_i)].detach() + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0,
                                                                       max=1.0)
        return outputs

    def compute_losses_0(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0

            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += (get_smooth_loss(outputs[("position", scale, frame_id)], color))
                
                # Get supervision target based on of_supervised_with_which
                of_supervised_with = getattr(self.opt, 'of_supervised_with_which', 'outputs_refined')
                if of_supervised_with == 'outputs_refined':
                    supervision_target = outputs[("refined", scale, frame_id)].detach()
                elif of_supervised_with == 'inputs_color':
                    supervision_target = inputs[("color", 0, 0)]
                else:
                    raise ValueError(f"of_supervised_with_which '{of_supervised_with}' not supported. Options: 'outputs_refined', 'inputs_color'")
                
                loss_registration += (
                    self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], supervision_target) * occu_mask_backward).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        # Handle enable_seq_inputs: use multi-frame input and cache result
        cached_depth_output = None
        cached_raw_model_output = None  # Cache raw model output for pose extraction
        if self.enable_seq_inputs:
            # Stack all frames in order specified by frame_ids
            # frame_ids should start with 0 (validated in __init__)
            frames_list = [inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids]
            frames_input = torch.stack(frames_list, dim=1)  # (B, S, 3, H, W) where S=len(frame_ids)
            
            # Call the underlying model directly ONCE to get raw output (for caching)
            wrapper = self.models["depth_model"]
            raw_model_output = wrapper.model(
                frames_input, 
                extrinsics=None, intrinsics=None,
                export_feat_layers=[], infer_gs=False, use_ray_pose=False
            )
            cached_raw_model_output = raw_model_output
            
            # Process through wrapper to get formatted outputs (depth, etc.) using cached raw output
            # This avoids calling the model again
            cached_depth_output = self.models["depth_model"](
                frames_input, 
                frame_id=None,  # Will be set per frame_id in predict_poses
                raw_model_output=raw_model_output  # Pass cached output to avoid re-calling model
            )
            outputs = cached_depth_output
        else:
            # shared by endoDAC and endoda3_naive_single_input
            # Original behavior: single frame input
            outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, None, cached_depth_output=None, cached_raw_model_output=cached_raw_model_output))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def get_K_invK(self, inputs, outputs, scale, batch_size):
        """
        Get camera intrinsics K and inv_K for a given scale.
        Priority: learn_intrinsics (predicted) > learnable_K (optimizable) > inputs
        """
        # if self.opt.learn_intrinsics and ('K', scale) in outputs:
        if self.opt.learn_intrinsics:
            assert ('K', scale) in outputs, f"K for scale {scale} not found in outputs"
            # Use predicted intrinsics from intrinsics_head
            return outputs[('K', scale)], outputs[('inv_K', scale)]
        elif self.learnable_K:
            assert 0, 'disabled...'
            # Build normalized K using differentiable operations for proper gradient flow
            K_norm = torch.cat([
                torch.cat([self.learnable_K_params, torch.zeros(3, 1, device=self.device, dtype=torch.float32)], dim=1),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32)
            ], dim=0)

            # Scale K: first row by w, second row by h
            h, w = self.opt.height // (2 ** scale), self.opt.width // (2 ** scale)
            scale_rows = torch.tensor([[w, w, w, w], [h, h, h, h], [1, 1, 1, 1], [1, 1, 1, 1]], 
                                     device=self.device, dtype=torch.float32)
            K_scaled = K_norm * scale_rows

            K = K_scaled.unsqueeze(0).repeat(batch_size, 1, 1)
            inv_K = torch.inverse(K)
            return K, inv_K
        else:
            # Fallback: use provided intrinsics from inputs
            return inputs[("K", scale)], inputs[("inv_K", scale)]

    def get_K_invK_perframe(self, inputs, outputs, scale, batch_size, frame_id):
        """
        Get per-frame camera intrinsics K and inv_K for a given scale and frame_id.
        Uses K_per_frame from inputs when use_perframe_gt_K is enabled.
        
        Args:
            inputs: Input dictionary
            outputs: Output dictionary
            scale: Scale level
            batch_size: Batch size
            frame_id: Frame ID to get K for
        """
        assert frame_id == 0,'we only need k/inv_k for target frame f0'
        if self.opt.learn_intrinsics:
            # by default the learned K is perframe!
            assert ("K", 0) in outputs, f"K for frame_id={frame_id}, scale={scale} not found in outputs"
            return outputs[("K", 0)], outputs[("inv_K", 0)]
            # assert ("K_per_frame", frame_id, scale) in outputs, f"K_per_frame for frame_id={frame_id}, scale={scale} not found in outputs"
            # return outputs[("K_per_frame", frame_id, scale)], outputs[("inv_K_per_frame", frame_id, scale)]
        else:
            if ("K_per_frame", frame_id, scale) in inputs:
                K = inputs[("K_per_frame", frame_id, scale)]
                inv_K = inputs[("inv_K_per_frame", frame_id, scale)]
                assert K.shape == (batch_size, 4, 4), f"K shape should be (batch_size, 4, 4), but got {K.shape}"
                return K, inv_K
            else:
                raise KeyError(f"K_per_frame for frame_id={frame_id}, scale={scale} not found in inputs. "
                            f"Available keys: {[k for k in inputs.keys() if 'K' in str(k)]}")

    def predict_poses(self, inputs, disps, cached_depth_output=None, cached_raw_model_output=None):
        """Predict poses between input frames for monocular sequences.
        disps: outputs from depth model
        cached_depth_output: Cached depth model output when enable_seq_inputs is True
        cached_raw_model_output: Cached raw frames input when enable_seq_inputs is True (for pose extraction)
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position
                    if self.opt.of_model_type == "raft":
                        # RAFT takes raw frames directly, no encoder
                        outputs_0 = self.models["position"](pose_feats[0], pose_feats[f_i]) # compute tgt2src flow
                        outputs_1 = self.models["position"](pose_feats[f_i], pose_feats[0])
                    else:
                        # separate_resnet: use encoder
                        # has historical order issue
                        position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                        position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                        outputs_0 = self.models["position"](position_inputs)
                        outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:

                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)])
                    
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)],  outputs[("occu_map_backward", scale, f_i)]= self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(outputs[("position", "high", scale, f_i)],
                                                                                                          outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:

                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()  + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0, max=1.0)
                        # outputs[("grad_refined", scale, f_i)] = get_gradmap(outputs[("refined", scale, f_i)])
                                                                                            

                    # pose and intrinsics
                    # da3_internal means using depthanything3's internal pose/K decoder
                    # When enable_seq_inputs is True, use cached output. Otherwise, call model per frame pair.
                    depth_output_dict = None
                    need_da3_output = (self.opt.pose_model_type == "da3_internal" or 
                                      (self.opt.learn_intrinsics and self.opt.k_model_type == "da3_internal"))
                    
                    if need_da3_output:
                        if self.enable_seq_inputs and cached_raw_model_output is not None:
                            # Extract pose/K from cached raw model output without re-calling the model
                            if f_i in self.opt.frame_ids:
                                index_in_spatial_S = self.opt.frame_ids.index(f_i)
                                B = pose_feats[0].shape[0]
                                H, W = pose_feats[0].shape[2], pose_feats[0].shape[3]
                                # Extract pose directly from cached raw model output
                                depth_output_dict = self.models["depth_model"].extract_pose_from_output(
                                    cached_raw_model_output,
                                    frame_id=f_i,
                                    index_in_spatial_S=index_in_spatial_S,
                                    B=B, H=H, W=W
                                )
                                # Also extract intrinsics if needed
                                if self.opt.learn_intrinsics and self.opt.k_model_type == "da3_internal":
                                    if hasattr(cached_raw_model_output, 'intrinsics') and cached_raw_model_output.intrinsics is not None:
                                        intrinsics = cached_raw_model_output.intrinsics
                                        if isinstance(intrinsics, torch.Tensor):
                                            wrapper = self.models["depth_model"]
                                            assert wrapper.model.ref_view_strategy == "first", \
                                                f"ref_view_strategy must be 'first' for intrinsics estimates"
                                            cam_K = intrinsics[:, 0]  # (B, 3, 3)
                                            inv_K = torch.inverse(cam_K)
                                            depth_output_dict[('K', 0)] = cam_K
                                            depth_output_dict[('inv_K', 0)] = inv_K
                            else:
                                raise ValueError(f"frame_id {f_i} not found in frame_ids {self.opt.frame_ids}")
                        else:
                            # endoda3_pair_posenet
                            # Original behavior: call depth model per frame pair (when enable_seq_inputs is False)
                            frames_input = torch.stack([pose_feats[0], pose_feats[f_i]], dim=1)  # (B, 2, 3, H, W)
                            depth_output_dict = self.models["depth_model"](frames_input, frame_id=f_i)
                    
                    # Extract pose from wrapper output
                    if self.opt.pose_model_type == "da3_internal":
                        # Wrapper already formatted pose outputs, just merge them
                        if depth_output_dict is None:
                            raise ValueError("da3_internal pose_model_type requires depth model call. "
                                           "This should not happen if logic is correct.")
                        # Merge formatted pose outputs from wrapper (all keys except disp and K/inv_K)
                        pose_keys = ["translation", "axisangle", "rot6d", "rot9d", "quat", "euler", "cam_T_cam"]
                        for key, value in depth_output_dict.items():
                            if isinstance(key, tuple) and len(key) >= 1:
                                if key[0] in pose_keys:
                                    outputs[key] = value
                    else:
                        # Original pose prediction logic
                        # historical order issue
                        pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                        rot_output, translation, intermediate_feature = self.models["pose"](pose_inputs, ret_intermediate_feat=True)
                        outputs[("translation", 0, f_i)] = translation
                        
                        rot_representation = getattr(self.opt, 'rot_representation', 'angle_axis')
                        self._store_pose_outputs(outputs, rot_output, translation, rot_representation, f_i)
                    
                    # Extract intrinsics from wrapper output
                    if self.opt.learn_intrinsics:
                        if self.opt.k_model_type == "da3_internal":
                            # Wrapper already formatted intrinsics, just merge them
                            if depth_output_dict is None:
                                raise ValueError("da3_internal k_model_type requires depth model call. "
                                               "This should not happen if logic is correct.")
                            if ('K', 0) in depth_output_dict:
                                outputs[('K', 0)] = depth_output_dict[('K', 0)]
                                outputs[('inv_K', 0)] = depth_output_dict[('inv_K', 0)]
                            else:
                                raise ValueError("da3_internal k_model_type requires depth model to output intrinsics. "
                                               "Ensure cam_dec is enabled in depth model config.")
                        else:
                            # Use intrinsics_head (requires intermediate_feature from pose model)
                            if self.opt.pose_model_type == "da3_internal":
                                raise ValueError("k_model_type 'mlp_with_pn_bottleneck_ipt' requires pose_model_type != 'da3_internal'. "
                                               "Use k_model_type 'da3_internal' when pose_model_type is 'da3_internal'.")
                            cam_K = self.models['intrinsics_head'](intermediate_feature, self.opt.width, self.opt.height)
                            inv_K = torch.inverse(cam_K)
                            outputs[('K', 0)] = cam_K
                            outputs[('inv_K', 0)] = inv_K
                    
                    # Optionally replace rotation with GT relative rotation if available
                    if self.replace_with_gt_rel_rotation:
                        if ("gt_c2w_poses", 0) in inputs and ("gt_c2w_poses", f_i) in inputs:
                            gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
                            gt_src_abs_poses = inputs[("gt_c2w_poses", f_i)]  # (B, 4, 4)
                            gt_tgt2src_rel_poses = torch.inverse(gt_src_abs_poses) @ gt_tgt_abs_poses
                            outputs[("cam_T_cam", 0, f_i)][:, :3, :3] = gt_tgt2src_rel_poses[:, :3, :3]
                            # If desired, translation could also be replaced; keeping network translation for now.
                    
        return outputs

    def _store_pose_outputs(self, outputs, rot_output, translation, rot_representation, f_i):
        """Store pose outputs in the correct format based on rotation representation.
        Reuses existing transformation_from_parameters functions.
        """
        if rot_representation == "angle_axis":
            outputs[("axisangle", 0, f_i)] = rot_output
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "6D":
            outputs[("rot6d", 0, f_i)] = rot_output
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_6D(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "9D":
            outputs[("rot9d", 0, f_i)] = rot_output
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_9D(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "quat":
            outputs[("quat", 0, f_i)] = rot_output
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_quat(rot_output[:, 0], translation[:, 0])
        elif rot_representation == "euler":
            outputs[("euler", 0, f_i)] = rot_output
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(rot_output[:, 0], translation[:, 0])
        else:
            raise ValueError(f"Unsupported rotation representation: {rot_representation}")

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            if self.opt.da3_depth_regression_target in ["disp", "depth2disp"]:
                disp = outputs[("disp", scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

                _, depth = disp_to_depth_v2(disp, self.opt.min_depth, self.opt.max_depth, 
                                            is_scaled_disp= (self.opt.da3_depth_regression_target == "depth2disp")) # sigmoid output is in range [0, 1]
                outputs[("depth", 0, scale)] = depth # only used for metric computation;
            else:
                raise ValueError(f"Unsupported depth regression target: {self.opt.da3_depth_regression_target}")

            source_scale = 0
            # Use per-frame K if enabled, otherwise use regular K
            if getattr(self.opt, 'use_perframe_gt_K', False):
                # Get K for frame 0 (depth is from frame 0)
                cam_K, inv_K = self.get_K_invK_perframe(inputs, outputs, source_scale, depth.shape[0], frame_id=0)# obtain the K for target frame f0
            else:
                cam_K, inv_K = self.get_K_invK(inputs, outputs, source_scale, depth.shape[0])
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    rot_representation = getattr(self.opt, 'rot_representation', 'angle_axis')
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    if rot_representation == "angle_axis":
                        axisangle = outputs[("axisangle", 0, frame_id)]
                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0])
                    elif rot_representation == "6D":
                        rot6d = outputs[("rot6d", 0, frame_id)]
                        T = transformation_from_parameters_6D(
                            rot6d[:, 0], translation[:, 0] * mean_inv_depth[:, 0])
                    elif rot_representation == "9D":
                        rot9d = outputs[("rot9d", 0, frame_id)]
                        T = transformation_from_parameters_9D(
                            rot9d[:, 0], translation[:, 0] * mean_inv_depth[:, 0])
                    elif rot_representation == "quat":
                        quat = outputs[("quat", 0, frame_id)]
                        T = transformation_from_parameters_quat(
                            quat[:, 0], translation[:, 0] * mean_inv_depth[:, 0])
                    elif rot_representation == "euler":
                        euler = outputs[("euler", 0, frame_id)]
                        T = transformation_from_parameters_euler(
                            euler[:, 0], translation[:, 0] * mean_inv_depth[:, 0])
                    else:
                        raise ValueError(f"Unsupported rotation representation: {rot_representation}")

                cam_points = self.backproject_depth[source_scale](
                    depth, inv_K)
                # Normalize K and T to expected shapes for Project3D
                # K should be (B, 3, 3) for Project3D
                cam_K_3x3 = cam_K[:, :3, :3] if cam_K.shape[1] == 4 else cam_K
                # T should be (B, 3, 4) for Project3D
                T_3x4 = T[:, :3, :] if T.shape[1] == 4 else T
                pix_coords = self.project_3d[source_scale](
                    cam_points, cam_K_3x3, T_3x4)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                # Reuse normalized K and T for position_depth (same shape requirement)
                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, cam_K_3x3, T_3x4)

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                # Get supervision target based on posedepth_supervised_with_which
                posedepth_supervised_with = getattr(self.opt, 'posedepth_supervised_with_which', 'outputs_refined')
                if posedepth_supervised_with == 'outputs_refined':
                    supervision_target = outputs[("refined", scale, frame_id)]
                else:
                    raise ValueError(f"posedepth_supervised_with_which '{posedepth_supervised_with}' not supported. Only 'outputs_refined' is supported.")
                
                loss_reprojection += (
                    self.compute_reprojection_loss(outputs[("color", frame_id, scale)], supervision_target) * occu_mask_backward).sum() / occu_mask_backward.sum()  
                loss_transform += (
                    torch.abs(outputs[("refined", scale, frame_id)] - outputs[("registration", 0, frame_id)].detach()).mean(1, True) * occu_mask_backward).sum() / occu_mask_backward.sum()
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)], inputs[("color", 0, 0)], outputs[("registration", scale, frame_id)].detach(), occu_mask_backward)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0) 
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self):
        """Validate the model on validation set
        Returns:
            Dictionary of validation metrics (or None if no metrics computed)
        """
        self.set_eval()
        if getattr(self.opt, 'val_full_eval', False):
            report_quantile_pose_err = True # used for compute quantile
            
            metrics_accum = {}
            metrics_trans_ang_err_raw_accum = {}
            metrics_rot_err_raw_accum = {}
            last_inputs = None
            last_outputs = None
            last_losses = None

            def _accum(acc, new_metrics):
                for k, v in new_metrics.items():
                    acc.setdefault(k, []).append(float(v))
            def _accum_raw(acc, new_metrics):
                for k, v in new_metrics.items():
                    # acc.setdefault(k, []).append(v)
                    acc.setdefault(k, []).extend(v)

            with torch.no_grad():
                for inputs in self.val_loader:
                    outputs, losses = self.process_batch_val(inputs)
                    last_inputs, last_outputs, last_losses = inputs, outputs, losses

                    if getattr(self.opt, 'compute_depth_metrics', False):
                        depth_metrics = compute_depth_metrics(inputs, outputs)
                        if depth_metrics:
                            _accum(metrics_accum, depth_metrics)

                    if getattr(self.opt, 'compute_pose_metrics', False):
                        if report_quantile_pose_err:
                            pose_metrics, trans_ang_err_metrics_raw, rot_err_metrics_raw = compute_pose_metrics(inputs, outputs, self.opt.frame_ids, ret_raw=True)
                        else:
                            pose_metrics = compute_pose_metrics(inputs, outputs, self.opt.frame_ids)
                        
                        if pose_metrics:
                            _accum(metrics_accum, pose_metrics)
                        
                        if report_quantile_pose_err:
                            _accum_raw(metrics_trans_ang_err_raw_accum, trans_ang_err_metrics_raw)
                            _accum_raw(metrics_rot_err_raw_accum, rot_err_metrics_raw)
            # Average accumulated metrics
            metrics = {k: sum(v_list) / len(v_list) for k, v_list in metrics_accum.items()} if metrics_accum else None
            if report_quantile_pose_err and metrics_trans_ang_err_raw_accum:
                q = [0.25,0.5,0.75]
                for q_i in q:
                    metrics_trans_ang_err_raw = {k + f'_Q{q_i}': np.quantile(v, q_i) for k, v in metrics_trans_ang_err_raw_accum.items()} if metrics_trans_ang_err_raw_accum else None
                    metrics_rot_err_raw = {k + f'_Q{q_i}': np.quantile(v, q_i) for k, v in metrics_rot_err_raw_accum.items()} if metrics_rot_err_raw_accum else None
                    metrics.update(metrics_trans_ang_err_raw)
                    metrics.update(metrics_rot_err_raw)


            if last_inputs is not None:
                self.log("val", None, last_outputs, last_losses, metrics=metrics)
                del last_inputs, last_outputs, last_losses
        else:
            try:
                inputs = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = next(self.val_iter)

            with torch.no_grad():
                outputs, losses = self.process_batch_val(inputs)
                
                # Compute metrics (depth and pose) if available
                metrics = {}
                if getattr(self.opt, 'compute_depth_metrics', False):
                    depth_metrics = compute_depth_metrics(inputs, outputs)
                    if depth_metrics:
                        metrics.update(depth_metrics)
                
                if getattr(self.opt, 'compute_pose_metrics', False):
                    pose_metrics = compute_pose_metrics(inputs, outputs, self.opt.frame_ids)
                    if pose_metrics:
                        metrics.update(pose_metrics)
                
                self.log("val", inputs, outputs, losses, metrics=metrics if metrics else None)
                del inputs, outputs, losses

        self.set_train()
        return metrics

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        # Handle multi-frame input mode (enable_seq_inputs)
        if self.enable_seq_inputs:
            # Stack all frames in order specified by frame_ids
            frames_list = [inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids]
            frames_input = torch.stack(frames_list, dim=1)  # (B, S, 3, H, W) where S=len(frame_ids)
            
            # Call the underlying model directly ONCE to get raw output (for caching)
            wrapper = self.models["depth_model"]
            raw_model_output = wrapper.model(
                frames_input, 
                extrinsics=None, intrinsics=None,
                export_feat_layers=[], infer_gs=False, use_ray_pose=False
            )
            cached_raw_model_output = raw_model_output
            
            # Process through wrapper to get formatted outputs (depth, etc.) using cached raw output
            cached_depth_output = self.models["depth_model"](
                frames_input, 
                frame_id=None,  # Will be set per frame_id in predict_poses
                raw_model_output=raw_model_output  # Pass cached output to avoid re-calling model
            )
            outputs = cached_depth_output
        else:
            # Original behavior: single frame input
            outputs = self.models["depth_model"](inputs["color_aug", 0, 0])
            cached_raw_model_output = None
            cached_depth_output = None

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, None, 
                                             cached_depth_output=None,
                                             cached_raw_model_output=cached_raw_model_output if self.enable_seq_inputs else None))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        #////////////////////////////////////////
        # enforce the losses are computed on data which has depth metrics computed
        # Check if GT depth is available in inputs
        if ("depth_gt", 0, 0) not in inputs:
            return {}
        # Get predicted depth from outputs
        if ("depth", 0, 0) not in outputs:
            return {}
        #////////////////////////////////////////

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []
            refine_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True), target.mean(1, True)))
                
                # Compute refinement loss: quality of refined image vs pose+K warped image
                if ("refined", scale, frame_id) in outputs and ("color", frame_id, scale) in outputs:
                    refine_losses.append(
                        ncc_loss(
                            outputs[("refined", scale, frame_id)].mean(1, True), 
                            outputs[("color", frame_id, scale)].mean(1, True)
                        ))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)
            loss_registration = registration_losses.mean()
            loss += loss_registration
            
            # Process refine_losses following the same style as registration_losses
            loss_refine = torch.tensor(0.0, device=self.device)
            if len(refine_losses) > 0:
                refine_losses_cat = torch.cat(refine_losses, 1)
                refine_losses_cat, idxs_refine = torch.min(refine_losses_cat, dim=1)
                loss_refine = refine_losses_cat.mean()
                loss += loss_refine
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["loss_registration/{}".format(scale)] = loss_registration
            losses["loss_refine/{}".format(scale)] = loss_refine

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, metrics=None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # Log metrics if provided
        if metrics is not None and len(metrics) > 0:
            metrics_prefix = "metrics"
            for m, v in metrics.items():
                writer.add_scalar("{}/{}".format(metrics_prefix, m), v, self.step)

        # Log learned camera intrinsics if enabled
        if self.learnable_K and hasattr(self, "learnable_K_params"):
            fx = self.learnable_K_params[0, 0].item()
            fy = self.learnable_K_params[1, 1].item()
            cx = self.learnable_K_params[0, 2].item()
            cy = self.learnable_K_params[1, 2].item()
            writer.add_scalar("intrinsics/fx", fx, self.step)
            writer.add_scalar("intrinsics/fy", fy, self.step)
            writer.add_scalar("intrinsics/cx", cx, self.step)
            writer.add_scalar("intrinsics/cy", cy, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:

                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)
            
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
        
        # Save the loaded config file
        if hasattr(self, 'endoda3_model_config'):
            config_save_path = os.path.join(models_dir, 'endoda3_model_config.yaml')
            OmegaConf.save(self.endoda3_model_config, config_save_path)
            print(f"Saved model config to: {config_save_path}")

    def save_model(self, mode='epoch'):
        """Save model weights to disk
        """
        if mode == 'epoch':
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        elif mode == 'last':
            save_folder = os.path.join(self.log_path, "models", "weights_last")
        elif mode == 'best_depth':
            save_folder = os.path.join(self.log_path, "models", "best_depth")
        elif mode == 'best_pose':
            save_folder = os.path.join(self.log_path, "models", "best_pose")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'depth_model':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def _save_best_metrics_info(self):
        """Save best metrics criteria, values, and epochs to exp_dir
        """
        best_metrics_info = {
            'best_depth_metric': self.best_depth_metric,
            'best_depth_value': self.best_depth_value,
            'best_depth_epoch': self.best_depth_epoch,
            'best_pose_metric': self.best_pose_metric,
            'best_pose_value': self.best_pose_value,
            'best_pose_epoch': self.best_pose_epoch,
        }
        
        # Save to JSON file in exp_dir
        info_path = os.path.join(self.log_path, "best_metrics_info.json")
        with open(info_path, 'w') as f:
            json.dump(best_metrics_info, f, indent=2)
        
        # Also print summary
        print(f"Best metrics info saved to: {info_path}")

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            info_n = self.models[n].load_state_dict(model_dict, strict=False)
            from utils.load_models import print_state_dict_info
            print_state_dict_info(info_n, model_name=n, max_levels=4)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")

