from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import sys
from pathlib import Path

# Setup path for depth_anything_3 imports
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))

# from third_party.EndoDAC.models.endodac import endodac, mark_only_part_as_trainable
from networks.endo_da3 import mark_only_part_as_trainable_v2, EndoDepthAnything3Net
from depth_anything_3.cfg import create_object, load_config
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
from utils.util import set_seed, readlines, normalize_image, sec_to_hm_str, disp_to_depth
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
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
import torch

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


class EndoDepthAnything3NetWrapper(torch.nn.Module):
    """
    Wrapper for EndoDepthAnything3Net to adapt interface for trainer_endoda3.
    Converts between (B, 3, H, W) input/output format and EndoDepthAnything3Net's (B, N, 3, H, W) format.
    Also converts depth to disparity and creates multi-scale outputs.
    """
    def __init__(self, model, min_depth=0.1, max_depth=150.0, scales=[0, 1, 2, 3]):
        super().__init__()
        self.model = model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scales = scales
        # Expose lora_type from wrapped model for compatibility
        self.lora_type = getattr(model, 'lora_type', 'none')
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Dictionary with keys ("disp", scale) for each scale
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
        assert C == 3, f"C shape: {C}"
        # Forward through EndoDepthAnything3Net
        # DPT DEPTH: B S H W 1; 
        # DPT depth_Conf: B S H W 

        # DualDPT DEPTH: B S H W;
        # DualDPT depth_Conf: B S H W;
        # DualDPT RAY: B S H W 6;
        # DualDPT RAY_CONF: B S H W ;
        # extrinsics: B S 3 4 ;
        # intrinsics: B S 3 3 ;

        # output is a dict
        output = self.model(x_mv, extrinsics=None, intrinsics=None, 
                           export_feat_layers=[], infer_gs=False, use_ray_pose=False)
        # Depth output: (B, S, H, W) if  DualDPT Head
        # Depth output: (B, S, H, W, 1) if DPT Head
        if isinstance(self.model.head, DPT):
            # update the dict
            # B,S,H,W,1 -> B,S,H,W
            output.depth = output.depth.squeeze(-1)

        depth = output.depth # extract key

        assert depth.dim() == 4, f"depth shape: {depth.shape}"
        assert depth.shape[1] == 1, f"depth shape: {depth.shape}"  

        # Extract depth: output.depth is (B, S, H, W) where S=1 for monocular
        # print(f"depth shape: {output.depth.shape}")
        # print(f"conf shape: {output.depth_conf.shape}")
        # print(f"extrinsics shape: {output.extrinsics.shape}")
        # print(f"intrinsics shape: {output.intrinsics.shape}")
        
 

        depth_clamped = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        disp = 1.0 / depth_clamped # (B, S, H, W)
     
        # Interpolate to match input image size if needed
        if disp.shape[-2:] != (256, 320):
            disp = F.interpolate(disp, size=(256, 320), mode="bilinear", align_corners=True)



        # Indrect create multi-scale outputs: not actually enable multi-resolution depth direcly from head 
        outputs = {}
        for scale in self.scales:
            if scale == 0:
                # Original scale - already at input resolution
                outputs[("disp", scale)] = disp
            else:
                # Downscale for other scales
                h_scale = H // (2 ** scale)
                w_scale = W // (2 ** scale)
                disp_scale = F.interpolate(disp, size=(h_scale, w_scale), 
                                          mode="bilinear", align_corners=True)
                outputs[("disp", scale)] = disp_scale
        
        if not single_frame_input:
            assert 0, 'not tested'
            # fuse B and S to construct outputs dimension B_S, H, W
            # then format as the pipeline required dim 1 for channel
            # B S H W -> B*S 1 H W
            outputs = {k: v.view(B*S,H, W).unsqueeze(1) for k, v in outputs.items()}

        return outputs


class RAFTWrapper(torch.nn.Module):
    """
    Wrapper for RAFT to match PositionDecoder interface.
    Generates multi-scale flows from single RAFT output.
    """
    def __init__(self, raft_model, scales, max_disp=None):
        super().__init__()
        self.raft_model = raft_model
        self.scales = scales
        self.max_disp = max_disp
        
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

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Construct models in order: depth -> pose -> k -> (of, af at end)
        self.construct_depth_model()
        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        if self.use_pose_net:
            self.construct_pose_model()
            # Add pose_encoder parameters if it exists
            if "pose_encoder" in self.models:
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())
            
            if self.opt.learn_intrinsics:
                self.construct_k_model()
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
        train_file = getattr(self.opt, 'train_data_file', 'train_files.txt') if not getattr(self.opt, 'of_samples', False) else getattr(self.opt, 'val_data_file', 'val_files.txt')
        val_file = getattr(self.opt, 'val_data_file', 'val_files.txt')
        test_file = getattr(self.opt, 'test_data_file', 'test_files.txt')
        
        train_fpath = os.path.join(splits_dir, train_file)
        val_fpath = os.path.join(splits_dir, val_file)
        test_fpath = os.path.join(splits_dir, test_file)
        
        train_filenames = readlines(train_fpath)
        val_filenames = readlines(val_fpath)
        test_filenames = readlines(test_fpath)
        img_ext = '.png'  

        if getattr(self.opt, 'of_samples', False):
            of_samples_num = getattr(self.opt, 'of_samples_num', 100)
            train_filenames = train_filenames[:of_samples_num]
            val_filenames = val_filenames[:of_samples_num]
            test_filenames = test_filenames[:of_samples_num]
            print("Overfitting mode: using {} Trn samples".format(len(train_filenames)))
            print("Overfitting mode: using {} Val samples".format(len(val_filenames)))
            print("Overfitting mode: using {} Test samples".format(len(test_filenames)))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # is_train = not getattr(self.opt, 'of_samples', False) # can be used for compute depth err
        shuffle = not getattr(self.opt, 'of_samples', False)  # Fixed order for overfitting
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
            load_gt_poses=False)
        self.test_loader = DataLoader(
            test_dataset, 1, False,
            num_workers=1, pin_memory=True, drop_last=True,)
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
        print("There are {:d} training items, {:d} validation items and {:d} testing items\n".format(
            len(train_dataset), len(val_dataset), len(test_dataset)))

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
        # Initialize EndoDepthAnything3Net from config file
        endoda3_model_config_path = self.opt.endoda3_model_config 
        assert os.path.exists(endoda3_model_config_path), f"Config file not found: {endoda3_model_config_path}"
        print(f"Loading depth model setting from config: {endoda3_model_config_path}")
        endoda3_model_config = load_config(endoda3_model_config_path)
        depth_model_base = create_object(endoda3_model_config)
        
        # Wrap the model to adapt interface
        self.models["depth_model"] = EndoDepthAnything3NetWrapper(
            depth_model_base,
            min_depth=self.opt.min_depth,
            max_depth=self.opt.max_depth,
            scales=self.opt.scales
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

    def construct_pose_model(self):
        """Construct and initialize the pose model.
        Supports different pose model types: separate_resnet, shared, posecnn.
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

        self.models["pose"].to(self.device)

    def construct_k_model(self):
        """Construct and initialize the intrinsics (K) model.
        Supports mlp_with_pn_bottleneck_ipt which uses pose network's intermediate feature.
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
            self.models["position"] = RAFTWrapper(
                raft_model=raft_model,
                scales=self.opt.scales,
                max_disp=getattr(self.opt, 'raft_max_disp', None)
            )
            self.models["position"].to(self.device)
        else:
            raise ValueError(f"Unsupported of_model_type: {self.opt.of_model_type}")

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
        for param in self.models["position"].parameters():
            param.requires_grad = True

        for param in self.models["depth_model"].parameters():
            param.requires_grad = False
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["pose"].parameters():
            param.requires_grad = False
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["transform"].parameters():
            param.requires_grad = False
        if self.opt.learn_intrinsics:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = False
            
        if "position_encoder" in self.models:
            self.models["position_encoder"].train()
        self.models["position"].train()

        self.models["depth_model"].eval()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].eval()

    def set_train(self):
        """Convert all models to training mode
        """
        if "position_encoder" in self.models:
            for param in self.models["position_encoder"].parameters():
                param.requires_grad = False
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

        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["transform"].parameters():
            param.requires_grad = True
        if self.opt.learn_intrinsics:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = True

        if "position_encoder" in self.models:
            self.models["position_encoder"].eval()
        self.models["position"].eval()

        self.models["depth_model"].train()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        if "pose_encoder" in self.models:
            self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        if "position_encoder" in self.models:
            self.models["position_encoder"].eval()
        self.models["position"].eval()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model(mode='epoch')            
            
            # if self.epoch == 0:
            #     rmse, a1 = self.run_epoch_eval()
            #     self.save_model(mode='epoch')
            # else:
            #     rmse_new, a1_new = self.run_epoch_eval()
            #     if rmse_new < rmse:
            #         rmse = rmse_new
            #         # a1 = a1_new
            #         self.save_model(mode='epoch')
            # self.save_model(mode='last')
            
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
                if getattr(self.opt, 'compute_metrics', False):
                    depth_metrics = compute_depth_metrics(inputs, outputs)
                    if depth_metrics:
                        metrics.update(depth_metrics)
                
                pose_metrics = compute_pose_metrics(inputs, outputs, self.opt.frame_ids)
                if pose_metrics:
                    metrics.update(pose_metrics)

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses, metrics=metrics if metrics else None)
                self.val()

            self.step += 1
            
        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    def run_epoch_eval(self):
        """Run a single epoch of evaluation
        """

        print("Evaluating")
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 150
        
        self.set_eval()
        pred_depths = []
        for batch_idx, inputs in enumerate(self.test_loader):
            input_color = inputs[("color", 0, 0)].cuda()

            if self.opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            # output = self.models["depth"](self.models["encoder"](input_color))
            output = self.models["depth_model"](input_color)
            _, pred_depth = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_depth = pred_depth[:, 0].cpu().detach().numpy()
            pred_depths.append(pred_depth)
            
        pred_depths = np.concatenate(pred_depths)
        
        errors = []
        ratios = []
        
        for i in range(pred_depths.shape[0]):
            # gt_depth = self.gt_depths[i]
            # obtain gt_depth from inputs
            gt_depth = inputs[("depth_gt", 0, 0)].cpu().detach().numpy().squeeze()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = pred_depths[i]
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            # print(pred_depth.max(), pred_depth.min())
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            # errors.append(compute_errors(gt_depth, pred_depth))
            errors.append(compute_depth_errors(gt_depth, pred_depth))
        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        writer = self.writers["train"]
        for i in range(len(mean_errors)):
            writer.add_scalar(self.depth_metric_names[i], mean_errors[i], self.epoch)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        
        self.set_train()
        
        return mean_errors[2], mean_errors[4]
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
                loss_registration += (
                    self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], outputs[("refined", scale, frame_id)].detach()) * occu_mask_backward).sum() / occu_mask_backward.sum()

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
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def get_K_invK(self, inputs, outputs, scale, batch_size):
        """
        Get camera intrinsics K and inv_K for a given scale.
        Priority: learn_intrinsics (predicted) > learnable_K (optimizable) > inputs
        """
        if self.opt.learn_intrinsics and ('K', scale) in outputs:
            # Use predicted intrinsics from intrinsics_head
            return outputs[('K', scale)], outputs[('inv_K', scale)]
        elif self.learnable_K:
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

    def predict_poses(self, inputs, disps):
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
                                                                                            

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    rot_output, translation, intermediate_feature = self.models["pose"](pose_inputs, ret_intermediate_feat=True)

                    if self.opt.learn_intrinsics:
                        cam_K = self.models['intrinsics_head'](
                        intermediate_feature, self.opt.width, self.opt.height)
                        inv_K = torch.inverse(cam_K)
                        outputs[('K', 0)] = cam_K
                        outputs[('inv_K', 0)] = inv_K
                    
                    rot_representation = getattr(self.opt, 'rot_representation', 'angle_axis')
                    if rot_representation == "angle_axis":
                        outputs[("axisangle", 0, f_i)] = rot_output
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            rot_output[:, 0], translation[:, 0])
                    elif rot_representation == "6D":
                        outputs[("rot6d", 0, f_i)] = rot_output
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_6D(
                            rot_output[:, 0], translation[:, 0])
                    elif rot_representation == "9D":
                        outputs[("rot9d", 0, f_i)] = rot_output
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_9D(
                            rot_output[:, 0], translation[:, 0])
                    elif rot_representation == "quat":
                        outputs[("quat", 0, f_i)] = rot_output
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_quat(
                            rot_output[:, 0], translation[:, 0])
                    elif rot_representation == "euler":
                        outputs[("euler", 0, f_i)] = rot_output
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters_euler(
                            rot_output[:, 0], translation[:, 0])
                    else:
                        raise ValueError(f"Unsupported rotation representation: {rot_representation}")
                    
                    # Optionally replace rotation with GT relative rotation if available
                    if self.replace_with_gt_rel_rotation:
                        if ("gt_c2w_poses", 0) in inputs and ("gt_c2w_poses", f_i) in inputs:
                            gt_tgt_abs_poses = inputs[("gt_c2w_poses", 0)]  # (B, 4, 4)
                            gt_src_abs_poses = inputs[("gt_c2w_poses", f_i)]  # (B, 4, 4)
                            gt_tgt2src_rel_poses = torch.inverse(gt_src_abs_poses) @ gt_tgt_abs_poses
                            outputs[("cam_T_cam", 0, f_i)][:, :3, :3] = gt_tgt2src_rel_poses[:, :3, :3]
                            # If desired, translation could also be replaced; keeping network translation for now.

                    outputs[("translation", 0, f_i)] = translation
                    
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0
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
                pix_coords = self.project_3d[source_scale](
                    cam_points, cam_K, T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, cam_K, T)

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
                
                loss_reprojection += (
                    self.compute_reprojection_loss(outputs[("color", frame_id, scale)], outputs[("refined", scale, frame_id)]) * occu_mask_backward).sum() / occu_mask_backward.sum()  
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
        """Validate the model on a single minibatch
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

                    if getattr(self.opt, 'compute_metrics', False):
                        depth_metrics = compute_depth_metrics(inputs, outputs)
                        if depth_metrics:
                            _accum(metrics_accum, depth_metrics)

                   
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
            if report_quantile_pose_err:
                q = [0.25,0.5,0.75]
                for q_i in q:
                    metrics_trans_ang_err_raw = {k + f'_Q{q_i}': np.quantile(v, q_i) for k, v in metrics_trans_ang_err_raw_accum.items()} if metrics_trans_ang_err_raw_accum else None
                    metrics_rot_err_raw = {k + f'_Q{q_i}': np.quantile(v, q_i) for k, v in metrics_rot_err_raw_accum.items()} if metrics_rot_err_raw_accum else None
                    metrics.update(metrics_trans_ang_err_raw)
                    metrics.update(metrics_rot_err_raw)


            if last_inputs is not None:
                self.log("val", last_inputs, last_outputs, last_losses, metrics=metrics)
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
                if getattr(self.opt, 'compute_metrics', False):
                    depth_metrics = compute_depth_metrics(inputs, outputs)
                    if depth_metrics:
                        metrics.update(depth_metrics)
                
                pose_metrics = compute_pose_metrics(inputs, outputs, self.opt.frame_ids)
                if pose_metrics:
                    metrics.update(pose_metrics)
                
                self.log("val", inputs, outputs, losses, metrics=metrics if metrics else None)
                del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

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

    def save_model(self, mode='epoch'):
        """Save model weights to disk
        """
        if mode == 'epoch':
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        elif mode == 'last':
            save_folder = os.path.join(self.log_path, "models", "weights_last")
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

