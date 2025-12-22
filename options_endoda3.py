from __future__ import absolute_import, division, print_function

import os
import argparse
import time

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def str2bool(v):
     if isinstance(v, bool):
          return v
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="EndoDAC options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "endovis_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # Model options
        self.parser.add_argument("--pretrained_path",
                                 type=str,
                                 help="pretrained weights path; load with hf name.",
                                 default='depth-anything/da3-base')
     #    self.parser.add_argument("--backbone_size",
     #                             type=str,
     #                             help="size of pretrained Dinov2 backbone",
     #                             choices=["small", "base", "large", "giant"],
     #                             default="base")
     #    self.parser.add_argument("--lora_type",
     #                             type=str,
     #                             help="which lora type use for the model",
     #                             choices=["lora", "dvlora", "none"],
     #                             default="dvlora")
     #    self.parser.add_argument("--lora_rank",
     #                             type=int,
     #                             help="the rank of lora",
     #                             default=4)
        self.parser.add_argument("--warm_up_step",
                                 type=int,
                                 help="warm up step",
                                 default=20000)
        # self.parser.add_argument("--residual_block_indexes",
        #                          nargs="*",
        #                          type=int,
        #                          help="indexes for residual blocks in vitendodepth encoder",
        #                          default=[2,5,8,11])
        # self.parser.add_argument("--include_cls_token",
        #                          type=str2bool,
        #                          help="includes the cls token in the transformer blocks",
        #                          default=True)
        self.parser.add_argument("--learn_intrinsics",
                                 help="if set, learns the camera intrinsics with a seperate decoder",
                                 action="store_true")

        # DA3 model options
        self.parser.add_argument("--endoda3_model_config",
                                 type=str,
                                 help="path to depth model config file (YAML)",
                              #    default="/mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/networks/configs/endo-da3-all-wowrapper.yaml",
                                 )
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared", "da3_internal"])
        self.parser.add_argument("--depth_model_type",
                                 type=str,
                                 help="depth model type (placeholder)",
                                 default="depthanything3",
                                 choices=["depthanything3","endodac"])
        self.parser.add_argument("--da3_depth_regression_target",
                                 type=str,
                                 help="depth regression target",
                                 default="depth2disp",
                                 choices=["disp", "depth2disp"])
        self.parser.add_argument("--of_model_type",
                                 type=str,
                                 help="optical flow model type",
                                 default="separate_resnet",
                                 choices=["separate_resnet", "raft"])
        self.parser.add_argument("--raft_num_flow_updates",
                                 type=int,
                                 help="number of flow updates for RAFT",
                                 default=12)
        self.parser.add_argument("--raft_max_disp",
                                 type=float,
                                 help="maximum displacement for RAFT flow clamping; if None, no clamping is applied",
                                 default=None)
        self.parser.add_argument("--raft_trainable_modules",
                                 nargs="*",
                                 type=str,
                                 help="RAFT feature_encoder modules to make trainable (e.g., 'convnormrelu', 'layer1', 'layer2_0'). "
                                      "Special: 'layer2_0' means first block of layer2. Default [] means RAFT is frozen.",
                                 default=[])
        self.parser.add_argument("--use_raft_multi_iters",
                                 help="if set, uses RAFT multi-iteration outputs for different scales",
                                 action="store_true")
        self.parser.add_argument("--raft_multi_iters",
                                 nargs="+",
                                 type=int,
                                 help="RAFT iteration steps to use for each scale (e.g., [2,5,8,11] for 4 scales)",
                                 default=[2, 5, 8, 11])
        self.parser.add_argument("--af_model_type",
                                 type=str,
                                 help="affine transform model type",
                                 default="separate_resnet",
                                 choices=["separate_resnet"])
        self.parser.add_argument("--k_model_type",
                                 type=str,
                                 help="intrinsics model type",
                                 default="mlp_with_pn_bottleneck_ipt",
                                 choices=["mlp_with_pn_bottleneck_ipt", "da3_internal"])
        self.parser.add_argument("--enable_seq_inputs",
                                 help="if set, enables sequential multi-frame input mode (requires depth_model_type=depthanything3)",
                                 action="store_true")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default=time.strftime('%m%d_%H%M'))
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endovis", "hamlyn"],
                                 default="endovis")
        self.parser.add_argument("--train_data_file",
                                 nargs="*",
                                 help="filename(s) for training data split (relative to splits/{split}/). Can be a single file or multiple files (space-separated)",
                                 default=["train_files.txt"])
        self.parser.add_argument("--val_data_file",
                                 nargs="*",
                                 help="filename(s) for validation data split (relative to splits/{split}/). Can be a single file or multiple files (space-separated)",
                                 default=["val_files.txt"])
        self.parser.add_argument("--test_data_file",
                                 nargs="*",
                                 help="filename(s) for test data split (relative to splits/{split}/). Can be a single file or multiple files (space-separated)",
                                 default=["test_files.txt"])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="endovis",
                                 choices=["endovis", "hamlyn"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--position_smoothness",
                                 type=float,
                                 help="registration smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--transform_constraint",
                                 type=float,
                                 help="transform constraint weight",
                                 default=0.01)
        self.parser.add_argument("--transform_smoothness",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--of_supervised_with_which",
                                 type=str,
                                 help="what to use for optical flow supervision: 'outputs_refined' (outputs['refined']) or 'inputs_color'",
                                 default="outputs_refined",
                                 choices=["outputs_refined", "inputs_color"])
        self.parser.add_argument("--posedepth_supervised_with_which",
                                 type=str,
                                 help="what to use for pose/depth supervision: 'outputs_refined' (outputs['refined'])",
                                 default="outputs_refined",
                                 choices=["outputs_refined"])
        # self.parser.add_argument("--reconstruction_constraint",
        #                          type=float,
        #                          help="consistency constraint weight",
        #                          default=0.2)
        # self.parser.add_argument("--reflec_constraint",
        #                          type=float,
        #                          help="epipolar constraint weight",
        #                          default=0.2)
        # self.parser.add_argument("--reprojection_constraint",
        #                          type=float,
        #                          help="geometry constraint weight",
        #                          default=1)
        # self.parser.add_argument("--reproj_supervise_type",
        #                          type=str,
        #                          help="type of reprojection supervision: 'color_warp' or 'reprojection_color_warp'",
        #                          default="reprojection_color_warp",
        #                          choices=["color_warp", 
        #                          "reprojection_color_warp",
        #                          "afstyle_color_warp",
        #                          "paba_color_warp"])
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=150.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        
        # Pose Net setting: we use the setting here if we use external pose net rather the sub_module cam_dec for endoDA3
        self.parser.add_argument("--trans_scale_factor",
                                 type=float,
                                 help="translation scale factor",
                                 default=0.001)
        self.parser.add_argument("--rot_scale_factor",
                                 type=float,
                                 help="rotation scale factor",
                                 default=0.001)
        self.parser.add_argument("--rot_representation",
                                 type=str,
                                 help="pose net type",
                                 default="angle_axis",
                                 choices=["9D", "6D", "angle_axis", "euler", "quat"])
        self.parser.add_argument("--explicit_bias_init_6d9d",
                                 help="enable explicit initialization for 6D/9D rotation representations",
                                 action="store_true",
                                 default=False)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--seed",
                                 type=int,
                                 help="random seed for reproducibility",
                                 default=42)
        self.parser.add_argument("--compute_depth_metrics",
                                 help="if set computes depth metrics during validation",
                                 action="store_true")
        self.parser.add_argument("--compute_pose_metrics",
                                 help="if set computes pose metrics during validation",
                                 action="store_true")
        self.parser.add_argument("--val_full_eval",
                                 help="if set, run validation on the full val set instead of a single batch",
                                 action="store_true")
        self.parser.add_argument("--use_perframe_gt_K",
                                 help="if set, uses per-frame GT K matrices (K_per_frame) instead of default K",
                                 action="store_true")

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["position_encoder", "position"])

        # LOGGING options
        self.parser.add_argument("--exp_suffix",
                                 type=str,
                                 help="experiment suffix to append to model_name when logging",
                                 default="")
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=400)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # OVERFITTING options
        self.parser.add_argument("--of_samples",
                                 help="if set enables overfitting mode with limited samples",
                                 action="store_true")
        self.parser.add_argument("--of_samples_num",
                                 type=int,
                                 help="number of samples to use for overfitting",
                                 default=100)

        # EVALUATION options
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endodac", "afsfm"],
                                 default="endodac")
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="endovis",
                                 choices=["hamlyn", "c3vd", "endovis"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--visualize_depth",
                                 help="if set saves visualized depth map",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--load_gt_from_npz",
                                 help="if set loads GT depths from npz file instead of on the fly",
                                 action="store_true")
        self.parser.add_argument("--save_poses_root",
                                 help="root directory to save pose predictions",
                                 type=str,
                                 default=None)
        self.parser.add_argument("--track_lengths",
                                 help="track lengths for pose evaluation (can specify multiple, e.g., --track_lengths 5 10)",
                                 nargs="+",
                                 type=int,
                                 default=[5])

        # EVALUATION options
        self.parser.add_argument("--save_recon",
                                 help="if set saves reconstruction files",
                                 action="store_true")
        
        # VISUALIZATION options
        self.parser.add_argument("--eval_model_appendix",
                                 help="appendix to add to saved pose prediction filename",
                                 type=str,
                                 default="")
        self.parser.add_argument("--plot_xyz_rpy",
                                 help="if set plots xyz and rpy components",
                                 action="store_true")
        self.parser.add_argument("--plot_conf",
                                 help="if set plots confidence values",
                                 action="store_true")
        self.parser.add_argument("--debug_only",
                                 help="if set limits number of frames for debugging",
                                 action="store_true")
        self.parser.add_argument("--plot_num",
                                 help="number of frames to plot when debug_only is set",
                                 type=int,
                                 default=None)
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
    
    def parse_notebook(self, args):
        """Parse arguments from a list (for notebook use)
        
        Args:
            args: List of argument strings (e.g., ['--batch_size', '2', '--num_workers', '1'])
                  For multi-value arguments (nargs="*" or nargs="+"), can pass as:
                  ['--raft_trainable_modules', 'convnormrelu layer1 layer2_0'] (space-separated)
                  or ['--raft_trainable_modules', 'convnormrelu', 'layer1', 'layer2_0'] (separate args)
        
        Returns:
            Parsed options object
        """
        # Get all actions that accept multiple values (nargs="*" or nargs="+")
        multi_value_actions = set()
        for action in self.parser._actions:
            if action.nargs in ('*', '+') and action.dest != 'help':
                # Get all option strings for this action
                for option_string in action.option_strings:
                    multi_value_actions.add(option_string)
        
        # Preprocess args to handle space-separated values for multi-value arguments
        processed_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            processed_args.append(arg)
            
            # Check if this is a multi-value argument and has a next value
            if arg in multi_value_actions and i + 1 < len(args):
                next_arg = args[i + 1]
                # If next arg is not another option and contains spaces, split it
                if not next_arg.startswith('-') and ' ' in next_arg:
                    # Split space-separated string into separate arguments
                    processed_args.extend(next_arg.split())
                    i += 1  # Skip the original space-separated string
                # If it's already separate values, they'll be handled normally
            
            i += 1
        
        self.options = self.parser.parse_args(processed_args)
        return self.options