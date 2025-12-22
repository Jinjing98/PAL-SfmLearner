from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2
import torch

from .mono_dataset import MonoDataset
from utils import (
    get_gt_poses,
    get_poses_for_frames,
    get_gt_Ks,
    get_k_for_frames,
)

DEFAULT_D7K4_SCENE_POINTS_DIR='/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Depth/dataset_7/keyframe_4/data/scene_points'
DATA_PATH='/mnt/cluster/datasets/SCARED/'
DEPTH_PATH='/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Depth/'

class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        

        self.gt_Ks_dict_registered = {}



        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.dataset_name = 'SCARED'

        # Load GT depths from npz file for validation (is_train=False)
        self.gt_depths_val = None
        # if not self.is_train:
        #     # splits_dir is at the same level as datasets directory
        #     splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")
        #     gt_path = os.path.join(splits_dir, 'endovis', "gt_depths_val.npz")
        #     if os.path.exists(gt_path):
        #         print("Loading GT depths from {}".format(gt_path))
        #         self.gt_depths_val = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        #         print("Loaded {} GT depth maps".format(len(self.gt_depths_val)))

    def check_depth(self):
        # Return True for validation to enable GT depth loading
        return not self.is_train
        # return True

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color



class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, load_gt_poses=True, load_gt_Ks=True, load_gt_depth=False, depth_offline_loading=False, **kwargs):
        # Set load_gt_depth before super().__init__() so check_depth() can access it
        self.load_gt_depth = load_gt_depth
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)
        self.load_gt_poses = load_gt_poses
        self.load_gt_Ks = load_gt_Ks
        self.depth_offline_loading = depth_offline_loading
        self.traj_data_root = DATA_PATH
        self.K_data_root = DEPTH_PATH
        self.trans_scale_gt_traj = 1000  # m to mm
        if self.load_gt_poses:
            self.trajs_dict = get_gt_poses(
                self.filenames,
                self.traj_data_root,
                trans_scale=self.trans_scale_gt_traj
            )
            print(f"Loaded {len(self.trajs_dict)} trajectories")
        if self.load_gt_Ks:
            self.gt_Ks_dict_registered = get_gt_Ks(
                self.filenames,
                self.K_data_root
            )
            print(f"Loaded {len(self.gt_Ks_dict_registered)} camera intrinsics")
        
        # Load GT depths from npz file if depth_offline_loading is True
        if self.depth_offline_loading and self.load_gt_depth:
            splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")
            gt_path = os.path.join(splits_dir, 'endovis', "gt_depths.npz")
            if os.path.exists(gt_path):
                print("Loading GT depths from {}".format(gt_path))
                self.gt_depths_val = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
                # also wrongly assert for of_samples
                # assert len(self.gt_depths_val) == len(self.filenames), "Number of GT depth maps does not match number of filenames {} vs {}".format(len(self.gt_depths_val), len(self.filenames))
                print("Loaded {} GT depth maps".format(len(self.gt_depths_val)))
            else:
                print("WARNING: GT depths file not found at {}. Online loading will be used.".format(gt_path))
                self.gt_depths_val = None

    def check_depth(self):
        # Hard-controlled load_gt_depth parameter
        return self.load_gt_depth

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_02/data", f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        # If npz data is available, return None (will be loaded from npz in __getitem__)
        if self.gt_depths_val is not None:
            return None
        
        # Otherwise, load from file
        #///////////////////////////////////////////////
        # f_str = "scene_points{:06d}.tiff".format(frame_index)

        # depth_path = os.path.join(
        #     self.data_path,
        #     folder,
        #     "image_0{}/data/groundtruth".format(self.side_map[side]),
        #     f_str)
        #///////////////////////////////////////////////

        def parse_folder_tokens(folder_str):
            # folder_str like 'dataset3/keyframe4' -> (3, 4)
            dataset_part, keyframe_part = folder_str.split('/')
            dataset_num = int(dataset_part.replace('dataset', ''))
            keyframe_num = int(keyframe_part.replace('keyframe', ''))
            return dataset_num, keyframe_num
        dataset_num, keyframe_num = parse_folder_tokens(folder)
        dataset_dir_candidates = [
            "dataset_{:02d}".format(dataset_num),
            "dataset_{}".format(dataset_num),
            "dataset{}".format(dataset_num),
        ]
        keyframe_dir_candidates = [
            "keyframe_{}".format(keyframe_num),
            "keyframe{:d}".format(keyframe_num),
        ]
        f_str = "scene_points{:06d}.tiff".format(frame_index - 1)
        depth_path = None

        from utils import map_traj_search, map_traj_search_SCARED_DEPTH

        traj_folder = map_traj_search(folder, DATA_PATH)
        depth_path = os.path.join(traj_folder, 'data', 'scene_points', f_str)
        # depth_path = os.path.join(depth_folder, 'data', f_str)
        # print(f"Depth path sanity: {depth_path} for folder: {folder} frame: {frame_index}")
        if not os.path.exists(depth_path):
            print(f"Depth file {depth_path} does not exist. d7k4?")
            depth_path = os.path.join(DEFAULT_D7K4_SCENE_POINTS_DIR, f_str)
            assert os.path.exists(depth_path), f"Depth file {depth_path} does not exist."

        # completely use depth from SCARED_Depth dataset
        # depth_folder = map_traj_search_SCARED_DEPTH(folder, DEPTH_PATH)
        # depth_path = os.path.join(depth_folder, 'data', 'scene_points', f_str)

        depth_gt = cv2.imread(depth_path, 3)
        if depth_gt is None:
            print('Depth file is broken/None in path {} for folder: {} frame: {}'.format(depth_path, folder, frame_index))
            print('We set broken depth to zeros...')
            # assert False, 'Depth file is broken/None in path {} for folder: {} frame: {}'.format(depth_path, folder, frame_index)
            return np.zeros((1024, 1280))
            # return None
        
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def __getitem__(self, index):
        """Override to add GT depth loading from gt_depths_val.npz for validation"""
        # Call parent __getitem__ to get all standard inputs
        inputs = super(SCAREDRAWDataset, self).__getitem__(index)
        
        # Load GT depth from npz file if available (for validation only)
        if self.gt_depths_val is not None and index < len(self.gt_depths_val):
            gt_depth = self.gt_depths_val[index]  # (H_gt, W_gt)
            # Convert to tensor and add channel dimension: (1, H_gt, W_gt)
            inputs[("depth_gt", 0, 0)] = torch.from_numpy(np.expand_dims(gt_depth, 0).astype(np.float32))

        # Parse folder from filename
        line = self.filenames[index].split()
        assert len(line) == 3, 'Expected 3 elements in line: {}'.format(line)
        folder = line[0]
        frame_index = int(line[1])
        
        # Load GT poses if enabled
        if getattr(self, "load_gt_poses", False):
            # SCARED trajectories are 0-indexed, images start at 1 => offset -1
            offset = -1
            for i in self.frame_idxs:
                if i == "s":
                    continue
                inputs[("gt_c2w_poses", i)] = torch.from_numpy(
                    get_poses_for_frames(self.trajs_dict, folder, [frame_index + i], offset=offset)
                )
        
        # Load GT K if enabled
        if getattr(self, "load_gt_Ks", False):
            # Get GT K (3x3) for this folder - this is in pixel coordinates for raw resolution (1024x1280)
            K_gt_3x3_raw = get_k_for_frames(self.gt_Ks_dict_registered, folder)  # (3, 3)
            
            # Scale from raw resolution to current resolution
            raw_height, raw_width = 1024, 1280
            x_scale, y_scale = self.width / raw_width, self.height / raw_height
            
            # Scale 3x3 K: fx and cx scale by x_scale, fy and cy scale by y_scale
            K_gt_3x3 = K_gt_3x3_raw.copy()
            K_gt_3x3[0, :] *= x_scale  # fx, 0, cx
            K_gt_3x3[1, :] *= y_scale  # 0, fy, cy
            
            # Convert to 4x4
            K_gt_4x4 = np.eye(4, dtype=np.float32)
            K_gt_4x4[:3, :3] = K_gt_3x3
            
            # Process K for each scale and frame_id, save as K_per_frame (like poses)
            for scale in range(self.num_scales):
                K = K_gt_4x4.copy()
                K[0, :] //= (2 ** scale)
                K[1, :] //= (2 ** scale)
                
                inv_K = np.linalg.pinv(K)
                
                # Save K for each frame_id (like poses)
                for i in self.frame_idxs:
                    if i == "s":
                        continue
                    inputs[("K_per_frame", i, scale)] = torch.from_numpy(K)
                    inputs[("inv_K_per_frame", i, scale)] = torch.from_numpy(inv_K)
        
        return inputs


if __name__ == "__main__":
    # Test GT depth loading for validation dataset
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from utils import readlines
    
    # Configuration for testing
    data_path = "/mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/"
    height = 256
    width = 320
    frame_ids = [0, -1, 1]
    split = "endovis"
    iterate_through_all_files = True
    # iterate_through_all_files = False
    
    # Read validation filenames
    splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")
    val_fpath = os.path.join(splits_dir, split, "val_files.txt")
    val_fpath = os.path.join(splits_dir, split, "d6_kf2.txt")
    val_fpath = os.path.join(splits_dir, split, "d6_kf2.txt")
    val_fpath = os.path.join(splits_dir, split, "train_files.txt")
    val_fpath = os.path.join(splits_dir, split, "test_files.txt")
    
    if not os.path.exists(val_fpath):
        print("Error: Validation split file not found at {}".format(val_fpath))
        sys.exit(1)
    
    val_filenames = readlines(val_fpath)
    
    print("=" * 60)
    print("Testing GT Depth Loading for Validation Dataset")
    print("=" * 60)
    print("Validation filenames: {}".format(len(val_filenames)))
    
    # Create validation dataset
    try:
        val_dataset = SCAREDRAWDataset(
            data_path, val_filenames, height, width,
            frame_ids, 4, is_train=False, img_ext='.png',
            load_gt_poses=False,# we can not load gt poses for d7k4 in test.txt
            load_gt_Ks=True,
            load_gt_depth=True,
            depth_offline_loading=os.path.basename(val_fpath) == 'test_files.txt',  # Load from gt_depths.npz when test_files.txt for perfect alignment
        )
        print("Dataset created successfully!")
        # print("GT depths loaded: {}".format(val_dataset.gt_depths_val is not None))
        
        # if val_dataset.gt_depths_val is not None:
        #     print("GT depths shape: {}".format(val_dataset.gt_depths_val.shape))
        #     print("Number of GT depth maps: {}".format(len(val_dataset.gt_depths_val)))
        #     print("GT depth dtype: {}".format(val_dataset.gt_depths_val.dtype))
        #     print("GT depth min/max: {:.3f} / {:.3f}".format(
        #         val_dataset.gt_depths_val.min(), val_dataset.gt_depths_val.max()))
        # else:
        #     print("WARNING: GT depths not loaded! Check if gt_depths_val.npz exists.")
        
        # Test loading a sample
        num_samples = len(val_filenames)
        for i in range(num_samples):
        # if len(val_filenames) > 0:
            print("\n" + "-" * 60)
            print(f"Testing __getitem__ for index {i}")
            print("-" * 60)
            try:
                # sample = val_dataset[0]
                sample = val_dataset[i]
                # print("Sample keys: {}".format(list(sample.keys())))
                
                if ("depth_gt", 0, 0) in sample:
                    depth_gt = sample[("depth_gt", 0, 0)]
                    print("GT depth loaded successfully!")
                    print("  Shape: {}".format(depth_gt.shape))
                    print("  Type: {}".format(type(depth_gt)))
                    print("  Dtype: {}".format(depth_gt.dtype))
                    if isinstance(depth_gt, torch.Tensor):
                        print("  Min/Max: {:.3f} / {:.3f}".format(
                            depth_gt.min().item(), depth_gt.max().item()))
                        print("  Mean: {:.3f}".format(depth_gt.mean().item()))
                else:
                    print("WARNING: ('depth_gt', 0, 0) not found in sample!")
                    print("  Available keys: {}".format(list(sample.keys())))

                # check the loaded pose and K
                if ("gt_c2w_poses", 0) in sample:
                    gt_c2w_poses = sample[("gt_c2w_poses", 0)]
                    print("GT pose loaded successfully!")
                    print("  Shape: {}".format(gt_c2w_poses.shape))
                    print("  Type: {}".format(type(gt_c2w_poses)))
                    print("  Dtype: {}".format(gt_c2w_poses.dtype))
                    print("  GT pose: \n{}".format(gt_c2w_poses))
                else:
                    print("WARNING: ('gt_c2w_poses', 0) not found in sample!")

                if ("K_per_frame", 0, 0) in sample:
                    K_per_frame = sample[("K_per_frame",0, 0)]
                    print("GT K loaded successfully!")
                    print("  Shape: {}".format(K_per_frame.shape))
                    print("  Type: {}".format(type(K_per_frame)))
                    print("  Dtype: {}".format(K_per_frame.dtype))
                    print("  GT K: \n{}".format(K_per_frame))

                else:
                    print("WARNING: ('K_per_frame', 0) not found in sample!")
            except Exception as e:
                print("Error loading sample: {}".format(e))
                import traceback
                traceback.print_exc()
            
            # whether loop over all samples
            if not iterate_through_all_files:
                break

        else:
            print("No validation filenames to test!")
            
    except Exception as e:
        print("Error creating dataset: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


