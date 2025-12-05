from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2
import torch

from .mono_dataset import MonoDataset

DEFAULT_D7K4_SCENE_POINTS_DIR='/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Depth/dataset_7/keyframe_4/data/scene_points'
DATA_PATH='/mnt/cluster/datasets/SCARED/'

class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

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
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

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
        # Special override for dataset7/keyframe4
        if dataset_num == 7 and keyframe_num == 4:
            override_candidate = os.path.join(DEFAULT_D7K4_SCENE_POINTS_DIR, f_str)
            if os.path.exists(override_candidate):
                depth_path = override_candidate

        # Default search under main data_path if not found via override
        if not (dataset_num == 7 and keyframe_num == 4) and depth_path is None:
            for subset in ["training", "testing"]:
                for ddir in dataset_dir_candidates:
                    for kdir in keyframe_dir_candidates:
                        candidate = os.path.join(
                            DATA_PATH,
                            subset,
                            ddir,
                            kdir,
                            "data",
                            "scene_points",
                            f_str
                        )
                        if os.path.exists(candidate):
                            depth_path = candidate
                            break
                    if depth_path is not None:
                        break
                if depth_path is not None:
                    break

        if depth_path is None:
            print("Warning: missing depth for {} frame {} (line {}).".format(folder, frame_index))
            assert False, 'depth_path is None for folder: {} frame: {}'.format(folder, frame_index)
        assert os.path.exists(depth_path), f"Depth file {depth_path} does not exist."
        depth_gt = cv2.imread(depth_path, 3)
        if depth_gt is None:
            print('Depth file is broken/None in path {} for folder: {} frame: {}'.format(depth_path, folder, frame_index))
            print('We set broken depth to zeros...')
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
        
        return inputs


if __name__ == "__main__":
    # Test GT depth loading for validation dataset
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from utils import readlines
    
    # Configuration for testing
    data_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/"
    height = 256
    width = 320
    frame_ids = [0, -1, 1]
    split = "endovis"
    
    # Read validation filenames
    splits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")
    val_fpath = os.path.join(splits_dir, split, "val_files.txt")
    val_fpath = os.path.join(splits_dir, split, "test_files.txt")
    val_fpath = os.path.join(splits_dir, split, "train_files.txt")
    val_fpath = os.path.join(splits_dir, split, "d6_kf2.txt")
    
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
            frame_ids, 4, is_train=False, img_ext='.png'
        )
        print("Dataset created successfully!")
        print("GT depths loaded: {}".format(val_dataset.gt_depths_val is not None))
        
        if val_dataset.gt_depths_val is not None:
            print("GT depths shape: {}".format(val_dataset.gt_depths_val.shape))
            print("Number of GT depth maps: {}".format(len(val_dataset.gt_depths_val)))
            print("GT depth dtype: {}".format(val_dataset.gt_depths_val.dtype))
            print("GT depth min/max: {:.3f} / {:.3f}".format(
                val_dataset.gt_depths_val.min(), val_dataset.gt_depths_val.max()))
        else:
            print("WARNING: GT depths not loaded! Check if gt_depths_val.npz exists.")
        
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
                
            except Exception as e:
                print("Error loading sample: {}".format(e))
                import traceback
                traceback.print_exc()
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


