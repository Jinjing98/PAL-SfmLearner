from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import cv2
import numpy as np
import PIL.Image as pil
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    result=(x - mi) / d
    return result

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def disp_to_depth_v2(disp, min_depth, max_depth, is_scaled_disp):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    # always skip scaling in training
    # if disp.requires_grad == True:
        # is_scaled_disp = True

    if is_scaled_disp:
        # already in reasonable range w.r.t min_depth and max_depth
        scaled_disp = disp
    else:
        assert disp.min() >= 0 and disp.max() <= 1, f"disp should be in range [0, 1], got {disp.min()} and {disp.max()}"
        # sigmoid output is in range [0, 1]
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def set_seed(seed):
    """Set random seed for reproducibility
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


def create_dataset_from_file_or_list(file_or_list, splits_dir, dataset_class, data_path, height, width, 
                                     frame_ids, num_input_images, is_train, img_ext='.png', opt=None, mode='train'):
    """Create a dataset from a single file or a list of files by concatenating dataset instances.
    
    This utility function can be used by both trainer_endoda3.py and trainer_endodac.py to handle
    multiple data files with concatenation.
    
    Args:
        file_or_list: Either a string (single file) or a list of strings (multiple files)
        splits_dir: Directory containing the split files
        dataset_class: The dataset class to instantiate (e.g., datasets.SCAREDRAWDataset)
        data_path: Path to the data directory
        height: Image height
        width: Image width
        frame_ids: List of frame IDs to load
        num_input_images: Number of input images (typically 4)
        is_train: Whether this is a training dataset
        img_ext: Image file extension (default: '.png')
        opt: Optional options object for accessing opt.of_samples, opt.of_samples_num, etc.
             If None, overfitting options will be ignored
        mode: 'train', 'val', or 'test' - used for determining dataset parameters
        
    Returns:
        Dataset instance (single dataset or ConcatDataset if multiple files)
    """
    # Normalize to list
    if isinstance(file_or_list, str):
        files = [file_or_list]
    elif file_or_list is None:
        # Handle None case - use default based on mode
        if mode == 'train':
            files = ['train_files.txt']
        elif mode == 'val':
            files = ['val_files.txt']
        else:  # test
            files = ['test_files.txt']
    else:
        files = file_or_list
    
    datasets_list = []
    total_samples = 0
    
    for f in files:
        fpath = os.path.join(splits_dir, f)
        filenames = readlines(fpath)
        
        # Apply overfitting limit if needed
        if opt is not None and getattr(opt, 'of_samples', False):
            of_samples_num = getattr(opt, 'of_samples_num', 100)
            filenames = filenames[:of_samples_num]
        
        # Determine dataset parameters based on file name
        # the flag for val data is critical to affect the data splict where val_err is reported
        # d7k4(cover a lot in test_files.txt) gt pose is missing
        is_test_file = os.path.basename(f) == 'test_files.txt'
        is_sequence_file = os.path.basename(f) in ['test_files_sequence1_val.txt', 'test_files_sequence2_val.txt']
        if mode == 'val':
            load_gt_poses = is_sequence_file
        elif mode == 'train':
             # always load_gt_poses for train so as to enable debug_With_gt_pose_Estimated
            load_gt_poses = True
        else:
            load_gt_poses = False
        load_gt_depth = is_test_file if mode == 'val' else False
        depth_offline_loading = is_test_file  # use gt_depths.npz
        teacher_depth_loading = getattr(opt, 'enable_teacher_student_training', False) and mode == 'train'
        # Create dataset for this file
        # Check if dataset class accepts load_gt_poses, load_gt_depth, depth_offline_loading
        # Some datasets (like in trainer_endodac) may not support all these parameters
        dataset_kwargs = {
            'load_gt_poses': load_gt_poses,
            'load_gt_depth': load_gt_depth,
            'depth_offline_loading': depth_offline_loading,
            'teacher_depth_loading': teacher_depth_loading,
        }
        
        dataset = dataset_class(
            data_path, filenames, height, width,
            frame_ids, num_input_images, is_train=is_train, img_ext=img_ext,
            **dataset_kwargs
        )

        
        datasets_list.append(dataset)
        total_samples += len(dataset)
        print(f"  Loaded {len(dataset)} samples from {f} for {mode} mode")
        print("Load_gt_poses: ", load_gt_poses)
        print("Load_gt_depth: ", load_gt_depth)
        print("depth_offline_loading: ", depth_offline_loading)
    
    # Concatenate if multiple datasets, otherwise return single dataset
    if len(datasets_list) > 1:
        print(f"  Concatenated {len(datasets_list)} datasets: {total_samples} total samples")
        return ConcatDataset(datasets_list)
    else:
        return datasets_list[0]

