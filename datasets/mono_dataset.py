from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F
import time
import errno
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_robust(path, retries=10, delay=0.1):
    """
    Robust image loader with retry logic for network file systems (NFS/ZFS).
    Catches BlockingIOError and generic OSErrors related to temporary unavailability.
    """
    for attempt in range(retries):
        try:
            # open path as file to avoid ResourceWarning
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')
                    
        except (BlockingIOError, OSError) as e:
            # Check if the error is essentially "Resource temporarily unavailable"
            if isinstance(e, BlockingIOError) or e.errno == errno.EAGAIN:
                # If we have attempts left, wait and retry
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
            
            # If it's a different error (e.g. FileNotFoundError) or we are out of retries, crash
            raise


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # Handle both old and new Pillow versions
        try:
            self.interp = Image.Resampling.LANCZOS
        except AttributeError:
            self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        # self.loader = pil_loader
        self.loader = pil_loader_robust
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, do_color_aug, do_trans_aug=False):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        
        if do_color_aug:
             fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

        # Per-frame translation fractions (generated on first encounter, reused for all scales)
        frame_translations = {}

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # apply trans_aug on 'color' and 'color_aug'
                if do_trans_aug:
                    # Generate translation fraction for this frame (if not already generated)
                    if im not in frame_translations:
                        frame_translations[im] = (
                            random.uniform(-0.03, 0.03),  # tx_fraction
                            random.uniform(-0.03, 0.03)   # ty_fraction
                        )
                    
                    # Scale translation linearly with image size at current scale
                    tx_fraction, ty_fraction = frame_translations[im]
                    scale = 2 ** i
                    tx = tx_fraction * (self.width // scale)
                    ty = ty_fraction * (self.height // scale)
                    
                    f = F.affine(f, angle=0, translate=(tx, ty), scale=1.0, shear=0.0,
                                interpolation=transforms.InterpolationMode.BILINEAR, fill=0)

                inputs[(n, im, i)] = self.to_tensor(f)
                if do_color_aug:
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            f = F.adjust_brightness(f, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            f = F.adjust_contrast(f, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            f = F.adjust_saturation(f, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            f = F.adjust_hue(f, hue_factor)
                # apply trans_aug on 'color_aug'
                
                inputs[(n + "_aug", im, i)] = self.to_tensor(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            ("depth_gt", 0, 0)                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        # extend tran only
        do_trans_aug = self.is_train and random.random() > 0.5
        do_trans_aug = False
        # do_trans_aug = True

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):

            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        self.preprocess(inputs, do_color_aug, do_trans_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            # Only add depth_gt if get_depth returned a valid depth (not None)
            # None means it will be loaded elsewhere (e.g., from npz in child class)
            if depth_gt is not None:
                inputs[("depth_gt", 0, 0)] = np.expand_dims(depth_gt, 0)
                inputs[("depth_gt", 0, 0)] = torch.from_numpy(inputs[("depth_gt", 0, 0)].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
