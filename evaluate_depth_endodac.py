from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import scipy.stats as st

# from utils.layers import disp_to_depth
# from utils.utils import readlines, compute_errors
# from options import MonodepthOptions
from options_endodac import MonodepthOptions 
import datasets

import third_party.EndoDAC.models.encoders as encoders
import third_party.EndoDAC.models.decoders as decoders
import third_party.EndoDAC.models.endodac as endodac
from utils.util import readlines, disp_to_depth, disp_to_depth_v2
from utils.metrics import compute_depth_errors

import sys
from pathlib import Path

import torchvision

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "third_party/depth_anything_3/src"))
from depth_anything_3.api import DepthAnything3

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def track_gt_excluded_pixels(gt_depth, mask, metadata_stats):
    """Track GT pixels excluded from computation.
    
    Args:
        gt_depth: Full GT depth map
        mask: Boolean mask of valid pixels (True = included in computation)
        metadata_stats: Dictionary to store statistics
    
    Returns:
        None (updates metadata_stats in place)
    """
    total_pixels = gt_depth.size
    excluded_mask = ~mask
    num_excluded = np.sum(excluded_mask)
    metadata_stats['gt_excluded_pixels_count'].append(num_excluded)
    metadata_stats['gt_excluded_pixels_percentage'].append(100.0 * num_excluded / total_pixels if total_pixels > 0 else 0.0)
    if num_excluded > 0:
        excluded_gt_values = gt_depth[excluded_mask]
        metadata_stats['gt_excluded_values'].extend(excluded_gt_values.tolist())


def track_pred_clipped_pixels(pred_depth, num_valid_pixels, metadata_stats, min_depth=1e-3, max_depth=150):
    """Track predicted pixels clipped to MIN_DEPTH or MAX_DEPTH.
    
    Args:
        pred_depth: Predicted depth values (before clipping)
        num_valid_pixels: Number of valid pixels in the prediction
        metadata_stats: Dictionary to store statistics
        min_depth: Minimum depth threshold (default: 1e-3)
        max_depth: Maximum depth threshold (default: 150)
    
    Returns:
        None (updates metadata_stats in place)
    """
    num_clipped_to_min = np.sum(pred_depth < min_depth)
    num_clipped_to_max = np.sum(pred_depth > max_depth)
    metadata_stats['pred_clipped_to_min_count'].append(num_clipped_to_min)
    metadata_stats['pred_clipped_to_max_count'].append(num_clipped_to_max)
    metadata_stats['pred_clipped_to_min_percentage'].append(100.0 * num_clipped_to_min / num_valid_pixels if num_valid_pixels > 0 else 0.0)
    metadata_stats['pred_clipped_to_max_percentage'].append(100.0 * num_clipped_to_max / num_valid_pixels if num_valid_pixels > 0 else 0.0)


def construct_gt_depth_filename(eval_split, filenames, frame_idx, ext_disp_to_eval=None):
    """Construct filename based on GT depth naming convention.
    
    Args:
        eval_split: Evaluation split name ('endovis', 'hamlyn', 'c3vd')
        filenames: List of filenames from test file
        frame_idx: Current frame index in the loop
        ext_disp_to_eval: Optional external disparity file path
    
    Returns:
        str: Constructed filename (without extension)
    """
    if eval_split == 'endovis':
        # Parse filename to get folder and frame_index
        if ext_disp_to_eval is None and frame_idx < len(filenames):
            filename_line = filenames[frame_idx].strip()
            parts = filename_line.split()
            folder = parts[0] if len(parts) > 0 else ""
            frame_index = int(parts[1]) if len(parts) > 1 else 0
            # Extract dataset and keyframe from folder (format: "dataset3/keyframe4")
            if '/' in folder:
                dataset_part, keyframe_part = folder.split('/')
                dataset_num = dataset_part.replace('dataset', '') if 'dataset' in dataset_part else ''
                keyframe_num = keyframe_part.replace('keyframe', '') if 'keyframe' in keyframe_part else ''
                # Format: dataset{num}_keyframe{num}_scene_points{frame_index:06d}
                filename = f"dataset{dataset_num}_keyframe{keyframe_num}_scene_points{frame_index - 1:06d}"
            else:
                # Fallback if folder format is unexpected
                filename = f"scene_points{frame_index - 1:06d}"
        else:
            # Fallback if filename not available or using ext_disp_to_eval
            filename = f"scene_points{frame_idx:06d}"
    else:
        # For other splits, use index-based naming
        filename = f"depth_{frame_idx:06d}"
    
    return filename


def print_metadata_stats(num_frames, metadata_stats):
    """Print metadata statistics.
    
    Args:
        num_frames: Total number of frames processed
        metadata_stats: Dictionary containing all metadata statistics
    """
    print("\n" + "="*60)
    print("EVALUATION METADATA(not checked)")
    print("="*60)
    print(f"Number of frames in computation: {num_frames}")
    
    # GT pixels not involved in computation
    if metadata_stats['gt_excluded_pixels_count']:
        avg_excluded_count = np.mean(metadata_stats['gt_excluded_pixels_count'])
        avg_excluded_percentage = np.mean(metadata_stats['gt_excluded_pixels_percentage'])
        total_excluded = np.sum(metadata_stats['gt_excluded_pixels_count'])
        print(f"\nGT pixels excluded from computation:")
        print(f"  Total excluded pixels: {total_excluded}")
        print(f"  Average excluded per frame: {avg_excluded_count:.1f} pixels ({avg_excluded_percentage:.2f}%)")
        if metadata_stats['gt_excluded_values']:
            excluded_values_arr = np.array(metadata_stats['gt_excluded_values'])
            print(f"  Excluded GT depth stats: min={excluded_values_arr.min():.4f}, max={excluded_values_arr.max():.4f}, mean={excluded_values_arr.mean():.4f}")
    
    # Estimated pixels clipped
    if metadata_stats['pred_clipped_to_min_count']:
        total_clipped_to_min = np.sum(metadata_stats['pred_clipped_to_min_count'])
        total_clipped_to_max = np.sum(metadata_stats['pred_clipped_to_max_count'])
        avg_clipped_to_min_pct = np.mean(metadata_stats['pred_clipped_to_min_percentage'])
        avg_clipped_to_max_pct = np.mean(metadata_stats['pred_clipped_to_max_percentage'])
        print(f"\nEstimated pixels clipped:")
        print(f"  Clipped to MIN_DEPTH: {total_clipped_to_min} pixels (avg {avg_clipped_to_min_pct:.2f}% per frame)")
        print(f"  Clipped to MAX_DEPTH: {total_clipped_to_max} pixels (avg {avg_clipped_to_max_pct:.2f}% per frame)")
    
    print("="*60 + "\n")


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        if opt.save_pred_disps and not opt.save_pred_disps_online:
            pred_disps = []
            pred_depths = []
        if opt.model_type in ['endodac', 'afsfm']:
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)

            print("-> Loading weights from {}".format(opt.load_weights_folder))
        elif opt.model_type == 'depthanything3':
            print("Evaluating Depth Anything 3 model")
        else:
            assert False, f"Invalid model type: {opt.model_type}"

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)
        elif opt.model_type == 'afsfm':
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path)

        assert len(opt.test_data_file) == 1, f"Only support one test data file for now, but got {len(opt.test_data_file)}"
        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, opt.test_data_file[0]))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False,
                                            load_gt_poses=False)
        elif opt.eval_split == 'hamlyn':
            filenames = []  # Not used for hamlyn, but initialize for consistency
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
            filenames = []  # Not used for c3vd, but initialize for consistency
            dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.model_type == 'endodac':
            depther = endodac.endodac(
                backbone_size = "base", r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=opt.dino_resize_hw, pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token)
            model_dict = depther.state_dict()
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'depthanything3':
            # Load model from Hugging Face Hub
            depther = DepthAnything3.from_pretrained("depth-anything/da3-base")
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'afsfm':
            encoder = encoders.ResnetEncoder(opt.num_layers, False)
            depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))
            depther = lambda image: depth_decoder(encoder(image))
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()
    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, opt.test_data_file))
            # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False)
        elif opt.eval_split == 'hamlyn':
            filenames = []  # Not used for hamlyn, but initialize for consistency
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
            filenames = []  # Not used for c3vd, but initialize for consistency
            dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

    if opt.eval_split == 'endovis':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        
    # Determine save folder: use saved_folder if provided, otherwise use load_weights_folder
    save_folder = opt.saved_folder if opt.saved_folder is not None else opt.load_weights_folder
    
    if opt.visualize_depth:
        vis_dir = os.path.join(save_folder, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Setup prediction saving directories (online mode)
    if opt.save_pred_disps_online:
        if opt.model_type == 'endodac' or opt.model_type == 'afsfm':
            pred_disps_dir = os.path.join(save_folder, "pred_disps_online")
            os.makedirs(pred_disps_dir, exist_ok=True)
        elif opt.model_type == 'depthanything3':
            pred_depths_dir = os.path.join(save_folder, "pred_depths_online")
            os.makedirs(pred_depths_dir, exist_ok=True)

    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []
    
    errors = []
    ratios = []
    # Metadata tracking (only if enabled)
    num_frames = 0
    if opt.compute_metadata_stats:
        metadata_stats = {
            'gt_excluded_pixels_count': [],
            'gt_excluded_pixels_percentage': [],
            'gt_excluded_values': [],
            'pred_clipped_to_min_count': [],
            'pred_clipped_to_max_count': [],
            'pred_clipped_to_min_percentage': [],
            'pred_clipped_to_max_percentage': []
        }
    else:
        metadata_stats = None
    
    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[("color", 0, 0)].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if opt.ext_disp_to_eval is None:
                if opt.model_type == 'endodac' or opt.model_type == 'afsfm':
                    time_start = time.time()
                    output = depther(input_color)
                    inference_time = time.time() - time_start
                    output_disp = output[("disp", 0)]
                    pred_disp, _ = disp_to_depth_v2(output_disp, opt.min_depth, opt.max_depth, is_scaled_disp=False)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disp = pred_disp[0]
                    if opt.save_pred_disps_online:
                        # Save immediately (online mode)
                        pred_filename_base = construct_gt_depth_filename(opt.eval_split, filenames, i, opt.ext_disp_to_eval)
                        pred_filename = f"{pred_filename_base}.npy"
                        pred_path = os.path.join(pred_disps_dir, pred_filename)
                        np.save(pred_path, pred_disp)
                    elif opt.save_pred_disps:
                        # Cache for later (cached mode)
                        pred_disps.append(pred_disp)
                elif opt.model_type == 'depthanything3':
                    # convert torch tensor image to pil format
                    input_color = data[("color", 0, 0)].cuda()
                    # convert torch tensor image to pil format as inference requires
                    input_color_pil = torchvision.transforms.ToPILImage()(input_color.squeeze(0))
                    images = [input_color_pil, input_color_pil]  # List of image paths, PIL Images, or numpy arrays
                    time_start = time.time()
                    prediction = depther.inference(
                        images,
                        process_res=280, # 256,320 -> 224,280
                        process_res_method="upper_bound_resize"
                        # export_dir="output",
                        # export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
                    )# already in numpy array format
                    inference_time = time.time() - time_start
                    pred_depth = prediction.depth.squeeze()[0].squeeze()# only get the 1st frame considering both frames equal
                    pred_conf = prediction.conf.squeeze()[0].squeeze()
                    if opt.save_pred_disps_online:
                        # Save immediately (online mode)
                        pred_filename_base = construct_gt_depth_filename(opt.eval_split, filenames, i, opt.ext_disp_to_eval)
                        pred_filename = f"{pred_filename_base}.npy"
                        pred_path = os.path.join(pred_depths_dir, pred_filename)
                        np.save(pred_path, pred_depth)
                    elif opt.save_pred_disps:
                        # Cache for later (cached mode)
                        pred_depths.append(pred_depth)
                else:
                    raise ValueError(f"Invalid model type: {opt.model_type}")
            else:
                pred_disp = pred_disps[i]
                inference_time = 1
            inference_times.append(inference_time)
            
            if opt.eval_split == 'endovis':
                gt_depth = gt_depths[i]
                # sequence = str(np.array(data['sequence'][0]))
                # keyframe = str(np.array(data['keyframe'][0]))
                # frame_id = "{:06d}".format(data['frame_id'][0])
            elif opt.eval_split == 'hamlyn' or opt.eval_split == 'c3vd':
                gt_depth = data["depth_gt"].squeeze().numpy()

            gt_height, gt_width = gt_depth.shape[:2]
            if opt.model_type == 'depthanything3':
                assert pred_depth.shape == (224, 280), f"pred_depth shape: {pred_depth.shape}"
                pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
            elif opt.model_type == 'endodac' or opt.model_type == 'afsfm':
                if opt.model_type == 'endodac':
                    assert pred_disp.shape == (256, 320), f"pred_disp shape: {pred_disp.shape} != (256, 320), it should be already resized as 256,320 before put in outputs"
                elif opt.model_type == 'afsfm':
                    assert pred_disp.shape == (256, 320), f"pred_disp shape: {pred_disp.shape} != (256, 320), it should be already resized as 256,320 before put in our model"
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1/pred_disp # raw depth from these two models roughly below 0.5 and median in 0.25
            else:
                raise ValueError(f"Invalid model type: {opt.model_type}")
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            
            # Track GT pixels not involved in computation (only if metadata stats enabled)
            if opt.compute_metadata_stats:
                track_gt_excluded_pixels(gt_depth, mask, metadata_stats)
            
            if opt.visualize_depth:
                # vis_pred_depth = render_depth(pred_disp)

                # convert vis_pred_depth from depth directly
                # use MIN_DEPTH and MAX_DEPTH to clamp the pred_depth, and self renormalizaiton
                vis_pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
                vis_pred_depth = (vis_pred_depth - vis_pred_depth.min()) / (vis_pred_depth.max() - vis_pred_depth.min()) * 255.0
                vis_pred_depth = vis_pred_depth.astype(np.uint8)

                # Construct filename based on GT depth naming convention
                vis_filename_base = construct_gt_depth_filename(opt.eval_split, filenames, i, opt.ext_disp_to_eval)
                vis_filename = f"{vis_filename_base}.png"
                vis_file_name = os.path.join(vis_dir, vis_filename)
                cv2.imwrite(vis_file_name, vis_pred_depth)
                print(f"-> Saving visualized depth to {vis_file_name}")

                # also save the confidence map as gray image
                vis_conf_filename = f"{vis_filename_base}_conf.png"
                vis_conf_file_name = os.path.join(vis_dir, vis_conf_filename)
                # print("pred_conf min max: ", pred_conf.min(), pred_conf.max())
                # expp1 activation
                pred_conf_gray = (pred_conf - pred_conf.min()) / (pred_conf.max() - pred_conf.min()) * 255.0
                pred_conf_gray = pred_conf_gray.astype(np.uint8)
                cv2.imwrite(vis_conf_file_name, pred_conf_gray)
                # cv2.imwrite(vis_conf_file_name, pred_conf*255.0,)
                # print(f"-> Saving visualized confidence map to {vis_conf_file_name}")                
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            
            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio
            
            # Track predicted pixels clipped to MIN_DEPTH or MAX_DEPTH (only if metadata stats enabled)
            if opt.compute_metadata_stats:
                num_valid_pixels = pred_depth.size
                track_pred_clipped_pixels(pred_depth, num_valid_pixels, metadata_stats, MIN_DEPTH, MAX_DEPTH)
            
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            num_frames += 1
            

            # error = compute_errors(gt_depth, pred_depth)
            error = compute_depth_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)


    # Save cached predictions (only if not using online mode)
    if opt.save_pred_disps and not opt.save_pred_disps_online:
        if pred_disps != []:
            assert pred_depths == [], "pred_depths should be empty if pred_disps is not empty"
            output_path = os.path.join(
                save_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)
        else:
            assert pred_depths != [], "pred_depths should not be empty if pred_disps is not empty"
            output_path = os.path.join(
                save_folder, "depths_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted depths to ", output_path)
            np.save(output_path, pred_depths)

    # Print metadata (only if enabled)
    if opt.compute_metadata_stats:
        print_metadata_stats(num_frames, metadata_stats)
    
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    # cls = []
    # for i in range(len(mean_errors)):
    #     cl = st.t.interval(alpha=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
    #     cls.append(cl[0])
    #     cls.append(cl[1])
    # cls = np.array(cls)
    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    # print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
