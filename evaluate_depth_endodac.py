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
from utils.util import readlines, disp_to_depth
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


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
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

        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, opt.test_data_file))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False,
                                            load_gt_poses=False)
        elif opt.eval_split == 'hamlyn':
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
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
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
            dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

    if opt.eval_split == 'endovis':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        
    if opt.visualize_depth:
        vis_dir = os.path.join(opt.load_weights_folder, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)

    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []
    
    errors = []
    ratios = []
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
                    pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disp = pred_disp[0]
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
                # print('pred_depth shape:')
                # print(pred_depth.shape, pred_depth.min(), pred_depth.max())
                pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
                # print('resized pred_depth shape:')
                # print(pred_depth.shape, pred_depth.min(), pred_depth.max())
            elif opt.model_type == 'endodac' or opt.model_type == 'afsfm':
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1/pred_disp
            else:
                raise ValueError(f"Invalid model type: {opt.model_type}")
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            
            # if opt.visualize_depth:
            #     vis_pred_depth = render_depth(pred_disp)
            #     vis_file_name = os.path.join(vis_dir, sequence + "_" +  keyframe + "_" + frame_id + ".png")
            #     cv2.imwrite(vis_file_name, vis_pred_depth)
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            
            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            # error = compute_errors(gt_depth, pred_depth)
            error = compute_depth_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)

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
