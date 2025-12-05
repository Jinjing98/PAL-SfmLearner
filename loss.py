from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_smooth_light(light, img):
    """Computes the smoothness loss for a light image
    """
    grad_light_x = torch.abs(light[:, :, :, :-1] - light[:, :, :, 1:])
    grad_light_y = torch.abs(light[:, :, :-1, :] - light[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    epsilon_x = 0.01 * torch.ones_like(grad_img_x)
    Denominator_x = torch.max(grad_img_x, epsilon_x)
    x_loss = torch.abs(torch.div(grad_light_x, Denominator_x))

    epsilon_y = 0.01 * torch.ones_like(grad_img_y)
    Denominator_y = torch.max(grad_img_y, epsilon_y)
    y_loss = torch.abs(torch.div(grad_light_y, Denominator_y))
    
    return x_loss.mean() + y_loss.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(pred, target, ssim):
    """Computes reprojection loss between predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def compute_losses(inputs, outputs, opt, ssim):
    """Computes all losses for the model
    """
    losses = {}
    total_loss = 0
    loss_reflec = 0
    loss_reprojection = 0
    loss_disp_smooth = 0
    loss_reconstruction = 0

    for frame_id in opt.frame_ids:
        loss_reconstruction += (compute_reprojection_loss(
            inputs[("color_aug", frame_id, 0)], 
            outputs[("reprojection_color", 0, frame_id)],
            ssim
        )).mean()

    for frame_id in opt.frame_ids[1:]: 
        mask = outputs[("valid_mask", 0, frame_id)]
        loss_reflec += (torch.abs(
            outputs[("reflectance", 0, 0)] - outputs[("reflectance_warp", 0, frame_id)]
        ).mean(1, True) * mask).sum() / mask.sum()
        loss_reprojection += (compute_reprojection_loss(
            inputs[("color_aug", 0, 0)], 
            outputs[("reprojection_color_warp", 0, frame_id)],
            ssim
        ) * mask).sum() / mask.sum()
            
    disp = outputs[("disp", 0)]
    color = inputs[("color_aug", 0, 0)]
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    loss_disp_smooth = get_smooth_loss(norm_disp, color)
 
    total_loss = (opt.reprojection_constraint * loss_reprojection / 2.0 + 
                  opt.reflec_constraint * (loss_reflec / 2.0) + 
                  opt.disparity_smoothness * loss_disp_smooth + 
                  opt.reconstruction_constraint * (loss_reconstruction / 3.0))

    losses["loss"] = total_loss
    return losses

