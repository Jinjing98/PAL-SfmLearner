from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import flow_to_image
import numpy as np

# Flow visualization functions for tensorboard logging
def flow_vis(flow_tensor):
    """Convert flow tensor to image for tensorboard logging (original version)"""
    img = flow_to_image(flow_tensor)    # returns (3,H,W) uint8
    img = img.float() / 255.0
    return img

def flow_vis_robust(flow_tensor):
    """Convert flow tensor to image for tensorboard logging (RAFT-style robust version)"""
    # Robust flow normalization using quantiles to handle extreme values
    # Similar to RAFT visualization approach
    
    # Convert to numpy for quantile computation
    flow_np = flow_tensor.detach().cpu().numpy()  # (2, H, W)
    flow_np = flow_np.transpose(1, 2, 0)  # (H, W, 2)
    
    # Use absolute values for quantile computation since flow can be positive/negative
    # Get 95th percentile of absolute values as upper bound
    upper_bound = np.quantile(np.abs(flow_np), 0.95)
    
    # Clamp flow to reasonable range [-upper_bound, +upper_bound]
    flow_clamped = np.clip(flow_np, 
                         a_min=-upper_bound, 
                         a_max=upper_bound)

    #clamp to avoid absurd values
    threshold = 1000 #px
    flow_clamped = np.clip(flow_clamped, a_min=-threshold, a_max=threshold)
    
    # Normalize to [-1, 1] range for flow_to_image
    if upper_bound > 1e-6:  # Avoid division by zero
        flow_normalized = flow_clamped / upper_bound
    else:
        flow_normalized = flow_clamped
    
    # Convert to image using torchvision
    flow_tensor_normalized = torch.from_numpy(flow_normalized.transpose(2, 0, 1)).float()  # (2, H, W)
    img = flow_to_image(flow_tensor_normalized)  # returns (3, H, W) uint8
    img = img.float() / 255.0
    
    return img

def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


class optical_flow(nn.Module):

    def __init__(self, size, batch_size, height, width, eps=1e-7):
        super(optical_flow, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        optical_flow =  pix_coords[:, [1,0], ...] - self.grid

        return optical_flow


class get_occu_mask_backward(nn.Module):

    def __init__(self, size):
        super(get_occu_mask_backward, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, flow, th=0.95):
        
        new_locs = self.grid + flow
        new_locs = new_locs[:, [1,0], ...]
        corr_map = get_corresponding_map(new_locs)
        occu_map = corr_map
        occu_mask = (occu_map > th).float()

        return occu_mask, occu_map


class get_occu_mask_bidirection(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super(get_occu_mask_bidirection, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, flow12, flow21, scale=0.01, bias=0.5):
        
        new_locs = self.grid + flow12
        shape = flow12.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        flow21_warped = F.grid_sample(flow21, new_locs, mode=self.mode, padding_mode="border")
        flow12_diff = torch.abs(flow12 + flow21_warped)
        # mag = (flow12 * flow12).sum(1, keepdim=True) + \
        # (flow21_warped * flow21_warped).sum(1, keepdim=True)
        # occ_thresh = scale * mag + bias
        # occ_mask = (flow12_diff * flow12_diff).sum(1, keepdim=True) < occ_thresh
        
        return flow12_diff

