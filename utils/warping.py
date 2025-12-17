from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    axisangle: B 1 3
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def transformation_from_parameters_6D(rot_6d, translation, invert=False):
    """Convert the network's (rot_6d, translation) output into a 4x4 matrix
    rot_6d: B 1 6
    translation: B 1 3
    """
    # Convert 6D representation to a proper rotation matrix
    R_3x3 = rot_from_6d(rot_6d).reshape(-1, 3, 3)
    # Lift to homogeneous coordinates
    R = torch.eye(4, device=rot_6d.device, dtype=rot_6d.dtype).unsqueeze(0).repeat(R_3x3.shape[0], 1, 1)
    R[:, :3, :3] = R_3x3

    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def transformation_from_parameters_9D(rot_9d, translation, invert=False):
    """Convert the network's (rot_9d, translation) output into a 4x4 matrix
    rot_9d: B 1 9
    translation: B 1 3
    """
    # Convert 9D representation to a proper rotation matrix
    R_3x3 = rot_from_9d(rot_9d)
    # Lift to homogeneous coordinates
    R = torch.eye(4, device=rot_9d.device, dtype=rot_9d.dtype).unsqueeze(0).repeat(R_3x3.shape[0], 1, 1)
    R[:, :3, :3] = R_3x3

    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def transformation_from_parameters_quat(quat, translation, invert=False):
    """Convert the network's (quat, translation) output into a 4x4 matrix
    quat: B 1 3 (or B 3 after indexing) - [x, y, z] format, w is computed from norm
    translation: B 1 3 (or B 3 after indexing)
    """
    # Handle shape: quat can be [B, 1, 3] or [B, 3]
    if quat.dim() == 3:
        quat = quat.squeeze(1)  # [B, 1, 3] -> [B, 3]
    
    # rot_from_quat expects [B, 3] as [x, y, z] and computes w based on norm
    # Use existing rot_from_quat function which computes w to have norm of 1
    R_3x3 = rot_from_quat(quat)
    
    # Lift to homogeneous coordinates
    B = quat.size(0)
    R = torch.eye(4, device=quat.device, dtype=quat.dtype).unsqueeze(0).repeat(B, 1, 1)
    R[:, :3, :3] = R_3x3
    
    t = translation.clone()
    if translation.dim() == 3:
        t = t.squeeze(1)  # [B, 1, 3] -> [B, 3]
    
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    
    T = get_translation_matrix(t)
    
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    
    return M

def transformation_from_parameters_euler(euler, translation, invert=False):
    """Convert the network's (euler, translation) output into a 4x4 matrix
    euler: B 1 3 (or B 3 after indexing) - yaw, pitch, roll
    translation: B 1 3 (or B 3 after indexing)
    """
    # Handle shape: euler can be [B, 1, 3] or [B, 3]
    if euler.dim() == 3:
        euler = euler.squeeze(1)  # [B, 1, 3] -> [B, 3]
    
    # Convert euler angles to rotation matrix
    R_3x3 = rot_from_euler(euler)
    
    # Lift to homogeneous coordinates
    B = euler.size(0)
    R = torch.eye(4, device=euler.device, dtype=euler.dtype).unsqueeze(0).repeat(B, 1, 1)
    R[:, :3, :3] = R_3x3
    
    t = translation.clone()
    if translation.dim() == 3:
        t = t.squeeze(1)  # [B, 1, 3] -> [B, 3]
    
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    
    T = get_translation_matrix(t)
    
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    
    return M

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_quat_xyzw(quat_xyzw):
    """Convert quaternion coefficients to rotation matrix.
    
    Args:
        quat_xyzw: Quaternion in XYZW format (scalar-last) -- size = [B, 4] or (..., 4)
                   Format: [x, y, z, w] where w is the scalar part
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3] or (..., 3, 3)
    """
    # Reorder from XYZW [x, y, z, w] to WXYZ [w, x, y, z] format
    # Extract components and reorder
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
    quat_wxyz = torch.stack([w, x, y, z], dim=-1)
    
    # Reuse rot_from_quat_wxyz which handles WXYZ format
    return rot_from_quat_wxyz(quat_wxyz)


def rot_from_quat_wxyz(quat_wxyz):
    """Convert quaternion coefficients to rotation matrix.
    
    Args:
        quat_wxyz: Quaternion in WXYZ format (scalar-first) -- size = [B, 4] or (..., 4)
                   Format: [w, x, y, z] where w is the scalar part
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3] or (..., 3, 3)
    """
    # Normalize the quaternion
    norm_quat = quat_wxyz / quat_wxyz.norm(p=2, dim=-1, keepdim=True)
    
    # Extract components: [w, x, y, z]
    w, x, y, z = norm_quat[..., 0], norm_quat[..., 1], norm_quat[..., 2], norm_quat[..., 3]
    
    # Get batch dimensions (handle arbitrary shape)
    batch_shape = quat_wxyz.shape[:-1]
    
    # Compute squares and products
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    # Build rotation matrix using standard quaternion to rotation matrix formula
    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
        2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
        2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2
    ], dim=-1).reshape(*batch_shape, 3, 3)
    
    return rotMat


def rot_from_quat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def rot_from_euler(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        yaw,pitch,roll
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def rot_from_9d(m):
    """Convert 9D representation to SO(3) using SVD orthogonalization.

    Args:
        m: [BATCH, 1, 9] 9D rotation representation.

    Returns:
        [BATCH, 1, 3, 3] SO(3) rotation matrices.
    """
    assert m.dim() == 3
    assert m.shape[2] == 9 and m.shape[1] == 1
    m = m.reshape((-1, 3, 3))
    # if m.dim() < 3:
        # m = m.reshape((-1, 3, 3))
    m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
    u, s, v = torch.svd(m_transpose)
    det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
    # Check orientation reflection.
    r = torch.matmul(
        torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
        u.transpose(-2, -1)
    )
    return r

def rot_from_6d(d6):  # code from pytorch3d
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    assert  d6.dim() == 3
    assert d6.shape[2] == 6 and d6.shape[1] == 1

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def pose_encoding_to_extri_intri_v2(
    pose_encoding,
    image_size_hw=None,
    rot_representation="quat_xyzw",
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.
    
    Args:
        pose_encoding: Tensor of shape (..., N) containing [T(3), rotation(M), fov_h(1), fov_w(1)]
                      Expected dimensions:
                      - "angle_axis": N=8 (3+3+1+1)
                      - "euler": N=8 (3+3+1+1)
                      - "quat": N=8 (3+3+1+1)
                      - "quat_xyzw": N=9 (3+4+1+1)
                      - "quat_wxyz": N=9 (3+4+1+1)
                      - "6D": N=11 (3+6+1+1)
                      - "9D": N=14 (3+9+1+1)
        image_size_hw: Tuple of (height, width) for intrinsics computation
        rot_representation: Rotation representation type
    
    Returns:
        extrinsics: Camera extrinsics tensor of shape (..., 3, 4)
        intrinsics: Camera intrinsics tensor of shape (..., 3, 3)
    """
    # Rotation dimension mapping
    rot_dims = {
        "angle_axis": 3, "euler": 3, "quat": 3,
        "quat_xyzw": 4, "quat_wxyz": 4,
        "6D": 6, "9D": 9
    }
    
    if rot_representation not in rot_dims:
        raise ValueError(
            f"Unsupported rotation representation: {rot_representation}. "
            f"Supported: {', '.join(rot_dims.keys())}"
        )
    
    # Sanity check: expected total dimension
    expected_dim = 3 + rot_dims[rot_representation] + 2  # T(3) + rot(M) + fov(2)
    actual_dim = pose_encoding.shape[-1]
    assert actual_dim == expected_dim, \
        f"pose_encoding dimension mismatch: expected {expected_dim} for {rot_representation}, got {actual_dim}"
    
    # Extract components (fov_h, fov_w are always last two)
    T = pose_encoding[..., :3]
    rot = pose_encoding[..., 3:3+rot_dims[rot_representation]]
    fov_h = pose_encoding[..., -2]
    fov_w = pose_encoding[..., -1]
    
    # Convert rotation to matrix
    if rot_representation == "angle_axis":
        original_shape = rot.shape[:-1]
        R_4x4 = rot_from_axisangle(rot.reshape(-1, 1, 3))
        R = R_4x4[:, :3, :3].reshape(*original_shape, 3, 3)
    elif rot_representation == "euler":
        R = rot_from_euler(rot)
    elif rot_representation == "quat":
        R = rot_from_quat(rot)
    elif rot_representation == "quat_xyzw":
        R = rot_from_quat_xyzw(rot)
    elif rot_representation == "quat_wxyz":
        R = rot_from_quat_wxyz(rot)
    elif rot_representation == "6D":
        R = rot_from_6d(rot)
    elif rot_representation == "9D":
        original_shape = rot.shape[:-1]
        R_flat = rot_from_9d(rot.reshape(-1, 1, 9))
        R = R_flat.reshape(*original_shape, 3, 3)

    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return extrinsics, intrinsics


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords


class Project3D_Raw(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_Raw, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        raw_pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        raw_pix_coords = raw_pix_coords.view(self.batch_size, 2, self.height, self.width)
        raw_pix_coords = raw_pix_coords.permute(0, 2, 3, 1)

        return raw_pix_coords


class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiantiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the source image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border", align_corners=True)

