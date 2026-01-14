"""
Ray-based Pose and Intrinsics Decoders

These modules regress pose and camera intrinsics from ray embeddings
output by the auxiliary head of the depth model.

Strategy: Reuse existing PoseDecoder and IntrinsicsHead by adding 
adapter layers to convert ray embeddings to compatible feature format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class RayPoseDecoder(nn.Module):
    """
    Predicts relative camera pose from ray embeddings.
    Uses existing PoseDecoder with an adapter to convert ray embeddings to features.
    
    Args:
        ray_dim: Dimension of ray embeddings (default: 6 for DualDPT aux head)
        trans_scale_factor: Scale factor for translation output
        rot_scale_factor: Scale factor for rotation output
        rot_representation: Rotation representation ('angle_axis', '6D', '9D', 'quat')
        explicit_bias_init_6d9d: Whether to use explicit bias initialization for 6D/9D
    """
    
    def __init__(self, ray_dim=6, trans_scale_factor=0.001, rot_scale_factor=0.001,
                 rot_representation='angle_axis', explicit_bias_init_6d9d=False):
        super(RayPoseDecoder, self).__init__()
        
        self.ray_dim = ray_dim
        
        # Adapter: Convert ray embeddings to encoder-like features
        # Input: [B, 2*7, H, W] → Output: encoder-like features
        self.adapter = nn.ModuleList([
            nn.Conv2d(ray_dim * 2, 64, kernel_size=7, stride=2, padding=3),   # [B, 64, H/2, W/2]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),            # [B, 128, H/4, W/4]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),           # [B, 256, H/8, W/8]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),           # [B, 512, H/16, W/16]
        ])
        self.relu = nn.ReLU()
        
        # Reuse existing PoseDecoder from networks (supports trans_scale_factor, etc.)
        # Simulate encoder output: num_ch_enc = [64, 64, 128, 256, 512] (ResNet-like)
        from networks.pose_decoder import PoseDecoder
        self.num_ch_enc = [64, 64, 128, 256, 512]  # Expose for intrinsics_head compatibility
        self.pose_decoder = PoseDecoder(
            num_ch_enc=self.num_ch_enc,
            num_input_features=1,  # Single input feature (concatenated pair)
            num_frames_to_predict_for=1,  # Predict one relative pose
            trans_scale_factor=trans_scale_factor,
            rot_scale_factor=rot_scale_factor,
            rot_representation=rot_representation,
            explicit_bias_init_6d9d=explicit_bias_init_6d9d
        )
    
    def forward(self, ray_embeddings):
        """
        Forward pass.
        
        Args:
            ray_embeddings: Ray embeddings [B, 2, C, H, W] from frame pair
        
        Returns:
            rot_output: Rotation in specified representation [B, 1, 1, rot_dim]
            translation: Translation [B, 1, 1, 3]
        """
        B, S, C, H, W = ray_embeddings.shape
        assert S == 2, f"Expected 2 frames (pair), got {S}"
        assert C == self.ray_dim, f"Expected ray_dim={self.ray_dim}, got {C}"
        
        # Concatenate ray embeddings from both frames along channel dimension
        x = ray_embeddings.view(B, S * C, H, W)  # [B, 14, H, W]
        
        # Pass through adapter to get encoder-like features
        features = []
        for i, conv in enumerate(self.adapter):
            x = self.relu(conv(x))
            features.append(x)
        
        # PoseDecoder expects input_features as list of [list of features per scale]
        # For single input, wrap as [[feat1, feat2, feat3, feat4, feat5]]
        # We need to add a dummy first layer to match ResNet structure [64, 64, ...]
        features_for_decoder = [[features[0], features[0], features[1], features[2], features[3]]]
        
        # Forward through PoseDecoder
        rot_output, translation = self.pose_decoder(features_for_decoder)
        
        return rot_output, translation


class RayIntrinsicsHead(nn.Module):
    """
    Predicts camera intrinsics from ray embeddings.
    Uses existing IntrinsicsHead with an adapter to convert ray embeddings to bottleneck features.
    
    Args:
        ray_dim: Dimension of ray embeddings (default: 6 for DualDPT aux head)
    """
    
    def __init__(self, ray_dim=6):
        super(RayIntrinsicsHead, self).__init__()
        
        self.ray_dim = ray_dim
        
        # Adapter: Convert ray embeddings to bottleneck features
        # Input: [B, 7, H, W] → Output: [B, 256, H', W'] (bottleneck-like feature)
        self.adapter = nn.Sequential(
            nn.Conv2d(ray_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Reuse existing IntrinsicsHead
        # IntrinsicsHead.forward() directly uses global pooling on input
        # It expects input to already be 256 channels (adapter output is perfect)
        from third_party.EndoDAC.models.decoders import IntrinsicsHead
        # num_ch_enc is only used to initialize convs_squeeze (which is unused in forward)
        # We pass 256 as the last channel to match adapter output
        num_ch_enc = [64, 64, 128, 256, 256]
        self.intrinsics_head = IntrinsicsHead(num_ch_enc)
    
    def forward(self, ray_embedding, width, height):
        """
        Forward pass.
        
        Args:
            ray_embedding: Ray embedding from reference frame [B, C, H, W]
            width: Target image width
            height: Target image height
        
        Returns:
            K: Camera intrinsics matrix [B, 4, 4] (to match IntrinsicsHead output format)
        """
        B, C, H_in, W_in = ray_embedding.shape
        assert C == self.ray_dim, f"Expected ray_dim={self.ray_dim}, got {C}"
        
        # Pass through adapter to get bottleneck-like features
        bottleneck = self.adapter(ray_embedding)  # [B, 256, H/8, W/8]
        
        # Forward through IntrinsicsHead
        # IntrinsicsHead returns [B, 4, 4]
        K = self.intrinsics_head(bottleneck, width, height)
        
        return K
