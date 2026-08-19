from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from networks.pose_decoder import PoseDecoder


class BackbonePoseDecoder(nn.Module):
    """
    Pose decoder that regresses pose from DINO3 backbone features.
    
    Takes backbone features from frame pairs and processes them through
    an adapter network to create ResNet-like pyramid features, then
    feeds to PoseDecoder.
    
    Args:
        backbone_dim: Dimension of backbone features (e.g., 1536 for ViT-Base with cat_token)
        trans_scale_factor: Scale factor for translation output
        rot_scale_factor: Scale factor for rotation output
        rot_representation: Rotation representation ('angle_axis', '6D', '9D', etc.)
        explicit_bias_init_6d9d: Whether to use explicit bias initialization for 6D/9D
    """
    def __init__(self, backbone_dim=1536, trans_scale_factor=0.001, rot_scale_factor=0.001,
                 rot_representation='angle_axis', explicit_bias_init_6d9d=False):
        super(BackbonePoseDecoder, self).__init__()
        
        self.backbone_dim = backbone_dim
        
        # Create adapter network to convert backbone features to ResNet-like pyramid
        # Input: concatenated features from frame pair [B, 2*backbone_dim, H, W]
        # Output: pyramid features matching ResNet structure [64, 64, 128, 256, 512]
        self.relu = nn.ReLU()
        self.adapter = nn.ModuleList([
            nn.Conv2d(2 * backbone_dim, 64, kernel_size=3, stride=1, padding=1),   # First level
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                # Downsample
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),               # Downsample
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),               # Downsample
        ])
        
        # Create PoseDecoder with ResNet-like num_ch_enc
        num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.pose_decoder = PoseDecoder(
            num_ch_enc=num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
            trans_scale_factor=trans_scale_factor,
            rot_scale_factor=rot_scale_factor,
            rot_representation=rot_representation,
            explicit_bias_init_6d9d=explicit_bias_init_6d9d
        )
    
    def forward(self, backbone_features):
        """
        Forward pass.
        
        Args:
            backbone_features: Backbone features [B, 2, H_patch, W_patch, C] from frame pair
        
        Returns:
            rot_output: Rotation in specified representation [B, 1, 1, rot_dim]
            translation: Translation [B, 1, 1, 3]
        """
        B, S, H, W, C = backbone_features.shape
        assert S == 2, f"Expected 2 frames (pair), got {S}"
        assert C == self.backbone_dim, f"Expected backbone_dim={self.backbone_dim}, got {C}"
        
        # Permute to [B, 2, C, H, W] then concatenate along channel dimension
        x = backbone_features.permute(0, 1, 4, 2, 3)  # [B, 2, C, H, W]
        x = x.reshape(B, S * C, H, W)  # [B, 2*C, H, W]
        
        # Pass through adapter to get encoder-like pyramid features
        features = []
        for i, conv in enumerate(self.adapter):
            x = self.relu(conv(x))
            features.append(x)
        
        # PoseDecoder expects input_features as list of [list of features per scale]
        # Match ResNet structure [64, 64, 128, 256, 512]
        features_for_decoder = [[features[0], features[0], features[1], features[2], features[3]]]
        
        # Forward through PoseDecoder
        rot_output, translation = self.pose_decoder(features_for_decoder)
        
        return rot_output, translation
