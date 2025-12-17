# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
import torch.nn as nn


class EndoCameraDec(nn.Module):
    def __init__(self, dim_in=1536, rot_representation="quat_xyzw", 
                 rot_scale_factor=1.0, trans_scale_factor=1.0,
                 explicit_bias_init_6d9d=True):
        super().__init__()
        # Rotation dimension mapping
        rot_dims = {
            "angle_axis": 3, "euler": 3, 
            "quat": 3, # estimate xyz, then obtain w based on the norm
            "quat_xyzw": 4, #default in da3 
            "quat_wxyz": 4,
            "6D": 6, "9D": 9
        }
        
        if rot_representation not in rot_dims:
            raise ValueError(
                f"Unsupported rotation representation: {rot_representation}. "
                f"Supported: {', '.join(rot_dims.keys())}"
            )
        
        self.rot_representation = rot_representation
        self.rot_scale_factor = rot_scale_factor
        self.trans_scale_factor = trans_scale_factor
        self.explicit_bias_init_6d9d = explicit_bias_init_6d9d
        rot_dim = rot_dims[rot_representation]
        
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, rot_dim)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())
        
        # Apply Special Init on self.fc_qvec if 6D/9D and flag is enabled
        if self.explicit_bias_init_6d9d and self.rot_representation in ["6D", "9D"]:
            # Initialize bias with Identity rotation matrix
            if self.rot_representation == "6D":
                # 6D representation: Identity matrix [1,0,0, 0,1,0] flattened
                bias_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            elif self.rot_representation == "9D":
                # 9D representation: Identity matrix [1,0,0, 0,1,0, 0,0,1] flattened
                bias_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            # Initialize bias (note: device will be set when model is moved to device)
            with torch.no_grad():
                self.fc_qvec.bias.copy_(bias_rot)

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        
        # Apply scale factors to translation and rotation
        out_t = self.trans_scale_factor * self.fc_t(feat.float()).reshape(B, N, 3)
        
        if camera_encoding is None:
            # estimate camera intrinsics and extrinsics from the feature.
            out_rot_raw = self.fc_qvec(feat.float()).reshape(B, N, -1)
            
            # Apply special scaling for 6D/9D with explicit_bias_init_6d9d
            if self.explicit_bias_init_6d9d and self.rot_representation == "6D":
                out_rot = out_rot_raw.clone()
                # scale the r2,r3,r4,r6 by rot_scale_factor
                out_rot[..., 1:4] *= self.rot_scale_factor
                out_rot[..., 5] *= self.rot_scale_factor
            elif self.explicit_bias_init_6d9d and self.rot_representation == "9D":
                out_rot = out_rot_raw.clone()
                # scale the r2,r3,r4, r6,r7,r8 by rot_scale_factor
                out_rot[..., 1:4] *= self.rot_scale_factor
                out_rot[..., 5:8] *= self.rot_scale_factor
            if self.rot_representation in ["quat_xyzw"]:
                # only scale the xyz: we can do this by assuming small theta for the rotation.
                out_rot = out_rot_raw.clone()
                out_rot[..., :3] *= self.rot_scale_factor
            elif self.rot_representation in ["quat_wxyz"]:
                # only scale the xyz: we can do this by assuming small theta for the rotation.
                out_rot = out_rot_raw.clone()
                out_rot[..., 1:4] *= self.rot_scale_factor
            elif self.rot_representation in ["quat", "angle_axis", "euler"]:
                # we can safely scaling for these representation.
                out_rot = self.rot_scale_factor * out_rot_raw
            elif self.rot_representation in ["9D"]:
                if self.explicit_bias_init_6d9d:
                    out_rot = out_rot_raw.clone()
                    # scale the r2,r3,r4, r6,r7,r8 by rot_scale_factor
                    out_rot[..., 1:4] *= self.rot_scale_factor
                    out_rot[..., 5:8] *= self.rot_scale_factor
                else:
                    out_rot = self.rot_scale_factor * out_rot_raw
            elif self.rot_representation in ["6D"]:
                if self.explicit_bias_init_6d9d:
                    out_rot = out_rot_raw.clone()
                    # scale the r2,r3,r4,r6 by rot_scale_factor
                    out_rot[..., 1:4] *= self.rot_scale_factor
                    out_rot[..., 5] *= self.rot_scale_factor
                else:
                    out_rot = self.rot_scale_factor * out_rot_raw
            else:
                assert False, "Unsupported rotation representation"
            
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            # Extract rotation and fov from camera_encoding
            # Format: [T(3), rotation(M), fov_h(1), fov_w(1)]
            # Get rotation dimension from the model's expected dimension
            rot_dim = self.fc_qvec.out_features
            out_rot = camera_encoding[..., 3:3+rot_dim]
            out_fov = camera_encoding[..., -2:]
        
        pose_enc = torch.cat([out_t, out_rot, out_fov], dim=-1)
        return pose_enc

