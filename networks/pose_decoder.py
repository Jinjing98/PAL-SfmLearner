from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, trans_scale_factor=0.001, rot_scale_factor=0.001, rot_representation="angle_axis", explicit_bias_init_6d9d=True):
        super(PoseDecoder, self).__init__()

        self.trans_scale_factor = trans_scale_factor
        self.rot_scale_factor = rot_scale_factor
        self.rot_representation = rot_representation
        self.explicit_bias_init_6d9d = explicit_bias_init_6d9d

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # adapt to different rotation representations: used for self.convs[("pose", 2)]
        self.trans_vec_dim = 3
        if self.rot_representation == "angle_axis":
            self.rot_vec_dim = 3
        elif self.rot_representation == "9D":
            self.rot_vec_dim = 9
        elif self.rot_representation == "6D":
            self.rot_vec_dim = 6
        elif self.rot_representation == "euler":
            # https://github.dev/ClementPinard/SfmLearner-Pytorch
            # xyz: yaw, pitch, roll
            self.rot_vec_dim = 3
        elif self.rot_representation == "quat":
            # https://github.dev/ClementPinard/SfmLearner-Pytorch
            # rx,ry,rz: rw is computed to have a norm of 1
            self.rot_vec_dim = 3

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, (self.trans_vec_dim + self.rot_vec_dim) * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

        # Apply Special Init on self.convs[("pose", 2)] if 6D/9D and flag is enabled
        if self.explicit_bias_init_6d9d and self.rot_representation in ["6D", "9D"]:
            # 2. 偏置初始化：显式构造 Identity Bias
            if self.rot_representation == "6D":
                # 旋转 (6 dims, Identity: [1,0,0, 0,1,0])+平移 (3 dims, 全0)
                # 假设输出顺序是 [r1, r2, r3, r4, r5, r6, tx, ty, tz, ]
                bias_one_frame = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            elif self.rot_representation == "9D":
                # 旋转 (9 dims, Identity Flattened)+平移 (3 dims, 全0)
                bias_one_frame = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            # 如果一次预测 N 帧，bias 需要重复 N 次
            bias_full = bias_one_frame.repeat(num_frames_to_predict_for)
            # 4. 赋值 (注意要处理 device 问题，确保 bias 和模型在同一个 device)
            with torch.no_grad():
                self.convs[("pose", 2)].bias.copy_(bias_full)

    def forward(self, input_features, ret_intermediate_feat=False):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        intermediate_feature = None
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i == 1 and ret_intermediate_feat:
                intermediate_feature = out
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        if self.rot_representation == "angle_axis":
            # out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)
            out = out.view(-1, self.num_frames_to_predict_for, 1, self.trans_vec_dim + self.rot_vec_dim)

            axisangle = self.rot_scale_factor*out[..., :3] # B num_f 1 3
            translation = self.trans_scale_factor*out[..., 3:]
            if ret_intermediate_feat:
                return axisangle, translation, intermediate_feature
            else:
                return axisangle, translation
        elif self.rot_representation == "euler":
            out = out.view(-1, self.num_frames_to_predict_for, 1, self.trans_vec_dim + self.rot_vec_dim)
            
            # naive mul: scale euler angles by rot_scale_factor
            euler = self.rot_scale_factor*out[..., :self.rot_vec_dim]
            translation = self.trans_scale_factor*out[..., self.rot_vec_dim:]
            if ret_intermediate_feat:
                return euler, translation, intermediate_feature
            else:
                return euler, translation
        elif self.rot_representation == "quat":
            out = out.view(-1, self.num_frames_to_predict_for, 1, self.trans_vec_dim + self.rot_vec_dim)
            
            # naive mul: scale quaternion by rot_scale_factor
            quat = self.rot_scale_factor*out[..., :self.rot_vec_dim]
            translation = self.trans_scale_factor*out[..., self.rot_vec_dim:]
            if ret_intermediate_feat:
                return quat, translation, intermediate_feature
            else:
                return quat, translation
        elif self.rot_representation == "9D":
            out = out.view(-1, self.num_frames_to_predict_for, 1, self.trans_vec_dim + self.rot_vec_dim)
            
            if self.explicit_bias_init_6d9d:
                rot_9d = out[..., :self.rot_vec_dim]
                # scale the r2,r3,r4, r6,r7,r8 by rot_scale_factor
                rot_9d[..., 1:4] *= self.rot_scale_factor
                rot_9d[..., 5:8] *= self.rot_scale_factor
            else:
                # naive mul as in https://github.com/amakadia/svd_for_pose?tab=readme-ov-file
                rot_9d = self.rot_scale_factor*out[..., :self.rot_vec_dim]

            translation = self.trans_scale_factor*out[..., self.rot_vec_dim:]
            if ret_intermediate_feat:
                return rot_9d, translation, intermediate_feature
            else:
                return rot_9d, translation
        elif self.rot_representation == "6D":
            out = out.view(-1, self.num_frames_to_predict_for, 1, self.trans_vec_dim + self.rot_vec_dim)
            if self.explicit_bias_init_6d9d:
                rot_6d = out[..., :self.rot_vec_dim]
                # scale the r2,r3,r4,r6 by rot_scale_factor
                rot_6d[..., 1:4] *= self.rot_scale_factor
                rot_6d[..., 5] *= self.rot_scale_factor
            else:
                # naive mul as in https://github.com/amakadia/svd_for_pose?tab=readme-ov-file
                rot_6d = self.rot_scale_factor*out[..., :self.rot_vec_dim]

            translation = self.trans_scale_factor*out[..., self.rot_vec_dim:]
            if ret_intermediate_feat:
                return rot_6d, translation, intermediate_feature
            else:
                return rot_6d, translation    

        

