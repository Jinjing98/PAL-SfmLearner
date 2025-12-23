import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from networks.layers import ConvBlock, Conv3x3

def upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class adjust_net(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1, enable_multiscale=False, scales=range(4)):
        super(adjust_net, self).__init__()
        self.input_channel = num_input_channels
        self.output_channel = num_output_channels
        self.enable_multiscale = enable_multiscale
        self.scales = scales
        
        self.convs = OrderedDict()
        
        if self.enable_multiscale:
            # Multi-scale architecture similar to TransformDecoder
            # Create feature pyramid at different scales
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])
            
            # Encoder-like layers that process input at different scales
            for i in range(5):
                if i == 0:
                    num_ch_in = self.input_channel
                else:
                    num_ch_in = self.num_ch_dec[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("conv", i)] = ConvBlock(num_ch_in, num_ch_out)
            
            # Decoder-like layers with upsampling
            for i in range(4, -1, -1):
                # upconv_0
                num_ch_in = self.num_ch_dec[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
                
                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if i > 0:
                    num_ch_in += self.num_ch_dec[i - 1]  # Skip connection
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
            # Output layers for each scale
            for s in self.scales:
                self.convs[("transform_conv", s)] = Conv3x3(self.num_ch_dec[s], self.output_channel)
        else:
            # Original single-scale architecture
            self.convs[("conv", 1)] = ConvBlock(self.input_channel, 32)
            self.convs[("conv", 2)] = ConvBlock(32, 32)
            self.convs[("conv", 3)] = ConvBlock(32, 32)
            self.convs[("conv", 4)] = nn.Conv2d(32, self.output_channel, kernel_size=1)
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.Tanh = nn.Tanh()

    def forward(self, input_features):
        if self.enable_multiscale:
            # Multi-scale forward pass similar to TransformDecoder
            self.outputs = {}
            
            # Encoder: process input through conv layers and downsample
            features = []
            x = input_features
            for i in range(5):
                x = self.convs[("conv", i)](x)
                features.append(x)
                if i < 4:  # Downsample except for last layer
                    x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            
            # Decoder: upsample and combine with skip connections
            x = features[-1]  # Start from the deepest feature
            for i in range(4, -1, -1):
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                if i > 0:
                    x += [features[i - 1]]  # Skip connection
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
                if i in self.scales:
                    self.outputs[("transform", i)] = self.Tanh(self.convs[("transform_conv", i)](x))
            
            return self.outputs
        else:
            # Original single-scale forward pass
            adjust_L = self.convs[("conv", 1)](input_features)
            adjust_L = self.convs[("conv", 2)](adjust_L)
            adjust_L = self.convs[("conv", 3)](adjust_L)
            adjust_L = self.convs[("conv", 4)](adjust_L)
            outputs = self.Tanh(adjust_L)
            
            return outputs

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = adjust_net().cuda()
    model.eval()

    tgt_img = torch.randn(4, 1, 256, 320).cuda()
    output=model(tgt_img)
    print(output.shape())
