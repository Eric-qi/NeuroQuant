import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Union
from hadamard_transform import hadamard_transform, pad_to_power_of_2

from quantization.quantizer import StraightThrough, UniformAffineQuantizer



def _next_power_of_two(n: int):
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))

def hadamard_along_channel_weight(x: torch.Tensor, normalize: bool = True):
    C_out, C_in, KH, KW = x.shape
    x2 = x.permute(0, 2, 3, 1).contiguous().view(-1, C_in)  # (C_out*KH*KW, C_in)
    y = hadamard_transform(x2)  # this package by default returns normalized transform (see docs)
    # if hadamard_transform is normalized, inverse is the same call
    y = y[:, :C_in].view(C_out, KH, KW, C_in).permute(0, 3, 1, 2).contiguous()
    return y

class QuantModule(nn.Module):
    r"""
        Convert module to quantmodule.
    """

    def __init__(self, org_module: Union[nn.Conv2d,], hadamard: bool=True, weight_quant_params: dict = {}):
        super(QuantModule, self).__init__()
        
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            raise ValueError('Not supported modules: {}'.format(org_module))
        
        
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        self.hadamard = hadamard
        if self.hadamard:
            C_out, C_in, KH, KW = self.weight.shape
            self.C = C_in
            pad_channels = _next_power_of_two(self.C) - self.C
            x_padded = F.pad(org_module.weight.data.clone(), (0, 0, 0, 0, 0, pad_channels))
            self.hadamard_weight = hadamard_along_channel_weight(x_padded)

        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # de-activate the quantized forward default
        self.use_weight_quant = False
        
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.bias_quantizer = UniformAffineQuantizer(**weight_quant_params)

        self.extra_repr = org_module.extra_repr
    
    def forward(self, input: torch.Tensor):
        
        if self.use_weight_quant:
            if self.hadamard:
                weight = hadamard_along_channel_weight(self.weight_quantizer(self.hadamard_weight))[:, :self.C, :, :]
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias_quantizer(self.bias)
            
        else:
            weight = self.org_weight
            bias = self.org_bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out
    
    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

    def get_weight_perturbation(self):
        weight = self.org_weight
        weight_hat = self.weight_quantizer(self.weight)
        return weight - weight_hat



# test similar to above
if __name__ == "__main__":

    print(_next_power_of_two(7))
    x = torch.randn(2, 8, 4, 4)  # prefer C power of two
    y = hadamard_along_channel_weight(x)
    x_rec = hadamard_along_channel_weight(y)  # same call recovers if normalized
    print("Reconstruction error:", (x - x_rec).abs().max().item())