import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi, ceil
from timm.models.layers import trunc_normal_, DropPath



def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


###################################  NeRV Block   ###################################
class NeRVBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 
                      out_channel * stride * stride, 
                      kernel_size, 
                      stride=1, 
                      padding=ceil((kernel_size - 1) // 2), 
                      bias=bias),
            nn.PixelShuffle(stride) if stride !=1 else nn.Identity(),
        )
        self.norm = NormLayer(norm, out_channel)
        self.act = ActivationLayer(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


###################################  PNeRV KFc_bias   ###################################
class KFc_bias(nn.Module):
    def __init__(self, in_batch=1, in_height=2, in_width=4, out_height=4, out_width=8, channels=4):
        super().__init__()
        self.in_b = in_batch
        self.in_h = in_height
        self.in_w = in_width
        self.c = channels
        self.out_h = out_height
        self.out_w = out_width

        self.w_L_ = torch.normal(0, 1/self.in_h,  (self.c, self.out_h, self.in_h))
        self.w_R_ = torch.normal(0, 1/self.out_w, (self.c, self.in_w, self.out_w))
        nn.init.kaiming_normal_(self.w_L_, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.w_R_, mode='fan_out', nonlinearity='relu')

        self.w_L = nn.Parameter(self.w_L_.repeat(self.in_b,1,1,1))
        self.w_R = nn.Parameter(self.w_R_.repeat(self.in_b,1,1,1))

        self.b_h = nn.Parameter(torch.zeros(self.out_h, 1))
        self.b_w = nn.Parameter(torch.zeros(1, self.out_w))
        self.b_c = nn.Parameter(torch.zeros(self.c, 1))
        
    def forward(self, x):
        b_ = self.b_h @ self.b_w
        b__ = b_.reshape(1, self.out_h*self.out_w)
        _ = self.b_c @ b__
        __ = _.reshape(self.c, self.out_h, self.out_w)
        b = __.repeat(self.in_b, 1, 1, 1)

        _ = torch.matmul(self.w_L, x)
        return torch.matmul(_, self.w_R) + b

    def extra_repr(self) -> str:
        return 'w_L={}, w_R={}, b_h={}, b_w={}, b_c={}'.format(self.w_L.shape, self.w_R.shape, self.b_h.shape, self.b_w.shape, self.b_c.shape)

    
###################################  Basic layers like position encoding/ normalization layers/ activation layers   ###################################
class PositionEncoding(nn.Module):
    def __init__(self, base, level):
        super(PositionEncoding, self).__init__()
        self.pe_bases = base ** torch.arange(int(level)) * pi

    def forward(self, pos):
        value_list = pos * self.pe_bases.to(pos.device)
        pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
        return pe_embed.view(pos.size(0), -1, 1, 1)
    
class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)
    

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(num_features=ch_width, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer
    

###################################  Code for ConvNeXt   ###################################
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x