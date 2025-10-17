import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

from ._layers import PositionEncoding, OutImg, NeRVBlock


class NeRV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # BUILD Encoder LAYERS
        crop_h, crop_w = cfg['crop_h'], cfg['crop_w'],
        self.fc_h = crop_h  // np.prod(cfg['dec_strides'])
        self.fc_w = crop_w  // np.prod(cfg['dec_strides'])
        
        base, level = cfg['base'], cfg['level']
        self.encoder = PositionEncoding(base, level)  
        
        # BUILD Decoder LAYERS
        dec_layers = []
        in_channel = cfg['dec_in_channel']
        dec_layer1 = nn.Conv2d(int(level*2), in_channel * self.fc_h * self.fc_w, 1, 1, 0)
        dec_layers.append(dec_layer1)
        
        for ks, stride in zip(cfg['dec_kernels'], cfg['dec_strides']):
            out_channel = int(max(round(in_channel / cfg['channel_reduce']), cfg['channel_lbound']))
            nerv_block = NeRVBlock(in_channel, out_channel, ks, stride, bias=True, 
                                   norm=cfg['dec_norm'], act=cfg['dec_acts'])
            dec_layers.append(nerv_block)
            in_channel = out_channel
        
        self.decoder = nn.ModuleList(dec_layers)
        self.head_layer = nn.Conv2d(in_channel, 3, 3, 1, 1) 
        self.out_bias = cfg['out_bias']
    
    def encode(self, img):
        img_embed = self.encoder(img[:, None]).float()
        return img_embed
    
    def decode(self, img_embed):
        embed_list = [img_embed]
        
        dec_start = time.time()
        
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)
            
        output = self.head_layer(output)
        img_out = OutImg(output, self.out_bias)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time
        
    def forward(self, input):
        img_embed = self.encode(input)
        
        img_out, embed_list, dec_time = self.decode(img_embed)
        
        return img_out, embed_list, dec_time