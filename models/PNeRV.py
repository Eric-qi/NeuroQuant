import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

from ._layers import ConvNeXt, OutImg, NeRVBlock, LayerNorm, Block, KFc_bias, ActivationLayer


class PNeRV1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ## BUILD Encoder LAYERS
        self.c1_dim = cfg['enc_channel']
        self.d_dim = cfg['emd_channel']
        
        # Content encoder 
        self.enc_layers = nn.ModuleList()
        for k, stride in enumerate(cfg['enc_strides']):
            c0 = 3 if k == 0 else self.c1_dim
            self.enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['enc_strides'][k], stride=cfg['enc_strides'][k]))
            self.enc_layers.append(LayerNorm(self.c1_dim, eps=1e-6, data_format="channels_first"))
            self.enc_layers.append(Block(dim=self.c1_dim))
        self.enc_layers.append(nn.Conv2d(self.c1_dim, self.d_dim, kernel_size=1, stride=1))
        
        
        # BUILD Decoder LAYERS
        ngf = int(self.d_dim)
        ngf__ = ngf
        
        new_ngf = int(cfg['kfc_h_w_c'][2])
        new_h , new_w = cfg['kfc_h_w_c'][0], cfg['kfc_h_w_c'][1]

        crop_h, crop_w = cfg['crop_h'], cfg['crop_w'],
        in_h, in_w = [crop_h, crop_w] // np.prod(cfg['enc_strides'])
        
        self.dec_exc_layers = NeRVBlock(self.d_dim, self.d_dim, kernel_size=1, stride=10, bias=True, 
                                        norm=cfg['dec_norm'], act=cfg['dec_acts'])
        
        self.dec_layers = nn.ModuleList()
        self.dec_shortcuts = nn.ModuleList()
        self.dec_bsm_z = nn.ModuleList()
        self.dec_bsm_r = nn.ModuleList()
        self.dec_bsm_h = nn.ModuleList()
        for i, stride in enumerate(cfg['kfc_strides']):
            new_h , new_w = new_h*stride, new_w*stride
            if i != len(cfg['kfc_strides'])-1:
                self.dec_shortcuts.append(KFc_bias(in_height=in_h, in_width=in_w, out_height=new_h, out_width=new_w, channels=ngf__))
                self.dec_shortcuts.append(nn.BatchNorm2d(ngf__, track_running_stats=False))
                self.dec_shortcuts.append(ActivationLayer(cfg['dec_acts']))
                
                self.dec_bsm_z.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))
                self.dec_bsm_r.append(nn.Conv2d(ngf__, new_ngf, kernel_size=3, stride=1, padding=1))
                self.dec_bsm_h.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))
                
            self.dec_layers.append(NeRVBlock(ngf, new_ngf, kernel_size=3, stride=stride, bias=True, 
                                    norm=cfg['dec_norm'], act=cfg['dec_acts']))
            ngf = new_ngf
        self.dec_head_layers = nn.Conv2d(new_ngf, 3, 3, 1, 1)
    
    def encode(self, content_gt):
        content_embedding = content_gt
        
        for convnext_layer in self.enc_layers:
            content_embedding = convnext_layer(content_embedding) 
            
        return content_embedding
    
    def decode(self, content):
        embed_list = [content]
        
        dec_start = time.time()
        output = self.dec_exc_layers(content)
        
        for ii in range(4):
            pym = self.dec_shortcuts[3*ii+0](content)
            pym = self.dec_shortcuts[3*ii+1](pym)
            pym = self.dec_shortcuts[3*ii+2](pym)
            output = self.dec_layers[ii](output)
            
            memory_z = self.dec_bsm_z[ii](output)
            memory_r = self.dec_bsm_r[ii](pym)
            memory = torch.relu(memory_z +memory_r )
            att = torch.sigmoid(self.dec_bsm_h[ii](memory))
            output = att*output + (1- att)* memory_r
        
        output = self.dec_layers[ii+1](output)
        output = self.dec_head_layers(output)
        img_out = torch.sigmoid(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time
        

    def forward(self, content_gt):
        output = self.encode(content_gt)
        
        img_out, embed_list, dec_time = self.decode(output)
        
        return img_out, embed_list, dec_time




class PNeRV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ## BUILD Encoder LAYERS
        self.c1_dim = cfg['enc_channel']
        self.d_dim = cfg['emd_channel']
        
        # Content encoder 
        self.enc_layers = nn.ModuleList()
        for k, stride in enumerate(cfg['enc_strides']):
            c0 = 3 if k == 0 else self.c1_dim
            self.enc_layers.append(nn.Conv2d(c0, self.c1_dim, kernel_size=cfg['enc_strides'][k], stride=cfg['enc_strides'][k]))
            self.enc_layers.append(LayerNorm(self.c1_dim, eps=1e-6, data_format="channels_first"))
            self.enc_layers.append(Block(dim=self.c1_dim))
        self.enc_layers.append(nn.Conv2d(self.c1_dim, self.d_dim, kernel_size=1, stride=1))
        
        # Diff encoder 
        self.enc_diff_layers = nn.ModuleList()
        
        
        # BUILD Decoder LAYERS
        ngf = int(self.d_dim)
        new_ngf = int(cfg['kfc_h_w_c'][2])
        new_h , new_w = cfg['kfc_h_w_c'][0], cfg['kfc_h_w_c'][1]
        
        self.dec_exc_layers = NeRVBlock(self.d_dim, self.d_dim, kernel_size=1, stride=10, bias=True, 
                                        norm=cfg['dec_norm'], act=cfg['dec_acts'])
        
        self.dec_layers = nn.ModuleList()
        self.dec_bsm_z = nn.ModuleList()
        self.dec_bsm_h = nn.ModuleList()
        for i, stride in enumerate(cfg['kfc_strides']):
            new_h , new_w = new_h*stride, new_w*stride
            if i != len(cfg['kfc_strides'])-1:
                self.dec_bsm_z.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))
                self.dec_bsm_h.append(nn.Conv2d(new_ngf, new_ngf, kernel_size=3, stride=1, padding=1))
                
            self.dec_layers.append(NeRVBlock(ngf, new_ngf, kernel_size=3, stride=stride, bias=True, 
                                    norm=cfg['dec_norm'], act=cfg['dec_acts']))
            ngf = new_ngf
        self.dec_head_layers = nn.Conv2d(new_ngf, 3, 3, 1, 1)
    
    def encode(self, content_gt):
        content_embedding = content_gt
        for convnext_layer in self.enc_layers:
            content_embedding = convnext_layer(content_embedding) 
            
        return content_embedding
    
    def decode(self, content):
        embed_list = [content]
        
        dec_start = time.time()
        output = self.dec_exc_layers(content)
        
        for ii in range(4):
            output = self.dec_layers[ii](output)
            memory_z = self.dec_bsm_z[ii](output)
            memory = torch.relu(memory_z)
            att = torch.sigmoid(self.dec_bsm_h[ii](memory))
            output = att*output
        
        output = self.dec_layers[ii+1](output)
        output = self.dec_head_layers(output)
        img_out = torch.sigmoid(output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time
        

    def forward(self, content_gt):
        output = self.encode(content_gt)
        
        img_out, embed_list, dec_time = self.decode(output)
        
        return img_out, embed_list, dec_time