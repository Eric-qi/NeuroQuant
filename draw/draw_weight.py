import os
import sys
import time
import random
import shutil
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.utils.data


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

sys.path.append('/sjq/NeuroQuant/')
from models import NeRV, HNeRV, PNeRV
from videosets import VideoDataSet
from utils import get_config


from torch.autograd.variable import Variable



def parse_args(argv):
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # initial
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--arch', type=str, help='the architecture of NeRV')

    # eval
    parser.add_argument('--weight', default='None', type=str, help='model for test')

    args = parser.parse_args(argv)
    return args


def draw_histogram(v, id, be_save=False):
    
    font_path = '/sjq/NeuroQuant/draw/TIMES.TTF'
    font_prop = fm.FontProperties(fname=font_path, size=14, )
    
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(6, 3), dpi=300)
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.6, zorder=0)
    plt.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3, zorder=0) 
    plt.minorticks_on()
    
    sns.distplot(v.cpu().detach().numpy(), bins = 320, kde = False, hist_kws = {'color':'steelblue', 'zorder':2}, label = 'histogram', norm_hist=True)
    sns.distplot(v.cpu().detach().numpy(), hist = False, kde_kws = {'color':'blue', 'linestyle':'-', 'zorder':2},
             norm_hist = True, label = ('density curve'))
    
    plt.xticks(fontproperties=font_prop,  size=14)
    plt.yticks(fontproperties=font_prop, size=14)
    plt.xlabel("value", fontproperties=font_prop)
    plt.ylabel("frequency", fontproperties=font_prop)
    #plt.legend(loc=4, prop=font1)
    if be_save:
        plt.savefig(f"draw/visual_weight/layer_weight{id}.pdf", dpi=600, format="pdf", bbox_inches = 'tight')
    plt.show()

def draw_list_histogram(v, id, be_save=False):
    
    font_path = 'draw/TIMES.TTF'
    font_prop = fm.FontProperties(fname=font_path, size=24, )
    
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(3, 6), dpi=300)
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.6, zorder=0) 
    plt.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3, zorder=0) 
    plt.minorticks_on()
    
    color_list = ['steelblue', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    kde_color = ['blue', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, weight in enumerate(v):
        color_hist = color_list[i % len(color_list)]
        color_kde = kde_color[i % len(kde_color)]
        
        sns.distplot(weight.cpu().detach().numpy(), bins=320, kde=False, 
                        hist_kws={'color': color_hist, 'zorder':2}, label=f'histogram {i}', norm_hist=True)
        sns.distplot(weight.cpu().detach().numpy(),  hist=False, 
                        kde_kws={'color': color_kde, 'linestyle':'-', 'zorder':2}, norm_hist=True, 
                        label=f'density curve {i}')
    
    # plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    plt.xticks(fontproperties=font_prop,  size=24)
    plt.yticks(fontproperties=font_prop, size=24)
    plt.xlabel("value", fontproperties=font_prop)
    plt.ylabel("frequency", fontproperties=font_prop)
    #plt.legend(loc=4, prop=font1)
    if be_save:
        plt.savefig(f"draw/visual_weight/layer_weight{id}.pdf", dpi=600, format="pdf", bbox_inches = 'tight')
    plt.show()
    

def draw_channel_minmax(v, id):
    font_path = 'draw/TIMES.TTF'
    font_prop = fm.FontProperties(fname=font_path, size=24, )
    
    #plt.legend(loc='lower right', prop=font1)
    fig = plt.figure(figsize=(3, 6), dpi=600)
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.6, zorder=0)
    plt.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3, zorder=0) 
    plt.minorticks_on()
    
    # plt.rcParams['font.sans-serif'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    
    
    plt.tight_layout()

    plt.xlabel("Channel's index", fontproperties=font_prop)
    plt.ylabel("Min/Max Value", fontproperties=font_prop)
    
    for i in range(v.shape[0]):
        w_max = v[i].max().item()
        w_min = v[i].min().item()
        
        plt.bar(i,w_max, color='purple', zorder=2)
        plt.bar(i,w_min, color='indianred', zorder=2)
        
    plt.xticks(fontproperties=font_prop, size=24)
    plt.yticks(fontproperties=font_prop, size=24)
    
    plt.savefig(f"draw/visual_weight/channel_weight{id}.pdf", dpi=600, format="pdf", bbox_inches = 'tight')


def draw(args, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build model
    if args.arch == 'hnerv':
        model = HNeRV(cfg)
    elif args.arch == 'pnerv':
        model = PNeRV(cfg)
    elif args.arch == 'nerv':
        model = NeRV(cfg)
    else:
        return ValueError('model arch wrong!')
    model.to(device)
    
    # load pretrained model for evaluation or resume
    if args.weight != 'None':
        logging.info("=> loading checkpoint '{}'".format(args.weight))
        checkpoint = torch.load(args.weight, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
    

    for k, v in model.decoder.named_parameters():
        if 'weight' in k:
            draw_channel_minmax(v, k)
    
    draw_list = []
    count = 0
    for k, v in model.decoder.named_parameters():
        if 'weight' in k:
            count += 1
            if count == 1 or count ==3:
                draw_list.append(v)
                # draw_histogram(v, k, True)
            
        
    
    draw_list_histogram(draw_list, 'all', True)
    

def main(argv):
    torch.set_printoptions(precision=4) 
    
    args = parse_args(argv)
    cfg = get_config(args.config)
    
    torch.set_printoptions(precision=2) 
    draw(args, cfg)
    

if __name__ == '__main__':
    main(sys.argv[1:])
