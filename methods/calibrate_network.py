import os
import sys
import time
import random
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Subset
from torchvision.utils import save_image


sys.path.append('/sjq/NeuroQuant/')
from quantization import *
from models import NeRV, HNeRV
from videosets import VideoDataSet
from utils import data_split, RoundTensor, psnr_fn_batch, msssim_fn_batch, \
    worker_init_fn, adjust_lr, loss_fn, psnr_fn_single, get_config, setup_logger


def parse_args(argv):
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # initial
    parser.add_argument('--seed', default=903, type=int, help='random seed for results reproduction')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--arch', type=str, help='the architecture of NeRV')
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    
    # dataset
    parser.add_argument('--data_path', type=str, help='data path for vid')
    parser.add_argument('--vid', type=str, help='video id',)
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    
    # quantization parameters
    parser.add_argument('--batch_size', default=12, type=int, help='mini-batch size for data loader')
    parser.add_argument('--precision', type=int, nargs='+', default=[8, 8, 8, 8, 8, 8, 8], help='layer-wise precision')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--hadamard', action='store_true', help='apply hadamard transform for weights')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--input_prob', default=1.0, type=float)
    parser.add_argument('--lr', default=0.0015, type=float)
    parser.add_argument('--norm_p', default=2.0, type=float, help='the norm of L-p')
    parser.add_argument('--init', default='max', type=str, help='param init type', 
                        choices=['max','mse', 'gaussian', 'l1', 'l2', ])
    parser.add_argument('--opt_mode', default='mse', type=str, help='optimization mode type', 
                        choices=['mse','fisher_diag', 'fisher_full', 'lp_norm', ])
    
    
    # eval
    parser.add_argument('--ckpt', default='None', type=str, help='model for test')
    parser.add_argument('--dump_vis', action='store_true', default=False, help='dump the prediction images')
    
    args = parser.parse_args(argv)
    return args

def seed_all(seed=903):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False  # for reproducation
    torch.backends.cudnn.deterministic = True
    


@torch.no_grad()
def evaluate(model, full_dataloader, args, cfg, dump_vis=False):
    img_embed_list = []
    metric_list = [[] for _ in range(len(args.metric_names))]
    dec_time_list = []
    
    model.eval()
    device = next(model.parameters()).device
    
    if dump_vis:
        visual_dir = f'{args.outf}/visualize_calib_network'
        logging.info(f'Saving predictions to {visual_dir}...')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)
    
    for i, sample in enumerate(full_dataloader):
        # forward
        img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
        if args.arch == 'hnerv':
            img_out, embed_list, dec_time = model.decode(model.encode(img_data))
        elif args.arch == 'nerv':
            img_out, embed_list, dec_time = model.decode(model.encode(norm_idx))
        img_embed_list.append(embed_list[0])
        
        # collect decoding fps
        dec_time_list.append(dec_time)
                
        # compute psnr and ms-ssim
        pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_data), msssim_fn_batch([img_out], img_data)
        for metric_idx, cur_v in enumerate([pred_psnr, pred_ssim]):
            for batch_i, cur_img_idx in enumerate(img_idx):
                metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                metric_list[metric_idx_start+metric_idx].append(cur_v[:,batch_i])
        
        # dump predictions
        if dump_vis:
            for batch_ind, cur_img_idx in enumerate(img_idx):
                full_ind = i * cfg['batch_size'] + batch_ind
                dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')
                
        # print eval results and add to log txt
        if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
            avg_time = sum(dec_time_list) / len(dec_time_list)
            fps = cfg['batch_size'] / avg_time
            for v_name, v_list in zip(args.metric_names, metric_list):
                if 'pred_seen_psnr' in v_name:
                    psnr = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                    psnr = RoundTensor(psnr, 2)
                elif 'pred_seen_ssim' in v_name:
                    ssim = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                    ssim = RoundTensor(ssim, 4)
            logging.info('[{}], Eval at Step [{}/{}], FPS {}, PSNR {}, MS-SSIM {}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), i+1, len(full_dataloader), round(fps, 1), psnr, ssim))
            
    # Collect results     
    results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
    args.fps = fps
    h,w = img_data.shape[-2:]
    model.train()
    
    return results_list, (h,w), img_embed_list


def calibrate(args, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim']
    
    # setup dataloader
    full_dataset = VideoDataSet(cfg, args)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['workers'], pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, False, 0)
    
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=cfg['workers'], pin_memory=True, sampler=None, drop_last=True, 
            worker_init_fn=worker_init_fn, persistent_workers=True)
    
    # build model
    if args.arch == 'hnerv':
        model = HNeRV(cfg)
        # get parameters and flops
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        embed_param = float(cfg['enc_channel'][-1]) / np.prod(cfg['enc_strides'])**2 * args.final_size * args.full_data_length
        total_param = decoder_param + embed_param / 1e6
    elif args.arch == 'nerv':
        model = NeRV(cfg)
        # get parameters and flops
        encoder_param = 0.
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        embed_param = 0.
        total_param = decoder_param + embed_param / 1e6
    else:
        return ValueError('model arch wrong!')
    model.to(device)
    
    args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
    args.outf = os.path.join(args.outf, f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M')
    args.outf = os.path.join(args.outf, "network-wise_calib/hadamard-{}_{}-init_batch{}_CW_weight{}_brange{}-{}_warmup{}_lr{}".format(args.hadamard, args.init, args.batch_size, args.weight, args.b_start, args.b_end, args.warmup, args.lr))
    
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)
    
    setup_logger(args.outf + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    logging.info('[PID] %s'%os.getpid())
    logging.info('================== Model Architecture=================')
    logging.info(str(model))
    logging.info(f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M')
    
    # load pretrained model for evaluation or resume
    assert args.ckpt != 'None'
    logging.info("=> loading checkpoint '{}'".format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    
    
    
    logging.info('=======================Full-precision model========================')
    print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.ckpt)
    results_list, _, embedding_list = evaluate(model, full_dataloader, args, cfg, args.dump_vis)
    for i, (metric_name, metric_value) in enumerate(zip(args.metric_names, results_list)):
        cur_v = RoundTensor(metric_value.max(), 2 if 'psnr' in metric_name else 4)
        print_str += f'best_{metric_name}: {cur_v} | '
    logging.info(print_str) 
    
    
    # build quantization parameters
    wq_params = {'n_bits': 8, 'channel_wise': args.channel_wise, 'scale_method': args.init}
    qnn = QuantModel(model=model, hadamard=args.hadamard, weight_quant_params=wq_params)
    qnn.to(device)

    qbits = qnn.set_bitwidth(args.precision)
    args.qbits = qbits

    qnn.eval()
    logging.info('quantized model architecture: {}'.format(qnn))
    
    # load calibration dataset
    cali_data = torch.cat(embedding_list, dim=0)
    device = next(qnn.parameters()).device
    logging.info('input embedding shape: {}'.format(cali_data.shape))
    
    
    # Initialize weight quantization parameters
    qnn.set_quant_state(True)
    logging.info(cali_data[:args.batch_size].shape)
    init_start = time.time()
    _ = qnn(cali_data[:args.batch_size].to(device))
    init_time = time.time() - init_start
    # print('Init time: {}'.format(init_time))
    logging.info('Init time: {}'.format(init_time))
    
    logging.info('=======================Close quantization model========================')
    qnn.set_quant_state(False)
    results_list, _, _ = evaluate(qnn, full_dataloader, args, cfg, args.dump_vis)
    print_str = 'Evaluation ... \n {} \n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    for i, (metric_name, metric_value) in enumerate(zip(args.metric_names, results_list)):
        cur_v = RoundTensor(metric_value.max(), 2 if 'psnr' in metric_name else 4)
        print_str += f'best_{metric_name}: {cur_v} | '
    logging.info(print_str) 
    
    
    logging.info('=======================Weight quantization model w/o opt========================')
    qnn.set_quant_state(True)
    results_list, _, _ = evaluate(qnn, full_dataloader, args, cfg, args.dump_vis)
    print_str = 'Evaluation ... \n {} \n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    for i, (metric_name, metric_value) in enumerate(zip(args.metric_names, results_list)):
        cur_v = RoundTensor(metric_value.max(), 2 if 'psnr' in metric_name else 4)
        print_str += f'best_{metric_name}: {cur_v} | '
    logging.info(print_str) 
    
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, gt=train_dataloader, arch = args.arch, batch_size = args.batch_size, 
                    iters=args.iters_w, weight=args.weight, opt_mode='mse', hadamard=args.hadamard,
                    b_range=(args.b_start, args.b_end),
                    warmup=args.warmup, 
                    p=args.norm_p, 
                    lr=args.lr)
    
    
    # Start calibration
    logging.info('[PID] %s'%os.getpid())
    msg = f'======================= Hyper Parameters ======================='
    logging.info(msg)
    logging.info('param init: {}'.format(args.init))
    logging.info('channel wise: {}'.format(args.channel_wise))
    logging.info('seed: {}'.format(args.seed))
    logging.info('iterations: {}'.format(args.iters_w))
    logging.info('batch_size: {}'.format(args.batch_size))
    logging.info('loss weight: {}'.format(args.weight))
    logging.info('input drop rate: {}'.format(args.input_prob))
    logging.info('average bit-width: {}'.format(args.qbits))
    end = f'========================== {args.arch} =========================='
    logging.info(end)
    
    
    logging.info(f'begin training in {next(qnn.parameters()).device}')
    start = datetime.now()
    
    qnn.set_quant_state(weight_quant=True)
    model_reconstruction(qnn, **kwargs)
    logging.info(f"Training complete in: {str(datetime.now() - start)}")

    qnn.set_quant_state(weight_quant=True)
    
    logging.info('=======================Weight quantization model w/ opt========================')
    print_str = 'Evaluation ... \n {} \n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    results_list, _, _ = evaluate(qnn, full_dataloader, args, cfg, args.dump_vis)
    for i, (metric_name, metric_value) in enumerate(zip(args.metric_names, results_list)):
        cur_v = RoundTensor(metric_value.max(), 2 if 'psnr' in metric_name else 4)
        print_str += f'best_{metric_name}: {cur_v} | '
    logging.info(print_str) 
    
    logging.info('save quantized model in {}'.format(args.outf))
    if args.channel_wise:
        torch.save(qnn, "{}/{}_W{}_prob{}_{}-init_CW.pth".format(args.outf ,args.arch, args.qbits, args.input_prob, args.init))
    else:
        torch.save(qnn, "{}/{}_W{}_prob{}_{}-init_LW.pth".format(args.outf ,args.arch, args.qbits, args.input_prob, args.init))
    

def main(argv):
    torch.set_printoptions(precision=4) 
    
    args = parse_args(argv)
    cfg = get_config(args.config)
    
    args.outf = os.path.join('results', args.outf)

    exp_id = f"{args.vid}_e{cfg['epoch']}_b{cfg['batch_size']}_lr{cfg['learning_rate']}_{cfg['loss']}"
    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)
    
    torch.set_printoptions(precision=2) 
    calibrate(args, cfg)
    

if __name__ == '__main__':
    main(sys.argv[1:])
