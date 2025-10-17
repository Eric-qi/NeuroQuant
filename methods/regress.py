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
import torch.optim as optim
from torch.utils.data import Subset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


sys.path.append('/sjq/NeuroQuant/')
from videosets import VideoDataSet
from models import NeRV, HNeRV, PNeRV
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
    
    # dataset
    parser.add_argument('--data_path', type=str, help='data path for vid')
    parser.add_argument('--vid', type=str, help='video id',)
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    
    # learning
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    
    # eval
    parser.add_argument('--weight', default='None', type=str, help='model for test')
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--dump_vis', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    

    args = parser.parse_args(argv)
    return args

def seed_all(seed=903):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


@torch.no_grad()
def evaluate(model, full_dataloader, args, cfg, dump_vis=False):
    img_embed_list = []
    metric_list = [[] for _ in range(len(args.metric_names))]
    dec_time_list = []
    
    model.eval()
    device = next(model.parameters()).device
    
    if dump_vis:
        visual_dir = f'{args.outf}/visualize_fp32'
        logging.info(f'Saving predictions to {visual_dir}...')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)
    
    for i, sample in enumerate(full_dataloader):
        # forward
        if args.arch == 'hnerv':
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            img_out, embed_list, dec_time = model(img_data)
        elif args.arch == 'nerv':
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            img_out, embed_list, dec_time = model(norm_idx)
        elif args.arch == 'pnerv':
            # img_idx, img_data, img_p, img_f = sample['img_id'].to(device), sample['img_gt'].to(device), sample['img_p'].to(device), sample['img_f'].to(device)
            # img_out, embed_list, dec_time = model(img_data, img_p, img_f)
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            img_out, embed_list, dec_time = model(img_data)
        img_embed_list.append(embed_list[0])
        
        # collect decoding fps
        dec_time_list.append(dec_time)
        if args.eval_fps:
            dec_time_list.pop()
            for _ in range(100):
                img_out, embed_list, dec_time = model.decode(embed_list[0])
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
                img_ind = cur_img_idx.cpu().numpy()
                # dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                # concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                save_image(img_out[batch_ind], f'{visual_dir}/pred_{img_ind:04d}_{temp_psnr_list}.png')

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
    
    return results_list, (h,w)


def train(args, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]
    
    # setup dataloader
    full_dataset = VideoDataSet(cfg, args)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['workers'], pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, False, 0)
    
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
            num_workers=cfg['workers'], pin_memory=True, sampler=None, drop_last=True, worker_init_fn=worker_init_fn)

    # build model
    if args.arch == 'hnerv':
        model = HNeRV(cfg)
        # get parameters and flops
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6) 
        embed_param = float(cfg['enc_channel'][-1]) / np.prod(cfg['enc_strides'])**2 * args.final_size * args.full_data_length
        total_param = decoder_param + embed_param / 1e6
    elif args.arch == 'pnerv':
        model = PNeRV(cfg)
        # get parameters and flops
        # enc_cont_param = (sum([p.data.nelement() for p in model.enc_layers.parameters()]) / 1e6) 
        # enc_diff_param = (sum([p.data.nelement() for p in model.enc_diff_layers.parameters()]) / 1e6) 
        # encoder_param = enc_cont_param + enc_diff_param
        # decoder_param = (sum([p.data.nelement() for p in model.parameters()]) / 1e6) - encoder_param
        # embed_cont_param = float(cfg['emd_channel']) / np.prod(cfg['enc_strides'])**2 * args.final_size * args.full_data_length
        # embed_diff_param = 2 / np.prod(cfg['dif_strides'])**2 * args.final_size * args.full_data_length
        # embed_param = embed_cont_param + embed_diff_param
        # total_param = decoder_param + embed_param / 1e6
        encoder_param = (sum([p.data.nelement() for p in model.enc_layers.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in model.parameters()]) / 1e6) - encoder_param
        embed_param = float(cfg['emd_channel']) / np.prod(cfg['enc_strides'])**2 * args.final_size * args.full_data_length
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
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)
    writer = SummaryWriter(os.path.join(args.outf, 'tensorboard'))
    
    setup_logger(args.outf + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    logging.info('[PID] %s'%os.getpid())
    logging.info('================== Model Architecture=================')
    logging.info(str(model))
    logging.info(f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M')
    
    # load pretrained model for evaluation or resume
    if args.weight != 'None':
        logging.info("=> loading checkpoint '{}'".format(args.weight))
        checkpoint = torch.load(args.weight, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
    
    
    
    if args.eval_only:
        print_str=''
        logging.info('Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight))
        results_list, _ = evaluate(model, full_dataloader, args, cfg, args.dump_vis)
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
            
        logging.info(print_str)       
        return
    
    # setup optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    args.lr = cfg['learning_rate']
    
    # train
    start = datetime.now()
    
    psnr_list = []
    ssim_list = []
    
    logging.info(f'begin training in {next(model.parameters()).device}')
    for epoch in range(cfg['epoch']):
        model.train()
        device = next(model.parameters()).device
        
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        
        # iterate over dataloader
        for i, sample in enumerate(train_dataloader):
            # forward and backward
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / cfg['epoch']
            lr = adjust_lr(optimizer, cur_epoch, args)
            if args.arch == 'hnerv':
                img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
                img_out, _, _ = model(img_data)
            elif args.arch == 'pnerv':
                # img_idx, img_data, img_p, img_f = sample['img_id'].to(device), sample['img_gt'].to(device), sample['img_p'].to(device), sample['img_f'].to(device)
                # img_out, _, _ = model(img_data, img_p, img_f)
                img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
                img_out, _, _ = model(img_data)
            elif args.arch == 'nerv':
                img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
                img_out, _, _ = model(norm_idx)
            
            final_loss = loss_fn(img_out, img_data, cfg['loss'])      
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            # write result
            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_data)) 
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                logging.info('[{}], Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch+1, cfg['epoch'], i+1, len(train_dataloader), lr, 
                    RoundTensor(pred_psnr, 2)))
        
        # write to Tensorboard
        h, w = img_out.shape[-2:]
        writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
        writer.add_scalar('Train/lr', lr, epoch+1)
        epoch_end_time = datetime.now()
        logging.info("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                (epoch_end_time - start).total_seconds() / (epoch + 1) ))
        
        # evaluation
        if (epoch + 1) % cfg['eval_freq'] == 0 or (cfg['epoch'] - epoch) in [1, 3, 5]:
            results_list, hw = evaluate(model, full_dataloader, args, cfg,
                args.dump_vis if epoch == cfg['epoch'] - 1 else False)            

            # write to Tensorboard
            print_str = f'Eval at epoch {epoch+1} for {hw}: '
            for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                if 'pred_seen_psnr' in metric_name:
                    writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                    writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                    psnr_list.append(metric_value.max())
                    print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
                elif 'pred_seen_ssim' in metric_name:
                    writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                    writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                    ssim_list.append(metric_value.max())
                    print_str += f'{metric_name}: {RoundTensor(metric_value, 4)} | '
                best_metric_list[i] = best_metric_value
            logging.info(print_str)
        
        # save
        torch.save(model.state_dict(), '{}/model_latest.pth'.format(args.outf))
        if (epoch + 1) % cfg['epoch'] == 0:
            torch.save(model.state_dict(), f'{args.outf}/epoch{epoch+1}.pth')
    
    logging.info(f"Training complete in: {str(datetime.now() - start)}")

def main(argv):
    torch.set_printoptions(precision=4) 
    
    args = parse_args(argv)
    cfg = get_config(args.config)
    
    args.outf = os.path.join('results', args.outf)

    exp_id = f"{args.vid}_e{cfg['epoch']}_b{cfg['batch_size']}_lr{cfg['learning_rate']}_{cfg['loss']}"
    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)
    
    torch.set_printoptions(precision=2) 
    train(args, cfg)
    

if __name__ == '__main__':
    main(sys.argv[1:])
