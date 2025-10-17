import os
import sys
import time
import copy
import random
import logging
import argparse
import numpy as np
from math import inf
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


sys.path.append('/sjq/NeuroQuant/')
from quantization import *
from models import NeRV, HNeRV
from videosets import VideoDataSet
from utils import data_split, RoundTensor, psnr_fn_batch, msssim_fn_batch, \
    worker_init_fn, adjust_lr, loss_fn, psnr_fn_single, get_config, setup_logger



# toy example
hnerv_candidate = {
        'candidate1': [2, 3, 4, 6, 4, 4, 2], # 4.96 bit
        'candidate2': [6, 5, 4, 5, 5, 6, 6], # 4.79 bit
    }

nerv_candidate = {
        'candidate1': [5, 6, 3, 4, 5, 4, 3], # 5.47 bit
        'candidate2': [6, 5, 5, 6, 7, 6, 7], # 5.12 bit
    }

def gradtensor_to_vec(net):
    """ Extract gradients from net, and return a gradient list.

        Args:
            net: trained model with gradient

        Returns:
            a list containing all gradients
    """
    params = []
    for k, v in net.named_parameters():
        if 'encoder' in k:
            continue
        else:
            if "weight" in k:
                params.append(v.grad.data)

    return params

def run_hessian_vector_product(arch, vec, params, net, criterion, dataloader, use_cuda=True):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net.
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]
    device = 'cuda' if use_cuda else 'cpu'

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net

    count = 0.
    for i, sample in enumerate(dataloader):
        # forward
        img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
        img_data, norm_idx, img_idx = Variable(img_data), Variable(norm_idx), Variable(img_idx)
        img_data, norm_idx, img_idx = img_data.to(device), norm_idx.to(device), img_idx.to(device)
        if arch == 'hnerv':
            img_out, embed_list, dec_time = net.decode(net.encode(img_data))
        elif arch == 'nerv':
            img_out, embed_list, dec_time = net.decode(net.encode(norm_idx))

        loss = criterion(img_out, img_data)
        '''Note: 
            The grad can be fixed for all candidates. 
            Here we calculated grad in each candidate for readers to understand the pipeline of Hessian Vector. 
        '''
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True, allow_unused=True)


        ########## verify unused parameter ############
        # unused_params = []
        # for p, g in zip(params, grad_f):
        #     if g is None:
        #         unused_params.append(p)
        # print(f"[INFO] {len(unused_params)} parameters not used in loss computation.")
        # for i, p in enumerate(unused_params[:10]):
        #     print(f"  - Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")


        # Compute inner product of gradient with the direction vector
        prod = Variable(torch.zeros(1)).type(type(grad_f[0].data)).to(device)
        for (g, v) in zip(grad_f, vec):
            prod = prod.cuda() + (g * v).sum()
        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward(retain_graph=True)

        count += 1
        if count >=10:
            break



def run_approx_param_fisher(arch, vec, params, net, criterion, dataloader, use_cuda=True):
    """
    Evaluate the parameters' Fisher Information Matrix (FIM) of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Note: actually it can use the activation (i.e., the layer/block/network output). 
        Here we use FIM of parameter instead of activation for toy example.
        The pipeline is similar and the readers can modify using activation.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net.
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]
    device = 'cuda' if use_cuda else 'cpu'

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net

    count = 0.
    for i, sample in enumerate(dataloader):
        # forward
        
        img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
        img_data, norm_idx, img_idx = Variable(img_data), Variable(norm_idx), Variable(img_idx)
        img_data, norm_idx, img_idx = img_data.to(device), norm_idx.to(device), img_idx.to(device)
        if arch == 'hnerv':
            img_out, embed_list, dec_time = net.decode(net.encode(img_data))
        elif arch == 'nerv':
            img_out, embed_list, dec_time = net.decode(net.encode(norm_idx))

        loss = criterion(img_out, img_data)
        '''Note: 
            The grad can be fixed for all candidates. 
            Here we calculated grad in each candidate for readers to understand the pipeline of Hessian Vector. 
        '''
        loss.backward(retain_graph=True)

        count += 1
        if count >=10:
            break


def sensitivity_criterion(mode, arch, net, qnn, dataloader, use_cuda=False):
    """
    Evaluate sensitivity criteria under different mixed-precision configs.

    Mode:
        omega: delta_w * Hessian * delta_w in Section 3.1 of paper.
        fisher_diag: diagonal Fisher Information of parameter in Appendix C of paper
    """

    params = []
    for k, v in net.named_parameters():
        if 'encoder' in k:
            continue
        else:
            if "weight" in k:
                v.requires_grad = True
                params.append(v)

    vec = qnn.get_perturbation()
    criterion = nn.MSELoss()

    if mode=="omega":
        run_hessian_vector_product(arch, vec, params, net, criterion, dataloader, use_cuda)

        hess_vec_prod = gradtensor_to_vec(net)
        omega = 0.
        count = 0
        for (g, v) in zip(hess_vec_prod, vec):
            cur_omega = (g * v).sum()
            omega += cur_omega
            logging.info(f"[{count:d}-th layer] {cur_omega:.3e}")
            count += 1
        return omega
    elif mode=="fisher_diag":
        run_approx_param_fisher(arch, vec, params, net, criterion, dataloader, use_cuda)

        grad = gradtensor_to_vec(net)
        fim_diag = 0.
        count = 0
        for (g, v) in zip(grad, vec):
            fisher_diag = (v.pow(2) * g.pow(2)).sum()
            fim_diag += fisher_diag
            logging.info(f"[{count:d}-th layer] {fisher_diag:.3e}")
            count += 1
        return fim_diag
    else:
        raise ValueError('Not implemented sensitivity criteria: {}'.format(mode))





@torch.no_grad()
def evaluate(model, full_dataloader, args, cfg):
    img_embed_list = []
    metric_list = [[] for _ in range(len(args.metric_names))]
    dec_time_list = []
    
    model.eval()
    device = next(model.parameters()).device
    
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


def assign(args, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim']
    
    # setup dataloader
    full_dataset = VideoDataSet(cfg, args)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=cfg['workers'], pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, False, 0)
    
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
    args.outf = os.path.join(args.outf, "sensitivity-{}_{}-init_batch{}_CW".format(args.mode, args.init, args.batch_size))
    
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    setup_logger(args.outf + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    
    # load pretrained model for evaluation or resume
    assert args.ckpt != 'None'
    logging.info("=> loading checkpoint '{}'".format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    
    logging.info('=======================Full-precision model========================')
    print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.ckpt)
    results_list, _, embedding_list = evaluate(model, full_dataloader, args, cfg)
    for i, (metric_name, metric_value) in enumerate(zip(args.metric_names, results_list)):
        cur_v = RoundTensor(metric_value.max(), 2 if 'psnr' in metric_name else 4)
        print_str += f'best_{metric_name}: {cur_v} | '
    logging.info(print_str) 
    

    # allocate bit-width according sensitivity score
    if args.arch == 'hnerv':
        candidate_dict = hnerv_candidate
    elif args.arch == 'nerv':
        candidate_dict = nerv_candidate

        
    best_score = float('inf')
    best_candidate = None
    best_bits = None

    for candidate, bits in candidate_dict.items():
        # build quantization parameters
        wq_params = {'n_bits': 8, 'channel_wise': args.channel_wise, 'scale_method': args.init}
        qnn = QuantModel(model=copy.deepcopy(model), hadamard=args.hadamard, weight_quant_params=wq_params)
        qnn.to(device)
        qnn.eval()

        qbits = qnn.set_bitwidth(bits)
        args.qbits = qbits

        # load calibration dataset
        cali_data = torch.cat(embedding_list, dim=0)
        device = next(qnn.parameters()).device

        # Initialize weight quantization parameters
        qnn.set_quant_state(True)
        _ = qnn(cali_data[:args.batch_size].to(device))
        avg_bits = float(args.qbits)
        logging.info(f"[{candidate}: {bits}] Average Quantization Bit-Width:\t{avg_bits:.4f}")

        # compute score
        omega = sensitivity_criterion(args.mode, args.arch, copy.deepcopy(model), qnn, full_dataloader, use_cuda=True)
        score = omega.item()
        logging.info(f"[{candidate}: {bits}] The {args.mode} sensitivity score =\t{score:.3e}")

        # update best
        if score < best_score:
            best_score = score
            best_candidate = candidate
            best_bits = bits

    # ==== Final result ====
    logging.info("=" * 60)
    logging.info(f"✅ Best Candidate: {best_candidate}")
    logging.info(f"✅ Bit Configuration: {best_bits}")
    logging.info(f"✅ Minimum Score: {best_score:.4e}")
    logging.info("=" * 60)

    return best_candidate, best_bits, best_score
        

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
    parser.add_argument('--hadamard', action='store_true', help='apply hadamard transform for weights')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--init', default='max', type=str, help='param init type', 
                        choices=['max','mse', 'gaussian', 'l1', 'l2', ])
    parser.add_argument('--mode', default='omega', type=str, help='optimization mode type', 
                        choices=['omega','fisher_diag',])
    
    # eval
    parser.add_argument('--ckpt', default='None', type=str, help='model for test')
    
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

def main(argv):
    seed_all()
    torch.set_printoptions(precision=4) 
    
    args = parse_args(argv)
    cfg = get_config(args.config)
    
    args.outf = os.path.join('results', args.outf)

    exp_id = f"{args.vid}_e{cfg['epoch']}_b{cfg['batch_size']}_lr{cfg['learning_rate']}_{cfg['loss']}"
    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)
    
    torch.set_printoptions(precision=2) 
    assign(args, cfg)
    

if __name__ == '__main__':
    main(sys.argv[1:])
