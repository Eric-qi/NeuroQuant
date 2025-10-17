import sys
import math
import yaml
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim





# get config
def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)
    
    
def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        return config


################## split one video into seen/unseen frames ##################
def data_split(img_list, split_num_list, shuffle_data, rand_num=0):
    valid_train_length, total_train_length, total_data_length = split_num_list
    # assert total_train_length < total_data_length
    temp_train_list, temp_val_list = [], []
    if shuffle_data:
        random.Random(rand_num).shuffle(img_list)
    for cur_i, frame_id in enumerate(img_list):
        if (cur_i % total_data_length) < valid_train_length:
            temp_train_list.append(frame_id)
        elif (cur_i % total_data_length) >= total_train_length:
            temp_val_list.append(frame_id)
    return temp_train_list, temp_val_list


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


def adjust_lr(optimizer, cur_epoch, args, eta_min=0.05):
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio)/ (1 - up_ratio)) + 1.0)
            lr_mult = max(lr_mult, eta_min)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult


############################ Function for loss compuation and evaluate metrics ############################

def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr


def loss_fn(pred, target, loss_type='l2', batch_average=True):
    target = target.detach()

    if loss_type == 'l2':
        loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'l1':
        loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'ssim':
        loss = 1 - ssim(pred, target, data_range=1, size_average=False)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion2':
        loss = 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion4':
        loss = 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion6':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion9':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion10':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion11':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion12':
        loss = 0.8 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    return loss.mean() if batch_average else loss

# psnr and ms-ssim
def psnr_fn_single(output, gt):
    l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()

def psnr_fn_batch(output_list, gt):
    psnr_list = [psnr_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(psnr_list, 0).cpu()


def msssim_fn_single(output, gt):
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()

def msssim_fn_batch(output_list, gt):
    msssim_list = [msssim_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(msssim_list, 0).cpu()


def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss + 1e-9)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr


def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim




# rgb convert
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)