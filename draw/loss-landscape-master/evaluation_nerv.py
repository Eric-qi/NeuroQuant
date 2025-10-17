"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable


def psnr_fn_single(output, gt):
    l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()

def psnr_fn_batch(output_list, gt):
    psnr_list = [psnr_fn_single(output.detach(), gt.detach()) for output in output_list]
    return torch.stack(psnr_list, 0).cpu()


def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    # net.eval()

    # with torch.no_grad():
    for i, sample in enumerate(loader):
        
        inputs, _, _ = sample['img'], sample['norm_idx'], sample['idx']
        targets = inputs.clone()
        
        
        batch_size = inputs.size(0)
        total += batch_size
        inputs = Variable(inputs)
        
        targets = Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs,_ ,_ = net(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()*batch_size
        correct += psnr_fn_batch([outputs], targets).sum().item()
        break

    # return total_loss/total, correct/total
    return loss, correct
