import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import logging

from models import HNeRV, NeRV
from quantization.quant_layer import QuantModule
from quantization.quant_model import QuantModel
from quantization.quantizer import AdaRoundQuantizer, lp_loss
from quantization.data_utils import LinearTempDecay


class LossFunction:
    def __init__(self,
                    model: nn.Module,
                    round_loss: str = 'relaxation',
                    weight: float = 1.,
                    rec_loss: str = 'mse',
                    max_count: int = 2000,
                    b_range: tuple = (10, 2),
                    decay_start: float = 0.0,
                    warmup: float = 0.0,
                    p: float = 2.,):
        
        self.model = model
        self.round = round_loss
        self.weight = weight
        self.rec = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                            start_b=b_range[0], end_b=b_range[1])
        self.count = 0
    
    def collect_round_loss(self, module, b):
        for name, module in module.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                round_vals = module.weight_quantizer.get_soft_targets()
                self.round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                self.collect_round_loss(module, b)
                
        
    def __call__(self, pred, tgt, grad=None):
        r"""
            Compute the total loss for calibration:  
                rec_loss is the output reconstruction loss of current layer, 
                round_loss is a regularization term to optimize the rounding policy.
                
        Args:
            pred (tensor): output from current quantized layer
            tgt (tensor): the floating-point output of current layer
            grad (tensor): gradients to compute fisher information
            return: total loss function
        """
        self.count += 1
        
        if self.rec == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec))
        
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round == 'none':
            b = self.round_loss = 0
        elif self.round == 'relaxation':
            self.round_loss = 0
            self.collect_round_loss(self.model, b)
        else:
            raise NotImplementedError
        
        total_loss = self.round_loss + rec_loss
        if self.count % 500 == 0:
            logging.info('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f})\tb={:.2f}\tcount={}'.format(
                            float(total_loss), float(rec_loss), float(self.round_loss), b, self.count))
        return total_loss


def model_reconstruction(model: QuantModel, cali_data: torch.Tensor, gt: DataLoader, arch: str = 'hnerv', 
                            batch_size: int = 8, iters: int = 20000, weight: float = 0.01, 
                            opt_mode: str = 'mse', hadamard: bool=True, b_range: tuple = (20, 2), 
                            warmup: float = 0.0, p: float = 2.0, lr: float = 0.0015):
    r"""
        Network-wise Calibration.
        
    Args:
            model: QuantModel
            cali_data: data for calibration
            gt: ground truth
            arch: the used architecture
            
            batch_size: mini-batch size
            iters: optimization iterations
            weight: the weight of rounding regularization term
            opt_mode: optimization mode

            b_range: temperature range
            warmup: proportion of iterations that no scheduling for temperature
            p: L_p norm minimization

            lr: learning rate
    """
    
    model.set_quant_state(True)
    round_mode = 'learned_hard_sigmoid'
    
    ## Calibrate scaling factor s
    opt_params_delta = []
    def set_optimizer(model: nn.Module, opt_params_delta):
        for name, module in model.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                opt_params_delta += [module.weight_quantizer.delta]
                opt_params_delta += [module.bias_quantizer.delta]
            else:
                set_optimizer(module, opt_params_delta)
    
    
    set_optimizer(model, opt_params_delta)
    optimizer0 = torch.optim.Adam(opt_params_delta, lr=0.001)
    scheduler = None
    
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    
    loss_func0 = LossFunction(model, round_loss='none', weight=weight, 
                                max_count=2100, rec_loss=rec_loss, b_range=b_range, 
                                decay_start=0, warmup=warmup, p=p)
    # opt s
    epochs = int(0.05 * iters  / len(gt))
    for epoch in range(epochs):
        model.train()
        device = next(model.parameters()).device
        for i, sample in enumerate(gt):
            
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            inputs = cali_data[img_idx]
            if arch == 'hnerv':
                img_out, _, _ = model(inputs)
            elif arch == 'nerv':
                img_out, _, _ = model(inputs)
            else:
                raise ValueError
                
            optimizer0.zero_grad()
            err = loss_func0(pred=img_out, tgt=img_data, grad=None)
            err.backward()
            
            optimizer0.step()
            if scheduler:
                scheduler.step()
            
    torch.cuda.empty_cache()
    
    ## Calibrate adaptive rounding parameter alpha
    opt_params = []
    def set_optimizer(model: nn.Module, opt_params):
        for name, module in model.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                if hadamard:
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.hadamard_weight.data)
                else:
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data)
                
                module.bias_quantizer = AdaRoundQuantizer(uaq=module.bias_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.bias.data)

                module.weight_quantizer.soft_targets = True
                opt_params += [module.weight_quantizer.alpha]
                module.bias_quantizer.soft_targets = True
                opt_params += [module.bias_quantizer.alpha]
            else:
                set_optimizer(module, opt_params)
    
    
    set_optimizer(model, opt_params)
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = None
    
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    
    loss_func = LossFunction(model, round_loss=loss_mode, weight=weight, 
                                max_count=iters, rec_loss=rec_loss, b_range=b_range, 
                                decay_start=0, warmup=warmup, p=p)
    # opt alpha
    epochs = int(iters / len(gt)) - epochs
    for epoch in range(epochs):
        model.train()
        device = next(model.parameters()).device
        for i, sample in enumerate(gt):
            
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            inputs = cali_data[img_idx]
            if arch == 'hnerv':
                img_out, _, _ = model(inputs)
            elif arch == 'nerv':
                img_out, _, _ = model(inputs)
            else:
                raise ValueError
                
            optimizer.zero_grad()
            err = loss_func(pred=img_out, tgt=img_data, grad=None)
            err.backward()
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
    torch.cuda.empty_cache()
    # model.eval()
    
    def set_quantizer(model: nn.Module):
        for name, module in model.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                module.weight_quantizer.soft_targets = False
            else:
                set_quantizer(module)
    
    set_quantizer(model)