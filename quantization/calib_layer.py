import torch
import torch.nn as nn

import time
import logging

from quantization.quant_layer import QuantModule
from quantization.quant_model import QuantModel
from quantization.quantizer import AdaRoundQuantizer, lp_loss
from quantization.data_utils import LinearTempDecay, save_grad_data, save_inp_oup_data


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


def layer_reconstruction(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor, 
                            batch_size: int = 8, iters: int = 20000, weight: float = 0.01, 
                            opt_mode: str = 'mse', asym: bool = False, b_range: tuple = (20, 2),
                            warmup: float = 0.0, input_prob: float = 1.0, p: float = 2.0, lr: float = 0.0015):
    r"""
        Layer-wise Calibration
        
    Args:
            model: QuantModel
            layer: the layer to be calibrated
            cali_data: data for calibration
            
            batch_size: mini-batch size
            iters: optimization iterations
            weight: the weight of rounding regularization term
            opt_mode: optimization mode
            
            asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
            b_range: temperature range
            
            warmup: proportion of iterations that no scheduling for temperature
            input_prob: drop strategy proposed in QDrop (0-1)
            p: L_p norm minimization

            lr: learning rate
    """

    model.set_quant_state(False)
    layer.set_quant_state(True)
    round_mode = 'learned_hard_sigmoid'
    
    # Replace weight quantizer to AdaRoundQuantizer
    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                weight_tensor=layer.org_weight.data)
    layer.weight_quantizer.soft_targets = True
    layer.bias_quantizer = AdaRoundQuantizer(uaq=layer.bias_quantizer, round_mode=round_mode,
                                                weight_tensor=layer.bias.data)
    layer.bias_quantizer.soft_targets = True
    
    # Set up optimizer
    layer.weight_quantizer.soft_targets = True
    opt_params += [layer.weight_quantizer.alpha]
    layer.bias_quantizer.soft_targets = True
    opt_params += [layer.bias_quantizer.alpha]

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = None
    
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    
    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                                max_count=iters, rec_loss=rec_loss, b_range=b_range,
                                decay_start=0, warmup=warmup, p=p)
    
    # Save data before optimizing the rounding
    cached_start = time.time()
    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, batch_size=10, input_prob=True, keep_gpu=False)
    cached_time = time.time() - cached_start
    logging.info('Cached init time: {}'.format(cached_time))
    
    
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, layer, cali_data, batch_size=1)
    else:
        cached_grads = None
        
    device = next(model.parameters()).device
    for i in range(iters):
        # Returns a random permutation of integers from 0 to n - 1.
        idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
        cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx].to(device)
        if input_prob < 1.0:
            cur_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None
        
        optimizer.zero_grad()
        out_quant = layer(cur_inp)
        
        err = loss_func(pred=out_quant, tgt=cur_out, grad=cur_grad)
        err.backward(retain_graph=True)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
    torch.cuda.empty_cache()
    
    layer.weight_quantizer.soft_targets = False
    layer.bias_quantizer.soft_targets = False