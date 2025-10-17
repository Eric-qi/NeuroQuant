import time
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F



from quantization.quant_model import QuantModel
from quantization.quant_layer import QuantModule
from quantization.quant_block import BaseQuantBlock



r"""
    The utils for calibration.
    Based on 
    BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction
    https://arxiv.org/abs/2102.05426.
"""


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b
        
    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))



def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                        asym: bool = False, batch_size: int = 8, keep_gpu: bool = True, input_prob: bool = False):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    
    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, input_prob=input_prob)
    cached_batches = []
    
    # Releases all unoccupied cached memory currently held by the caching allocator 
    # so that those can be used in other GPU application and visible in nvidia-smi.
    torch.cuda.empty_cache() 
    
    for i in range(int(cali_data.size(0) / batch_size)):
        if input_prob:
            cur_inp, cur_out, cur_sym = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
        else:
            cur_inp, cur_out = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
            
    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()

    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    
    if input_prob:
        return (cached_inps, cached_sym), cached_outs
    return (cached_inps,), cached_outs


def save_grad_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                    batch_size: int = 8, keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device)
    cached_batches = []
    torch.cuda.empty_cache()
    
    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())
        
    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        
        self.input_store = None
        self.output_store = None
        
    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], 
                    device: torch.device, asym: bool = False, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):
        self.model.eval()
        self.model.set_quant_state(False)

        r"""Registers a forward hook on the module.
        The hook will be called every time after :func:`forward` has computed an output.
        It should have the following signature::
            hook(module, input, output) -> None or modified output
        The hook can modify the output. It can modify the input inplace but
        it will not have effect on forward since this is called after
        :func:`forward` is called.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass
            
            if self.input_prob:
                input_sym = self.data_saver.input_store[0].detach()
                
            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                
            self.data_saver.store_output = True
            
        handle.remove()
        
        self.model.set_quant_state(False)
        self.layer.set_quant_state(True)
        self.model.train()
        
        if self.input_prob:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach(), input_sym
        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()



class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None
        
    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                    device: torch.device):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False)
                out_fp, _, __ = self.model(inputs)
                quantize_model_till(self.model, self.layer)
                out_q, _, __ = self.model(inputs)
                loss = F.mse_loss(out_fp, out_q, reduction='none').flatten(1).mean(1)
                loss.backward()
            except StopForwardException:
                pass
            
        handle.remove()
        self.model.set_quant_state(False)
        self.layer.set_quant_state(True)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantModule, layer: Union[QuantModule, BaseQuantBlock]):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True)
        if module == layer:
            break