import torch
import torch.nn as nn

from typing import Union

from models import NeRV, HNeRV
from quantization.quant_block import specials, BaseQuantBlock
from quantization.quant_layer import QuantModule, StraightThrough


class QuantModel(nn.Module):

    def __init__(self, model: Union[NeRV, HNeRV], hadamard: bool=True, weight_quant_params: dict = {}):
        super().__init__()
        self.model = model
        self.hadamard = hadamard
        self.quant_module_refactor(self.model, weight_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}):
        r"""
            Recursively replace the module to QuantModule / QuantBlock.
        Args:
            module: nn.Module with children modules
            weight_quant_params: quantization parameters for weight quantizer
        """
        for name, child_module in module.named_children():
            
            if 'encoder' in name:
                continue
            
            elif type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, self.hadamard, weight_quant_params))

            elif isinstance(child_module, (nn.Conv2d, )):
                setattr(module, name, QuantModule(child_module, self.hadamard, weight_quant_params))
                
            elif isinstance(child_module, StraightThrough):
                continue                   
            
            else:
                self.quant_module_refactor(child_module, weight_quant_params)
                
    def set_quant_state(self, weight_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant)
    
    def encode(self, input):
        return self.model.encode(input)
    
    def decode(self, input):
        return self.model.decode(input)
        
    def forward(self, input):
        return self.model.decode(input)
    

    def set_bitwidth(self, bit, init=False):
        count = 0
        bits = 0.
        num_param = 0.
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.weight_quantizer.bitwidth_refactor(bit[count])
                m.weight_quantizer.inited = init
                m.bias_quantizer.bitwidth_refactor(bit[count])
                m.bias_quantizer.inited = init

                bits += m.weight_quantizer.n_bits * m.weight.numel() + m.bias_quantizer.n_bits * m.bias.numel()
                num_param += m.weight.numel() + m.bias.numel()
                count += 1
        return bits/num_param
    
    def get_quantized_param(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m.weight_quantizer.x_quant]
                module_list += [m.bias_quantizer.x_quant]
        return module_list

    def get_perturbation(self):
        perturbation_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                perturbation_list += [m.get_weight_perturbation()]
        return perturbation_list
