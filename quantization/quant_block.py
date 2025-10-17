import torch.nn as nn

from models._layers import NeRVBlock
from quantization.quant_layer import QuantModule


class BaseQuantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.trained = False
        self.ignore_reconstruction = False
        
    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant)


class QuantNeRVBlock(BaseQuantBlock):
    """
    Implementation of Quantized NeRVBlock.
    """
    def __init__(self, basic_block: NeRVBlock, hadamard: bool=True, weight_quant_params: dict = {}):
        super().__init__()
        self.conv = QuantModule(basic_block.conv[0], hadamard, weight_quant_params)
        self.pixelshuffle = basic_block.conv[1]
        self.act = basic_block.act
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixelshuffle(x)
        x = self.act(x)
        return x

specials = {
    NeRVBlock: QuantNeRVBlock,
}
