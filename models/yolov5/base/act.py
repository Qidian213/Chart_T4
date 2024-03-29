import torch.nn as nn
import torch.nn.functional as F

"""

We define a activation layer
by a simple cfg


Do u know which activation func is better?

ReLU is the base, others just too fancy, their impact
on final metric is not so big maybe??

"""

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
        
activation_cfg = {
    # layer_abbreviation: module
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'SELU': nn.SELU,
    'CELU': nn.CELU,
    # new added
    'Hardswish': HSwish  # check pytorch version, >= 1.6
}


def build_activation_layer(cfg):
    """ Build activation layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a activation layer.

    Returns:
        layer (nn.Module): Created activation layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    if layer_type not in activation_cfg:
        raise KeyError('Unrecognized activation type {}'.format(layer_type))
    else:
        activation = activation_cfg[layer_type]
        if activation is None:
            raise NotImplementedError
    layer = activation(**cfg_)
    return layer
