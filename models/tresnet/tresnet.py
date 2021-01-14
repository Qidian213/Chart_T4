from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.squeeze_and_excite import SEModule
from .layers.space_to_depth import SpaceToDepthModule
from inplace_abn import InPlaceABN

def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children(): IABN2Float(child)
    return module

def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )

class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
        
class CBNModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBNModule, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding=padding, bias=bias)
        self.bn   = nn.BatchNorm2d(outchannel)
        self.act  = HSwish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ContextModule(nn.Module):
    def __init__(self, inchannel):
        super(ContextModule, self).__init__()
    
        self.inconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)

        half = inchannel // 2
        self.upconv    = CBNModule(half, half, 3, 1, padding=1)
        self.downconv  = CBNModule(half, half, 3, 1, padding=1)
        self.downconv2 = CBNModule(half, half, 3, 1, padding=1)

    def forward(self, x):
        x    = self.inconv(x)
        up, down = torch.chunk(x, 2, dim=1)
        up   = self.upconv(up)
        down = self.downconv(down)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)
        
class UpModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=2, stride=2,  bias=False):
        super(UpModule, self).__init__()
        self.dconv = nn.Upsample(scale_factor=2)
        self.conv  = nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=bias)
        self.bn    = nn.BatchNorm2d(outchannel)
        self.act   = HSwish()
    
    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DetectModule(nn.Module):
    def __init__(self, inchannel):
        super(DetectModule, self).__init__()
    
        self.upconv  = CBNModule(inchannel, inchannel, 3, 1, padding=1)
        self.context = ContextModule(inchannel)

    def forward(self, x):
        up   = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out

class TResNet(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, remove_aa_jit=True):
        super(TResNet, self).__init__()

        # JIT layers
        self.space_to_depth = SpaceToDepthModule()
        anti_alias_layer    = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes   = int(64 * width_factor)
        self.conv1    = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        self.layer1   = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        self.layer2   = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        self.layer3   = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        self.layer4   = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7
                                  
                                  
        self.conn4  = CBNModule(64,   64, 1, 1)  # s4
        self.conn8  = CBNModule(128,  64, 1, 1)  # s8
        self.conn16 = CBNModule(1024, 64, 1, 1)  # s16
        self.conn32 = CBNModule(2048, 64, 1, 1)  # s32

        self.up8    = UpModule(64, 64, 2, 2) # s8  -> s4
        self.up16   = UpModule(64, 64, 2, 2) # s16 -> s8 
        self.up32   = UpModule(64, 64, 2, 2) # s32 -> s16

        self.cout   = DetectModule(64)
        
        self.head_hm  = nn.Conv2d(128, 2, 1)
        self.head_reg = nn.Conv2d(128, 2, 1)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat_list = {}
        
        x = self.space_to_depth(x)
        x = self.conv1(x)
        
        x = self.layer1(x)
        feat_list['s4']  = x
        
        x = self.layer2(x)
        feat_list['s8']  = x
        
        x = self.layer3(x)
        feat_list['s16'] = x
        
        x = self.layer4(x)
        feat_list['s32'] = x
        
        feat_list['s16'] = self.up32(self.conn32(feat_list['s32'])) + self.conn16(feat_list['s16'])
        feat_list['s8']  = self.up16(feat_list['s16']) + self.conn8(feat_list['s8'])
        feat_list['s4']  = self.up8(feat_list['s8']) + self.conn4(feat_list['s4'])
        
        feat_list['s4']  = self.cout(feat_list['s4'])
        
        hm  = self.head_hm(feat_list['s4'])
        hm  = hm.sigmoid()
        hm  = torch.clamp(hm, min=1e-4, max=1-1e-4)
        
        reg = self.head_reg(feat_list['s4'])
        
        result = {}
        result['heatmap'] = hm
        result['reg']     = reg
        return result
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def load_param(self, file):
        checkpoint = torch.load(file)
        
        if('state_dict' in checkpoint.keys()):
            checkpoint = checkpoint['state_dict']
            
        if('model' in checkpoint.keys()):
            checkpoint = checkpoint['model']
        
        model_state_dict = self.state_dict()
        new_state_dict   = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '').replace('body.', '')
            new_state_dict[name] = v
            if name in model_state_dict:
                if v.shape != model_state_dict[name].shape:
                    print('Skip loading parameter {}, required shape{}, '\
                          'loaded shape{}.'.format(name, model_state_dict[name].shape, v.shape))
                    new_state_dict[name] = model_state_dict[name]
            else:
                print('Drop parameter {}.'.format(name))

        for key in model_state_dict.keys():
            if(key not in new_state_dict.keys()):
                print('No param {}.'.format(key))
                new_state_dict[name] = model_state_dict[key]
        self.load_state_dict(new_state_dict, strict=False)
            
def TResnetM():
    """ Constructs a medium TResnet model.
    """
    model = TResNet(layers=[3, 4, 11, 3])
    return model

def TResnetL():
    """ Constructs a large TResnet model.
    """
    model = TResNet(layers=[4, 5, 18, 3],width_factor=1.2)
    return model

def TResnetXL():
    """ Constructs an extra-large TResnet model.
    """
    model = TResNet(layers=[4, 5, 24, 3], width_factor=1.3)
    return model
