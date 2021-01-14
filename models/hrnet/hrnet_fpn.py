# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method    = fuse_method
        self.num_branches   = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches    = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu        = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], 
                            stride, 
                            downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[x[i].shape[2], x[i].shape[3]],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

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
        
blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class FPHighResolutionNet(nn.Module):
    def __init__(self, config, **kwargs):
        self.inplanes = 64
        extra = config
        super(FPHighResolutionNet, self).__init__()

        # stem net
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
        self.conv2  = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn2    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
        self.relu   = nn.ReLU(inplace=True)
        self.sf     = nn.Softmax(dim=1)
        
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg  = extra['STAGE2']
        num_channels     = self.stage2_cfg['NUM_CHANNELS']
        block            = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg  = extra['STAGE3']
        num_channels     = self.stage3_cfg['NUM_CHANNELS']
        block            = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg  = extra['STAGE4']
        num_channels     = self.stage4_cfg['NUM_CHANNELS']
        block            = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.Conv1_t2 = CBNModule(self.stage2_cfg['NUM_CHANNELS'][0], self.stage2_cfg['NUM_CHANNELS'][0], 1, 1)  # s4
        self.Conv1_t3 = CBNModule(self.stage2_cfg['NUM_CHANNELS'][0], self.stage2_cfg['NUM_CHANNELS'][0], 1, 1)  # s8
        self.Conv1_t4 = CBNModule(self.stage2_cfg['NUM_CHANNELS'][0], self.stage2_cfg['NUM_CHANNELS'][0], 1, 1)  # s16

        ### 1,2
        self.head_hm = nn.Sequential(
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][0], out_channels = self.stage2_cfg['NUM_CHANNELS'][1], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(self.stage2_cfg['NUM_CHANNELS'][1], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][1], out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
        )

        self.head_reg = nn.Sequential(
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][0], out_channels = self.stage2_cfg['NUM_CHANNELS'][1], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(self.stage2_cfg['NUM_CHANNELS'][1], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][1],out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
        )

        self.head_tag = nn.Sequential(
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][0], out_channels = self.stage2_cfg['NUM_CHANNELS'][1], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(self.stage2_cfg['NUM_CHANNELS'][1], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = self.stage2_cfg['NUM_CHANNELS'][1],out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
        )
        
        self.init_weights()
        
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur  = len(num_channels_cur_layer)
        num_branches_pre  = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels  = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules  = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks   = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block        = blocks_dict[layer_config['BLOCK']]
        fuse_method  = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer1(x)

        Fpn_list = []
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)  ## [1, 18, 256, 256], [1, 36, 128, 128]
        Fpn_list.append(y_list[0])

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)  ## [1, 18, 256, 256], [1, 36, 128, 128], [1, 72, 64, 64]
        Fpn_list.append(y_list[0])

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list) ##[1, 18, 256, 256],[1, 36, 128, 128],[1, 72, 64, 64],[1, 144, 32, 32]
        Fpn_list.append(y_list[0])

        #x = Fpn_list[0] + Fpn_list[1] + Fpn_list[2]          ###  1  
        x = self.Conv1_t2(Fpn_list[0]) + self.Conv1_t3(Fpn_list[1]) + self.Conv1_t4(Fpn_list[2])  ### 2  
        #x = torch.cat(Fpn_list, dim=1)       ### 3

        hm  = self.head_hm(x).sigmoid()
        hm  = torch.clamp(hm, min=1e-4, max=1-1e-4)
        
        reg = self.head_reg(x)
        tag = self.head_tag(x)
        
        result = {}
        result['heatmap'] = hm
        result['reg']     = reg
        result['tag']     = tag
        return result

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_param(self, file):
        checkpoint = torch.load(file)
        
        if('state_dict' in checkpoint.keys()):
            checkpoint = checkpoint['state_dict']
        
        model_state_dict = self.state_dict()
        new_state_dict   = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
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
                new_state_dict[key] = model_state_dict[key]
            
        self.load_state_dict(new_state_dict, strict=False)
