import os
import torch
from torch import nn
from collections import OrderedDict
from .blocks.bottleneck_blocks import SimBottleneckCSP
from .blocks.trans_blocks import Focus
from .blocks.head_blocks import SPP
from .blocks.conv_blocks import ConvBase

class YoloV5(nn.Module):
    def __init__(self, ch=3):
        super(YoloV5, self).__init__()
        
        # divid by
        cd = 2
        wd = 3

        self.focus = Focus(ch, 64//cd)                                               # 2s   [1, 32, 256, 256]
        
        self.conv1 = ConvBase(64//cd, 128//cd, 3, 2)                                 # 4s   [1, 64, 128, 128]
        self.csp1  = SimBottleneckCSP(128//cd, 128//cd, n=3//wd)                     # 4s   [1, 64, 128, 128]
        
        self.conv2 = ConvBase(128//cd, 256//cd, 3, 2)                                # 8s   [1, 64, 64, 64]
        self.csp2  = SimBottleneckCSP(256//cd, 256//cd, n=9//wd)                     # 8s   [1, 128, 64, 64]
        
        self.conv3 = ConvBase(256//cd, 512//cd, 3, 1)                                # 16s   [1, 128, 64, 64]
        self.csp3  = SimBottleneckCSP(512//cd, 512//cd, n=9//wd)                     # 16s  [1, 256, 64, 64]
        
        # self.conv4 = ConvBase(512//cd, 1024//cd, 3, 2)                               # 16s
        # self.spp   = SPP(1024//cd, 1024//cd)                                         # 32s
        # self.csp4  = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False)   # 32s

        # PANet
        self.conv5 = ConvBase(512//cd, 512//cd)                                     #
        self.up1   = nn.Upsample(scale_factor=2)                                     #
        self.csp5  = SimBottleneckCSP(1024//cd, 512//cd, n=3//wd, shortcut=False)    #

        self.conv6 = ConvBase(384, 256//cd)                                      #
        self.up2   = nn.Upsample(scale_factor=2)                                     #
        self.csp6  = SimBottleneckCSP(256//cd, 128//cd, n=3//wd, shortcut=False)     #

        self.head_hm = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_reg = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_tag = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )
            
    def _build_backbone(self, x):
        x    = self.focus(x)
        x    = self.conv1(x)
        x    = self.csp1(x)
        x_p3 = self.conv2(x)  # P3
        x    = self.csp2(x_p3)
        x_p4 = self.conv3(x)  # P4
        x    = self.csp3(x_p4)
        return x_p3, x_p4, x

    def _build_head(self, p3, p4, feas):
        h_p5     = self.conv5(feas)  
        x_concat = torch.cat([h_p5, p4], dim=1)  
        x        = self.csp5(x_concat)  # [1, 256, 64, 64]

        x        = torch.cat([x, p3], dim=1)
        x        = self.conv6(x)  
        x        = self.up2(x)            
        x        = self.csp6(x)
        return x

    def forward(self, x):
        p3, p4, feas = self._build_backbone(x)
        x = self._build_head(p3, p4, feas)
        
        hm  = self.head_hm(x)
        hm  = hm.sigmoid()
        hm  = torch.clamp(hm, min=1e-4, max=1-1e-4)
        
        reg = self.head_reg(x)
        tag = self.head_tag(x)
        
        result = {}
        result['heatmap'] = hm
        result['reg']     = reg
        result['tag']     = tag
        return result

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
                new_state_dict[name] = model_state_dict[key]
            
        self.load_state_dict(new_state_dict, strict=False)

class YoloV5_2(nn.Module):
    def __init__(self, ch=3):
        super(YoloV5, self).__init__()
        
        # divid by
        cd = 2
        wd = 3

        self.focus = Focus(ch, 64//cd)                                               # 2s   [1, 32, 256, 256]
        
        self.conv1 = ConvBase(64//cd, 128//cd, 3, 2)                                 # 4s   [1, 64, 128, 128]
        self.csp1  = SimBottleneckCSP(128//cd, 128//cd, n=3//wd)                     # 4s   [1, 64, 128, 128]
        
        self.conv2 = ConvBase(128//cd, 256//cd, 3, 2)                                # 8s   [1, 64, 64, 64]
        self.csp2  = SimBottleneckCSP(256//cd, 256//cd, n=9//wd)                     # 8s   [1, 128, 64, 64]
        
        self.conv3 = ConvBase(256//cd, 512//cd, 3, 1)                                # 8s   [1, 128, 64, 64]
        self.csp3  = SimBottleneckCSP(512//cd, 512//cd, n=9//wd)                     # 8s   [1, 256, 64, 64]
        
        # self.conv4 = ConvBase(512//cd, 1024//cd, 3, 2)                               # 16s
        # self.spp   = SPP(1024//cd, 1024//cd)                                         # 32s
        # self.csp4  = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False)   # 32s

        # PANet
        self.conv5 = ConvBase(512//cd, 512//cd)                                     #
        self.up1   = nn.Upsample(scale_factor=2)                                     #
        self.csp5  = SimBottleneckCSP(1024//cd, 512//cd, n=3//wd, shortcut=False)    #

        self.conv6 = ConvBase(384, 256//cd)                                      #
        self.up2   = nn.Upsample(scale_factor=2)                                     #
        self.csp6  = SimBottleneckCSP(256//cd, 128//cd, n=3//wd, shortcut=False)     #

        self.head_hm = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_reg = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_tag = nn.Sequential(
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )
            
    def _build_backbone(self, x):
        x    = self.focus(x)
        x    = self.conv1(x)
        x    = self.csp1(x)
        x_p3 = self.conv2(x)  # P3
        x    = self.csp2(x_p3)
        x_p4 = self.conv3(x)  # P4
        x    = self.csp3(x_p4)
        return x_p3, x_p4, x

    def _build_head(self, p3, p4, feas):
        h_p5     = self.conv5(feas)  
        x_concat = torch.cat([h_p5, p4], dim=1)  
        x        = self.csp5(x_concat)  # [1, 256, 64, 64]

        x        = torch.cat([x, p3], dim=1)
        x        = self.conv6(x)  
        x        = self.up2(x)            
        x        = self.csp6(x)
        return x

    def forward(self, x):
        p3, p4, feas = self._build_backbone(x)
        x = self._build_head(p3, p4, feas)
        
        hm  = self.head_hm(x)
        hm  = hm.sigmoid()
        hm  = torch.clamp(hm, min=1e-4, max=1-1e-4)
        
        reg = self.head_reg(x)
        tag = self.head_tag(x)
        
        result = {}
        result['heatmap'] = hm
        result['reg']     = reg
        result['tag']     = tag
        return result

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
                new_state_dict[name] = model_state_dict[key]
            
        self.load_state_dict(new_state_dict, strict=False)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    img = torch.randn([1, 3, 512, 512]).cuda()

    model = YoloV5().cuda().eval()

    result = model(img)
    print(result['heatmap'].size())
    print(result['reg'].size())
    print(result['tag'].size())
    