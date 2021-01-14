import torch
from torch import nn
from collections import OrderedDict
from .blocks.bottleneck_blocks import SimBottleneckCSP
from .blocks.trans_blocks import Focus
from .blocks.head_blocks import SPP
from .blocks.conv_blocks import ConvBase
from .blocks.asff_blocks import ASFFV5 as ASFF

class YoloV5_ASFF(nn.Module):

    def __init__(self, ch=3):
        super(YoloV5_ASFF, self).__init__()

        # divid by
        cd = 2
        wd = 3

        self.focus = Focus(ch, 64//cd)
        
        self.conv1 = ConvBase(64//cd, 128//cd, 3, 2)
        self.csp1 = SimBottleneckCSP(128//cd, 128//cd, n=3//wd) # 4s
        
        self.conv2 = ConvBase(128//cd, 256//cd, 3, 1)
        self.csp2 = SimBottleneckCSP(256//cd, 256//cd, n=9//wd) # 8s
        
        self.conv3 = ConvBase(256//cd, 512//cd, 3, 2)
        self.csp3 = SimBottleneckCSP(512//cd, 512//cd, n=9//wd) # 16s
        
        self.conv4 = ConvBase(512//cd, 1024//cd, 3, 2)
        self.spp = SPP(1024//cd, 1024//cd)
        self.csp4 = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False) # 32s

        # asff
        self.l0_fusion = ASFF(level=0, multiplier=1/cd)
        self.l1_fusion = ASFF(level=1, multiplier=1/cd)
        self.l2_fusion = ASFF(level=2, multiplier=1/cd)

        # PANet
        self.conv5 = ConvBase(1024//cd, 512//cd)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(1024//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv6 = ConvBase(512//cd, 256//cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512//cd, 256//cd, n=3//wd, shortcut=False)

        self.conv7 = ConvBase(256//cd, 256//cd, 3, 2)
        self.csp7 = SimBottleneckCSP(512//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv8 = ConvBase(512//cd, 512//cd, 3, 2)
        self.csp8 = SimBottleneckCSP(512//cd, 1024//cd, n=3//wd, shortcut=False)

        self.head_hm = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_reg = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )

        self.head_tag = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 128, out_channels = 2, kernel_size = 1, stride = 1, padding = 0)
            )
            
    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x_p3 = self.conv2(x)  # P3
        x = self.csp2(x_p3)
        x_p4 = self.conv3(x)  # P4
        x = self.csp3(x_p4)
        x_p5 = self.conv4(x)  # P5
        x = self.spp(x_p5)
        x = self.csp4(x)
        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x)
        return x_small, x_medium, x_large

    def forward(self, x):
        p3, p4, p5, feas = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, feas)
        xl = self.l0_fusion(xl, xm, xs)  # 4s
        xm = self.l1_fusion(xl, xm, xs)  # 8s
        xs = self.l2_fusion(xl, xm, xs)  # 16s

        hm  = self.head_hm(xs)
        hm  = hm.sigmoid()
        hm  = torch.clamp(hm, min=1e-4, max=1-1e-4)
        
        reg = self.head_reg(xs)
        tag = self.head_tag(xs)
        
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    img = torch.randn([1, 3, 512, 512]).cuda()

    model = YoloV5().cuda().eval()

    result = model(img)
    print(result['heatmap'].size())
    print(result['reg'].size())
    print(result['tag'].size())


