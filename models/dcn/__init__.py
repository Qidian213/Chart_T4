from .Dla_DCN import DLASeg
from .ResNet_DCN import PoseResNet, BasicBlock, Bottleneck


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_pose_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model

def get_pose_net(model_name):
    if(model_name == 'DLA_34'):
        model = DLASeg('dla34', heads= {'heatmap':2, 'reg': 2, 'tag': 2},
                        pretrained   = False,
                        down_ratio   = 4,
                        final_kernel = 1,
                        last_level   = 5,
                        head_conv    = 256
                    )
        return model

    if(model_name == 'DLA_46C'):
        model = DLASeg('dla46_c', heads= {'heatmap':2, 'reg': 2, 'tag': 2},
                        pretrained   = False,
                        down_ratio   = 4,
                        final_kernel = 1,
                        last_level   = 5,
                        head_conv    = 256
                    )
        return model

    if(model_name == 'DLA_46XC'):
        model = DLASeg('dla46x_c', heads= {'heatmap':2, 'reg': 2, 'tag': 2},
                        pretrained   = False,
                        down_ratio   = 4,
                        final_kernel = 1,
                        last_level   = 5,
                        head_conv    = 256
                    )
        return model

    if(model_name == 'DLA_60XC'):
        model = DLASeg('dla60x_c', heads= {'heatmap':2, 'reg': 2, 'tag': 2},
                        pretrained   = False,
                        down_ratio   = 4,
                        final_kernel = 1,
                        last_level   = 5,
                        head_conv    = 256
                    )
        return model

    if(model_name == 'DLA_60'):
        model = DLASeg('dla60', heads= {'heatmap':2, 'reg': 2, 'tag': 2},
                        pretrained   = False,
                        down_ratio   = 4,
                        final_kernel = 1,
                        last_level   = 5,
                        head_conv    = 256
                    )
        return model

    if(model_name == 'ResNet_Dcn18'):
        block_class, layers = resnet_spec[18]
        model = PoseResNet(block_class, layers,
                          heads     = {'heatmap':2, 'reg': 2, 'tag': 2},
                          head_conv = 64
                        )
        model.init_weights()
        return model
        
    if(model_name == 'ResNet_Dcn34'):
        block_class, layers = resnet_spec[34]
        model = PoseResNet(block_class, layers,
                          heads     = {'heatmap':2, 'reg': 2, 'tag': 2},
                          head_conv = 64
                        )
        model.init_weights()
        return model
        
    if(model_name == 'ResNet_Dcn50'):
        block_class, layers = resnet_spec[50]
        model = PoseResNet(block_class, layers,
                          heads     = {'heatmap':2, 'reg': 2, 'tag': 2},
                          head_conv = 64
                        )
        model.init_weights()
        return model
        
    if(model_name == 'ResNet_Dcn101'):
        block_class, layers = resnet_spec[101]
        model = PoseResNet(block_class, layers,
                          heads     = {'heatmap':2, 'reg': 2, 'tag': 2},
                          head_conv = 64
                        )
        model.init_weights()
        return model
        
    if(model_name == 'ResNet_Dcn152'):
        block_class, layers = resnet_spec[152]
        model = PoseResNet(block_class, layers,
                          heads     = {'heatmap':2, 'reg': 2, 'tag': 2},
                          head_conv = 64
                        )
        model.init_weights()
        return model
        
        