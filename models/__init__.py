import torch
from .mobilenet import DBFace
from .hrnet import HighResolutionNet,FPHighResolutionNet
from .tresnet import TResnetM, TResnetL, TResnetXL
from .dcn import get_pose_net
from .yolov5 import YoloV5, YoloV5_ASFF

MODEL_HR18 = {}
MODEL_HR18['STAGE2'] = {}
MODEL_HR18['STAGE2']['NUM_CHANNELS'] = [18,36]
MODEL_HR18['STAGE2']['BLOCK']        = 'BASIC'
MODEL_HR18['STAGE2']['NUM_MODULES']  = 1
MODEL_HR18['STAGE2']['NUM_BRANCHES'] = 2
MODEL_HR18['STAGE2']['NUM_BLOCKS']   = [4,4]
MODEL_HR18['STAGE2']['FUSE_METHOD']  = 'SUM'

MODEL_HR18['STAGE3'] = {}
MODEL_HR18['STAGE3']['NUM_CHANNELS'] = [18,36,72]
MODEL_HR18['STAGE3']['BLOCK']        = 'BASIC'
MODEL_HR18['STAGE3']['NUM_MODULES']  = 4
MODEL_HR18['STAGE3']['NUM_BRANCHES'] = 3
MODEL_HR18['STAGE3']['NUM_BLOCKS']   = [4,4,4]
MODEL_HR18['STAGE3']['FUSE_METHOD']  = 'SUM'

MODEL_HR18['STAGE4'] = {}
MODEL_HR18['STAGE4']['NUM_CHANNELS'] = [18,36,72,144]
MODEL_HR18['STAGE4']['BLOCK']        = 'BASIC'
MODEL_HR18['STAGE4']['NUM_MODULES']  = 3
MODEL_HR18['STAGE4']['NUM_BRANCHES'] = 4
MODEL_HR18['STAGE4']['NUM_BLOCKS']   = [4,4,4,4]
MODEL_HR18['STAGE4']['FUSE_METHOD']  = 'SUM'

MODEL_HR18['FINAL_CONV_KERNEL']      = 1

###########################################################
MODEL_HR32 = {}
MODEL_HR32['STAGE2'] = {}
MODEL_HR32['STAGE2']['NUM_CHANNELS'] = [32,64]
MODEL_HR32['STAGE2']['BLOCK']        = 'BASIC'
MODEL_HR32['STAGE2']['NUM_MODULES']  = 1
MODEL_HR32['STAGE2']['NUM_BRANCHES'] = 2
MODEL_HR32['STAGE2']['NUM_BLOCKS']   = [4,4]
MODEL_HR32['STAGE2']['FUSE_METHOD']  = 'SUM'

MODEL_HR32['STAGE3'] = {}
MODEL_HR32['STAGE3']['NUM_CHANNELS'] = [32,64,128]
MODEL_HR32['STAGE3']['BLOCK']        = 'BASIC'
MODEL_HR32['STAGE3']['NUM_MODULES']  = 4
MODEL_HR32['STAGE3']['NUM_BRANCHES'] = 3
MODEL_HR32['STAGE3']['NUM_BLOCKS']   = [4,4,4]
MODEL_HR32['STAGE3']['FUSE_METHOD']  = 'SUM'

MODEL_HR32['STAGE4'] = {}
MODEL_HR32['STAGE4']['NUM_CHANNELS'] = [32,64,128,256]
MODEL_HR32['STAGE4']['BLOCK']        = 'BASIC'
MODEL_HR32['STAGE4']['NUM_MODULES']  = 3
MODEL_HR32['STAGE4']['NUM_BRANCHES'] = 4
MODEL_HR32['STAGE4']['NUM_BLOCKS']   = [4,4,4,4]
MODEL_HR32['STAGE4']['FUSE_METHOD']  = 'SUM'

MODEL_HR32['FINAL_CONV_KERNEL']      = 1

HRNet_Dict = {
    'HR18': MODEL_HR18,
    'HR32': MODEL_HR32,
    'FPHR18': MODEL_HR18,
    'FPHR32': MODEL_HR32
}

def Get_model(cfg):
    if(cfg.Model_Set['Model_name'] == 'DBFace'):
        model = DBFace()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))
        
    if(cfg.Model_Set['Model_name'] == 'HR18'):
        hr_cfg = HRNet_Dict[cfg.Model_Set['Model_name']]
        model  = HighResolutionNet(hr_cfg)
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'HR32'):
        hr_cfg = HRNet_Dict[cfg.Model_Set['Model_name']]
        model  = HighResolutionNet(hr_cfg)
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'FPHR18'):
        hr_cfg = HRNet_Dict[cfg.Model_Set['Model_name']]
        model  = FPHighResolutionNet(hr_cfg)
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'FPHR32'):
        hr_cfg = HRNet_Dict[cfg.Model_Set['Model_name']]
        model  = FPHighResolutionNet(hr_cfg)
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'TResnetM'):
        model = TResnetM()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'TResnetL'):
        model = TResnetL()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))
        
    if(cfg.Model_Set['Model_name'] == 'TResnetXL'):
        model = TResnetXL()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'DLA_34'):
        model = get_pose_net('DLA_34')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'DLA_46C'):
        model = get_pose_net('DLA_46C')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'DLA_46XC'):
        model = get_pose_net('DLA_46XC')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'DLA_60XC'):
        model = get_pose_net('DLA_60XC')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'DLA_60'):
        model = get_pose_net('DLA_60')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'ResNet_Dcn18'):
        model = get_pose_net('ResNet_Dcn18')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'ResNet_Dcn34'):
        model = get_pose_net('ResNet_Dcn34')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'ResNet_Dcn50'):
        model = get_pose_net('ResNet_Dcn50')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))
        
    if(cfg.Model_Set['Model_name'] == 'ResNet_Dcn101'):
        model = get_pose_net('ResNet_Dcn101')
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))

    if(cfg.Model_Set['Model_name'] == 'YoloV5'):
        model = YoloV5()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))
        
    if(cfg.Model_Set['Model_name'] == 'YoloV5_ASFF'):
        model = YoloV5_ASFF()
        print('Model name = {}'.format(cfg.Model_Set['Model_name']))
        
    return model
