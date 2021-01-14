import os
import torch
from mobilenet import DBFace
from hrnet import HighResolutionNet, FPHighResolutionNet
from tresnet import TResnetM, TResnetL, TResnetXL
from dcn import get_pose_net
from yolov5 import YoloV5, YoloV5_ASFF
from resnest import resnest50, resnest101, resnest200, resnest269

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    'HR32': MODEL_HR32
}

model  = resnest50() ## YoloV5_ASFF() #get_pose_net('DLA_34')
#model.load_param('/data/zzg/PreTrained/tresnet_m.pth')
model.eval()
model.cuda()

with torch.no_grad():
    img = torch.rand(1,3,1024,1024)
    img = img.cuda()
    
    result = model(img)
    print(result[0].size())
    print(result[1].size())
    print(result[2].size())
    # print(result['heatmap'].size())
    # print(result['reg'].size())
    # print(result['tag'].size())
    