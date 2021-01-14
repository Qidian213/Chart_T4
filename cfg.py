import os
import json

Model_pretrained = {
                    'HR18':'/data/zzg/PreTrained/hrnetv2_w18_imagenet_pretrained.pth', 
                    'HR32':'/data/zzg/PreTrained/hrnet_w32-36af842e.pth', 
                    'FPHR18':'/data/zzg/PreTrained/hrnetv2_w18_imagenet_pretrained.pth', 
                    'FPHR32':'/data/zzg/PreTrained/hrnet_w32-36af842e.pth', 
                    'DBFace': '/data/zzg/PreTrained/dbface.pth',
                    'TResnetM':'/data/zzg/PreTrained/tresnet_m.pth',
                    'TResnetL':'/data/zzg/PreTrained/tresnet_l.pth',
                    'TResnetXL':'/data/zzg/PreTrained/tresnet_xl.pth',
                    'DLA_34': '/data/zzg/PreTrained/ctdet_coco_dla_2x.pth',
                    'DLA_46C': '',
                    'DLA_46XC': '',
                    'DLA_60XC': '',
                    'DLA_60': '',
                    'ResNet_Dcn18':'/data/zzg/PreTrained/ctdet_coco_resdcn18.pth',
                    'ResNet_Dcn34': '',
                    'ResNet_Dcn50': '', 
                    'ResNet_Dcn101': '',
                    'YoloV5': '',
                    'YoloV5_ASFF': ''
                    }

class Cfg_Opts(object):
    def __init__(self,):
    ### Data setting
        self.DataSet                    = {}
        self.DataSet['Train_dir']       = '/data/Dataset/Chart/Task4/'
        self.DataSet['Train_json']      = '/data/Dataset/Chart/Task4/PMCTrain_Test.json'
        self.DataSet['Val_dir']         = '/data/Dataset/Chart/Task4/'
        self.DataSet['Val_json']        = '/data/Dataset/Chart/Task4/PMCTrain_Test.json'
        self.DataSet['batch_size']      = 24
        self.DataSet['num_worker']      = 4
        self.DataSet['weight_sample']   = False
        self.DataSet['Class_map']       = {0:'point', 'point':0}

    ### Data augmentation
        self.Data_Aug                   = {}
        self.Data_Aug['AutoAug']        = {'Use': False, 'Probs': 0.5}
        self.Data_Aug['ColorJitter']    = {'Use': True,  'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0}
        self.Data_Aug['Rotation']       = {'Use': False, 'Angle': 15}
        self.Data_Aug['HorizontalFlip'] = {'Use': True,  'Probs': 0.5}
        self.Data_Aug['VerticalFlip']   = {'Use': False, 'Probs': 0.5}
        self.Data_Aug['Cutmix']         = {'Use': False, 'Probs': 0.5, 'alpha': 0.2}
        self.Data_Aug['Mixup']          = {'Use': False, 'Probs': 0.5, 'alpha': 0.2}
        self.Data_Aug['RandomErasing']  = {'Use': False, 'Probs': 0.5}
        self.Data_Aug['RGB_MEAN']       = [0.408, 0.447, 0.47]
        self.Data_Aug['RGB_STD']        = [0.289, 0.274, 0.278]
        self.Data_Aug['Size']           = [1024,1024]

    ### Test setting 
        self.Test_Setting               = {}
        self.Test_Setting['TTA']        = {'Use': False}
        
    ### Model Setting
        self.Model_Set                  = {}
        self.Model_Set['Model_name']    = 'DLA_34'
        self.Model_Set['Pre_Trained']   = '' #Model_pretrained[self.Model_Set['Model_name']] 
        self.Model_Set['Resume']        = None

    ### Optimizer Setting
        self.Optimizer                  = {}
        self.Optimizer['Optim_Type']    = 'Momentum'
        self.Optimizer['Sche_Type']     = 'multistep'
        self.Optimizer['Lr_Base']       = 0.001
        self.Optimizer['LrDecay_Mode']  = 'epoch'
        self.Optimizer['Lr_Adjust']     = [35,45,55,60]
        self.Optimizer['Step_size']     = 50
        self.Optimizer['Step_gamma']    = 0.9
        self.Optimizer['Warmup_epoch']  = 3
        self.Optimizer['Start_epoch']   = 0
        self.Optimizer['Max_epoch']     = 65
        self.Optimizer['Split_BN']      = False
        
    ### Loss Setting
        self.Loss_set                   = {}
        self.Loss_set['HeatMap_type']   = 'Focal'
        self.Loss_set['Reg_type']       = 'SmoothL1'
        self.Loss_set['Tag_type']       = 'SmoothL1'
        self.Loss_set['AE_type']        = 'AELoss'
        
    ### Work space setting
        self.Work_Space                 = {}
        self.Work_Space['Save_dir']     = './work_space'
        self.Work_Space['Log_Inter']    = 50
        self.Work_Space['Cuda_env']     = '0,1'

    ### work mode 
        self.Work_Mode                  = 'Train'
#        self.Work_Mode                  = 'Val'

    def List_Setting(self, logger):
        for name,value in vars(self).items():
            if(isinstance(value, dict)):
                logger.info(f"{name}\r")
                for key in value.keys():
                    logger.info(f"{key}={value[key]}\r")
            else:
                logger.info(f"{name}={value}\r")