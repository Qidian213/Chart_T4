import sys
import torch 
import torch.nn as nn
from .loss import * 
from .ae_loss import AELoss

def Get_loss_function(cfgs):
    """ return given loss function
    """
    loss_fun = {}
    
    if(cfgs.Loss_set['HeatMap_type'] == 'Softmax'):
        function = nn.CrossEntropyLoss()
        loss_fun['heat_loss'] = function
    elif(cfgs.Loss_set['HeatMap_type'] =='Focal'):
        function = FocalLoss()
        loss_fun['heat_loss'] = function
    else:
        print('the function name you have entered is not supported yet')
        sys.exit()

    if(cfgs.Loss_set['Reg_type'] == 'SmoothL1'):
        function = SmoothL1Loss()
        loss_fun['reg_loss'] = function
    else:
        print('the function name you have entered is not supported yet')
        sys.exit()

    if(cfgs.Loss_set['Tag_type'] == 'SmoothL1'):
        function = SmoothL1Loss()
        loss_fun['tag_loss'] = function
    else:
        print('the function name you have entered is not supported yet')
        sys.exit()

    if(cfgs.Loss_set['AE_type'] == 'AELoss'):
        function = AELoss('max')
        loss_fun['ae_loss'] = function
    else:
        print('the function name you have entered is not supported yet')
        sys.exit()
        
    return loss_fun