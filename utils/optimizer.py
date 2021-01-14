import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .warmuplr import WarmUpLR
from .util import separate_irse_bn_paras, separate_resnet_bn_paras
import sys

def Get_optimizer(args, net, iter_per_epoch):
    """ return given optimizer
    """
    optim_scheduler = {}
    
    if(not args.Optimizer['Split_BN']):
        if(args.Optimizer['Optim_Type'] == 'SGD'):
            optimizer = optim.SGD(list(net.parameters()), lr=args.Optimizer['Lr_Base'], weight_decay=1e-4)
        elif(args.Optimizer['Optim_Type'] == 'Momentum'):
            optimizer = optim.SGD(list(net.parameters()), lr=args.Optimizer['Lr_Base'], momentum=0.9, weight_decay=1e-4)
        elif(args.Optimizer['Optim_Type'] == 'Adam'):
            optimizer = optim.Adam(list(net.parameters()), lr=args.Optimizer['Lr_Base'], betas=(0.9, 0.99), weight_decay=1e-4)
        elif(args.Optimizer['Optim_Type'] == 'AdamW'):
            optimizer = optim.AdamW(list(net.parameters()), lr=args.Optimizer['Lr_Base'], betas=(0.9, 0.99), weight_decay=1e-4)
        else:
            print('the optimizer name you have entered is not supported yet')
            sys.exit()
    else:
        if(args.Model_Set['Model_name'].find("IR") >= 0):
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(net) 
        else:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(net)
            
        params = [{'params': backbone_paras_wo_bn + list(head.parameters()), 'weight_decay': 1e-4},{'params': backbone_paras_only_bn}]
                        
        if(args.Optimizer['Optim_Type'] == 'SGD'):
            optimizer = optim.SGD(params, lr=args.Optimizer['Lr_Base'])
        elif(args.Optimizer['Optim_Type'] == 'Momentum'):
            optimizer = optim.SGD(params, lr=args.Optimizer['Lr_Base'], momentum=0.9)
        elif(args.Optimizer['Optim_Type'] == 'Adam'):
            optimizer = optim.Adam(params, lr=args.Optimizer['Lr_Base'], betas=(0.9, 0.99))
        elif(args.Optimizer['Optim_Type'] == 'AdamW'):
            optimizer = optim.AdamW(params, lr=args.Optimizer['Lr_Base'], betas=(0.9, 0.99))
        else:
            print('the optimizer name you have entered is not supported yet')
            sys.exit()

    if(args.Optimizer['Sche_Type']   == "step"):
        scheduler = lr_scheduler.StepLR(optimizer, args.Optimizer['Step_size'], gamma=0.1)
    elif(args.Optimizer['Sche_Type'] == "multistep"):
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.Optimizer['Lr_Adjust'], gamma=0.1)
    elif(args.Optimizer['Sche_Type'] == "exponential"):
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.Optimizer['Step_gamma'])
    elif(args.Optimizer['Sche_Type'] == "constant"):
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif(args.Optimizer['Sche_Type'] == "cosine"):
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, (args.Optimizer['Max_epoch']-args.Optimizer['Warmup_epoch'])*iter_per_epoch, eta_min=1e-7)
    elif(args.Optimizer['Sche_Type'] == "CosineAnnealingWarmRestarts"):
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, (args.Optimizer['Max_epoch']-args.Optimizer['Warmup_epoch'])*iter_per_epoch, eta_min=1e-7)
    elif(args.Optimizer['Sche_Type'] == "ReduceLROnPlateau"):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, threshold=1e-4, eps=1e-6)
    else:
        print('the scheduler name you have entered is not supported yet')
        sys.exit()
        
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.Optimizer['Warmup_epoch'])

    optim_scheduler['optimizer']        = optimizer
    optim_scheduler['scheduler']        = scheduler
    optim_scheduler['warmup_scheduler'] = warmup_scheduler
    
    return optim_scheduler
    


