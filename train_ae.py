import os
import torch
import random
import numpy as np
from torch import nn
from tensorboardX  import SummaryWriter
from data   import Get_dataloader
from models import Get_model
from losses import Get_loss_function
from utils  import WarmUpLR, Get_optimizer, get_time_stamp, getLogger, AverageMeter
from cfg    import Cfg_Opts

def Setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class Mainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.Data_dict    = Get_dataloader(self.cfg)
        self.epoch_batchs = len(self.Data_dict['Train_loader'])

        self.model = Get_model(self.cfg)

        if(self.cfg.Model_Set['Pre_Trained'] != ''):
            self.model.load_param(self.cfg.Model_Set['Pre_Trained'])

        self.model.cuda()
        self.model = nn.DataParallel(self.model)

        self.optim_schedulers = Get_optimizer(self.cfg, self.model, self.epoch_batchs)
        self.Loss_meter       = Get_loss_function(self.cfg)
        
        self.writer           = SummaryWriter(self.cfg.Work_Space['Save_dir'])
        self.best_acc         = 0.0
        self.best_epoch       = 0
        self.best_step        = 0

        self.cfg.List_Setting(logger)

    def train(self):
        best_epoch = 0
        best_loss  = 100000.0
        for epoch in range(self.cfg.Optimizer['Start_epoch'], self.cfg.Optimizer['Max_epoch']):
            self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            if(val_loss < best_loss):
                best_epoch = epoch
                best_loss  = val_loss
                save_file  = os.path.join(self.cfg.Work_Space['Save_dir'], 'Epoch_best.pth')
                torch.save(self.model.state_dict(),save_file)

            save_file = os.path.join(self.cfg.Work_Space['Save_dir'], 'Epoch_ck.pth')
            torch.save(self.model.state_dict(),save_file)

            logger.info(f"Best_epoch: {best_epoch}, Val_loss: {best_loss:.5f}\r")

    def train_epoch(self, epoch):
        self.model.train()

        log_loss     = AverageMeter()
        log_hm_loss  = AverageMeter()
        log_reg_loss = AverageMeter()
        log_tag_loss = AverageMeter()
        for step,(images, heatmap_gt, heatmap_posweight, heatmap_offset, heatmap_off_mask, tag_offset, joints, num_objs) in enumerate(self.Data_dict['Train_loader']):
            batch_objs = sum(num_objs)
            if batch_objs == 0:
                batch_objs = 1
                
            images            = images.cuda()
            heatmap_gt        = heatmap_gt.cuda()
            heatmap_posweight = heatmap_posweight.cuda()
            heatmap_offset    = heatmap_offset.cuda()
            heatmap_off_mask  = heatmap_off_mask.cuda()
            tag_offset        = tag_offset.cuda() 

            out_dicts = self.model(images)
            hm_loss   = self.Loss_meter['heat_loss'](out_dicts['heatmap'], heatmap_gt, heatmap_posweight)/batch_objs
            reg_loss  = self.Loss_meter['reg_loss'](out_dicts['reg'], heatmap_offset, heatmap_off_mask)
            tag_loss  = self.Loss_meter['tag_loss'](out_dicts['tag'], tag_offset, heatmap_off_mask)
            loss      = hm_loss + reg_loss + tag_loss

            self.optim_schedulers['optimizer'].zero_grad()
            loss.backward()
           # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.optim_schedulers['optimizer'].step()

            if(self.cfg.Optimizer['LrDecay_Mode'] == 'step' or epoch < self.cfg.Optimizer['Warmup_epoch']):
                if(epoch < self.cfg.Optimizer['Warmup_epoch']):
                    self.optim_schedulers['warmup_scheduler'].step()
                else:
                    self.optim_schedulers['scheduler'].step()

            log_loss.update(loss.item(), batch_objs)
            log_hm_loss.update(hm_loss.item(), batch_objs)
            log_reg_loss.update(reg_loss.item(), batch_objs)
            log_tag_loss.update(tag_loss.item(), batch_objs)
            
            if(step % self.cfg.Work_Space['Log_Inter'] == 0):
                self.writer.add_scalar('Train_loss:', log_loss.avg, self.epoch_batchs*epoch + step//self.cfg.Work_Space['Log_Inter'])
                self.writer.add_scalar('Hm_loss:', log_hm_loss.avg, self.epoch_batchs*epoch + step//self.cfg.Work_Space['Log_Inter'])
                self.writer.add_scalar('Offset_loss:', log_reg_loss.avg, self.epoch_batchs*epoch + step//self.cfg.Work_Space['Log_Inter'])
                self.writer.add_scalar('Tag_loss:', log_tag_loss.avg, self.epoch_batchs*epoch + step//self.cfg.Work_Space['Log_Inter'])
                logger.info(f"iter: {step}/{self.epoch_batchs}/{epoch}, loss: {log_loss.avg:.5f}, hm_loss: {log_hm_loss.avg:.5f}, reg_loss: {log_reg_loss.avg:.5f}, tag_loss: {log_tag_loss.avg:.5f}, LR: {'%e'%self.optim_schedulers['optimizer'].param_groups[0]['lr']}\r")

        if(self.cfg.Optimizer['LrDecay_Mode'] == 'epoch'):
            self.optim_schedulers['scheduler'].step()
                
    def val_epoch(self, epoch):
        self.model.eval()
        
        log_loss = 0
        with torch.no_grad():
            for step,(images, heatmap_gt, heatmap_posweight, heatmap_offset, heatmap_off_mask, tag_offset, joints, num_objs) in enumerate(self.Data_dict['Val_loader']):
                batch_objs = sum(num_objs)
                if batch_objs == 0:
                    batch_objs = 1
                    
                images            = images.cuda()
                heatmap_gt        = heatmap_gt.cuda()
                heatmap_offset    = heatmap_offset.cuda()
                heatmap_posweight = heatmap_posweight.cuda()
                heatmap_off_mask  = heatmap_off_mask.cuda()
                tag_offset        = tag_offset.cuda() 

                out_dicts = self.model(images)
                hm_loss   = self.Loss_meter['heat_loss'](out_dicts['heatmap'], heatmap_gt, heatmap_posweight)/batch_objs
                reg_loss  = self.Loss_meter['reg_loss'](out_dicts['reg'], heatmap_offset, heatmap_off_mask)
                tag_loss  = self.Loss_meter['tag_loss'](out_dicts['tag'], tag_offset, heatmap_off_mask)
                log_loss  = log_loss + hm_loss.item() + reg_loss.item() + tag_loss.item()

        logger.info(f"Val: {epoch}, Val_loss: {log_loss:.5f}\r")
        self.writer.add_scalar(f'Val_loss: ', log_loss, epoch)

        return log_loss

if __name__ == '__main__':
    cfg = Cfg_Opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.Work_Space['Cuda_env']

    if(cfg.Work_Mode == 'Train'):
        cfg.Work_Space['Save_dir'] = os.path.join(cfg.Work_Space['Save_dir'], cfg.Model_Set['Model_name'] +'_'+get_time_stamp())
        
        if not os.path.exists(cfg.Work_Space['Save_dir']):
            os.makedirs(cfg.Work_Space['Save_dir'])

        logger = getLogger("Face" , cfg.Work_Space['Save_dir']+ '/' + cfg.Model_Set['Model_name'] +'.log')
        logger.info("startup... \r")

        mainer = Mainer(cfg)
        logger.info("Train mode ... \r")
        mainer.train()
        
    if(cfg.Work_Mode == 'Val'):
        mainer = Mainer(cfg)
        logger.info("Val mode ... \r")
        mainer.val_epoch(0,0)
