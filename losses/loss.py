import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def forward(self, pred, gt, pos_weights):
        loss = 0

        if 0:
            hm_pre = pred[0,0].cpu().data.numpy()*255
            hm_gt  = gt[0,0].cpu().data.numpy() * 255
            cv2.imwrite("pre.jpg", hm_pre)
            cv2.imwrite("gt.jpg", hm_gt) 
            print(pred.size(), gt.size(), pos_weights.size())

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_weights
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        loss -= (pos_loss + neg_loss)
            
        return loss

class SmoothL1LossNoAvg(nn.Module):
    def forward(self, x, t, weight, sigma=1):
        sigma2 = sigma ** 2

        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()

class SmoothL1Loss(nn.Module):
    def forward(self, x, t, weight, sigma=1):

        avg_factor = torch.sum(weight)
        if avg_factor == 0:
            print("avg landmark is zero")
            return torch.tensor(0.0)

        sigma2 = sigma ** 2

        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
             (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum() / avg_factor

class RegL1Loss(nn.Module):
    def forward(self, x, t, weight, sigma=1):
    
        sigma2 = sigma ** 2

        diff = (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.mean()

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()

        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, weight, sigma=1):
        
        avg_factor = torch.sum(weight)

        if avg_factor == 0:
            print("WingLoss avg is zero")
            return torch.tensor(0.0)

        diff = weight * (x - t)
        abs_diff = diff.abs()

        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum() # / avg_factor

class LandmarkLossDel(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.shift = None
        self.smoothL1 = SmoothL1Loss()

    def forward(self, pred, gt, weight):

        h, w = pred.shape[2:]
        avg_factor = torch.sum(weight)

        if avg_factor == 0:
            print("avg landmark is zero")
            return torch.tensor(0.0)

        if self.shift is None:
            x = torch.arange(0, w, device=pred.device)
            y = torch.arange(0, h, device=pred.device)
            shift_y, shift_x = torch.meshgrid(y, x)
            self.shift = torch.stack((shift_x, shift_y), dim=0).float()   # 2, h, w

        pred_point = torch.cat((
            self.shift[0, :, :] + pred[:, [0, 1, 2, 3, 4]],
            self.shift[1, :, :] + pred[:, [5, 6, 7, 8, 9]]
        ), dim=1)  # b, 4, h, w
        return self.smoothL1(pred_point, gt, weight) / avg_factor

class GIoULoss(nn.Module):
    def __init__(self):
        super(GIoULoss, self).__init__()
        self.shift = None

    def forward(self, pred, gt, weight):
        # pred is   b, 4, h, w
        # gt is     b, 4, h, w
        # mask is   b, 1, h, w
        # 4 channel is x, y, r, b - cx
        loss = 0

        pred_type        = pred
        gt_type          = gt
        pos_weights_type = weight
        
        h, w = pred_type.shape[2:]
        pos_weights_type = pos_weights_type.view(-1, h, w)
        mask = pos_weights_type > 0
        pos_weights_type = pos_weights_type[mask]
        avg_factor = torch.sum(pos_weights_type)

        if avg_factor == 0:
            print("avg is zero")
            return torch.tensor(0.0)

        if self.shift is None:
            x = torch.arange(0, w, device=pred_type.device)
            y = torch.arange(0, h, device=pred_type.device)
            shift_y, shift_x = torch.meshgrid(y, x)
            self.shift = torch.stack((shift_x, shift_y), dim=0).float()   # 2, h, w
        
        pred_boxes = torch.cat((
            self.shift - pred_type[:, [0, 1]],
            self.shift + pred_type[:, [2, 3]]
        ), dim=1).permute(0, 2, 3, 1)  # b, 4, h, w   to   b, h, w, 4

        gt_boxes = gt_type.permute(0, 2, 3, 1)

        pred_boxes = pred_boxes[mask].view(-1, 4)
        gt_boxes = gt_boxes[mask].view(-1, 4)

        # max x, max y
        lt = torch.max(pred_boxes[:, :2], gt_boxes[:, :2])

        # min r, min b
        rb = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:])
        wh = (rb - lt + 1).clamp(0) # n, 2

        enclose_lt = torch.min(pred_boxes[:, :2], gt_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], gt_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt + 1).clamp(0)  # n, 2
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        overlap = wh[:, 0] * wh[:, 1]

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        ious = overlap / (pred_area + gt_area - overlap)

        u = pred_area + gt_area - overlap
        gious = ious - (enclose_area - u) / enclose_area
        iou_distance = 1 - gious
        
        loss += torch.sum(iou_distance * pos_weights_type) / avg_factor
        
        return loss
