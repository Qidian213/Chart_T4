import os
import cv2
import math
import numpy as np
import json
import random
import torch
import torchvision.transforms as T
import pandas as pd
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from .augment import augmentWithColorJittering

def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low

def gaussian_truncate_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
    
def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h_radius = math.ceil(h_radius)
    w_radius = math.ceil(w_radius)
    h, w     = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x  = w / 6
    sigma_y  = h / 6
    gaussian = gaussian_truncate_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    x, y     = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right   = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom   = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return gaussian

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h    = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# return shape: (2, height, width)
def generate_constant_paf(shape, joint_from, joint_to, paf_width=2):
    joint_distance = np.linalg.norm(joint_to - joint_from)
    unit_vector = (joint_to - joint_from) / joint_distance 
    rad = np.pi / 2
    
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    vertical_unit_vector = np.dot(rot_matrix, unit_vector) 
    
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() 
    
    horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
    horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)

    vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
    vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width 

    paf_flag = horizontal_paf_flag & vertical_paf_flag

    constant_paf = np.stack((paf_flag, paf_flag)) #* np.broadcast_to(unit_vector, shape[:-1] + [2,]).transpose(2, 0, 1)

    return constant_paf
    
class LDataset(Dataset):
    def __init__(self, data_dir, data_json, cfg, mode):
        self.cfg               = cfg
        self.mode              = mode 
        self.data_dir          = data_dir
        self.Class_map         = cfg.DataSet['Class_map']
        self.num_classes       = len(list(self.Class_map.keys()))/2
        self.rgb_mean          = cfg.Data_Aug['RGB_MEAN']
        self.rgb_std           = cfg.Data_Aug['RGB_STD']
        self.width,self.height = cfg.Data_Aug['Size']
        
        self.posweight_radius  = 2
        self.stride            = 4
        self.radius            = 2
        self.fm_width          = self.width // self.stride
        self.fm_height         = self.height // self.stride
        
        self.json_data = json.load(open(data_json,'r'))
        self.data_set  = []
        self.x_axis    = []
        self.y_axis    = []

        self.pre_read()
        
        self.train_transform = A.Compose(
                                    [
                                        # A.OneOf([
                                            # A.IAAAdditiveGaussianNoise(),
                                            # A.GaussNoise(),
                                        # ], p=0.3),
                                        # A.OneOf([
                                            # A.MotionBlur(p=.2),
                                            # A.MedianBlur(blur_limit=3, p=0.1),
                                            # A.Blur(blur_limit=3, p=0.1),
                                        # ], p=0.3),
                                        # A.OneOf([
                                            # A.CLAHE(clip_limit=2),
                                            # A.IAASharpen(),
                                            # A.IAAEmboss(),
                                            # A.RandomBrightnessContrast(),            
                                        # ], p=0.3),
                                        # A.ChannelShuffle(p=0.5),
                                        # A.HueSaturationValue(p=0.5),
                                        # A.InvertImg(p=0.2),
                                        # A.ToGray(p=0.2),
                                        A.Blur(blur_limit=3, p=0.5),
                                        A.ChannelShuffle(p=0.5),
                                        A.HueSaturationValue(p=0.5),
                                        A.InvertImg(p=0.2),
                                        A.ToGray(p=0.2),
                                    ]
                                )
        
    def pre_read(self):
        for key in self.json_data.keys():
            data_dict = self.json_data[key]
            x_axi     = data_dict['x-axis']
            y_axi     = data_dict['y-axis']
            self.data_set.append(key)
            self.x_axis.append(x_axi)
            self.y_axis.append(y_axi)
            
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        info_item = self.data_set[index]
        File_path = self.data_dir  + info_item
        
        x_points = self.x_axis[index]
        y_points = self.y_axis[index]
        # x_points = np.array(x_points)
        # y_points = np.array(y_points)

        image    = cv2.imread(File_path)
        o_h, o_w = image.shape[0:2]

###
        tx_points = x_points
        ty_points = y_points
        flip_flag = False
        # if(randrf(0.0, 1.0) >0.5):
            # flip_flag = True
            # image = cv2.flip(image, 0)
            # image = cv2.flip(image, 1)
            # image = cv2.transpose(image)
            # tx_points = []
            # ty_points = []
            # for point in x_points:
                # tx_points.append([o_h-point[1], o_w-point[0]])
            # for point in y_points:
                # ty_points.append([o_h-point[1], o_w-point[0]])
###

        o_h, o_w = image.shape[0:2]
        re_scale = randrf(0.5, 1.2)
        r_h, r_w = o_h*re_scale, o_w*re_scale

        if(r_h > r_w and r_h > self.height):
            scale = r_h/self.height
            r_h   = self.height
            r_w   = r_w/scale
            re_scale = re_scale/scale
            
        if(r_w > r_h and r_w > self.width):
            scale = r_w/self.width
            r_w   = self.width
            r_h   = r_h/scale
            re_scale = re_scale/scale
        
        image = cv2.resize(image, (int(r_w), int(r_h)))
        deta_x, deta_y = 0, 0 #int(512-r_w/2), int(512-r_h/2)

        newImage = np.zeros((self.height, self.width, 3), np.uint8)
        newImage[deta_y:image.shape[0]+deta_y, deta_x:image.shape[1]+deta_x, :] = image

        newImage = self.train_transform(image=newImage)["image"]
        
        newImage = ((newImage / 255.0 - self.rgb_mean) / self.rgb_std).astype(np.float32)
        newImage = newImage.transpose(2, 0, 1)

        heatmap_gt        = np.zeros((2, self.fm_height, self.fm_width), np.float32)
        heatmap_posweight = np.zeros((2, self.fm_height, self.fm_width), np.float32)
        tag_offset        = np.zeros((2, self.fm_height, self.fm_width), np.float32)
        heatmap_offset    = np.zeros((2, self.fm_height, self.fm_width), np.float32)
        heatmap_off_mask  = np.zeros((1, self.fm_height, self.fm_width), np.float32)

        for point in tx_points:
            bb = point[2]
            if('height' in list(bb.keys())):
                bx, by  = bb['x0'] + bb['width']/2.0, bb['y0'] + bb['height']/2.0
            else:
                bx, by  = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0, (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0
                
            bx, by  = (re_scale*bx+deta_x)/self.stride, (re_scale*by+deta_y)/self.stride
            
            cx, cy  = (re_scale*point[0]+deta_x)/self.stride, (re_scale*point[1]+deta_y)/self.stride
            gaussian_map = draw_truncate_gaussian(heatmap_gt[0, :, :], (cx, cy), self.radius, self.radius)
            draw_gaussian(heatmap_posweight[0, :, :], (cx, cy), self.posweight_radius)
            
            int_cx, int_cy = int(cx), int(cy)
            if(int_cx>=self.fm_width or int_cy>=self.fm_height):
                continue
            heatmap_off_mask[:, int_cy, int_cx] = 1
            heatmap_offset[0, int_cy, int_cx] = cx - int_cx
            heatmap_offset[1, int_cy, int_cx] = cy - int_cy
            tag_offset[0, int_cy, int_cx]     = bx - int_cx
            tag_offset[1, int_cy, int_cx]     = by - int_cy
            # paf = generate_constant_paf([self.fm_height,self.fm_width,3], np.array([cx,cy]), np.array([bx,by]))
            # tag_offset += paf

        for point in ty_points:
            bb = point[2]
            if('height' in list(bb.keys())):
                bx, by  = bb['x0'] + bb['width']/2.0, bb['y0'] + bb['height']/2.0
            else:
                bx, by  = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0, (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0
                
            bx, by  = (re_scale*bx+deta_x)/self.stride, (re_scale*by+deta_y)/self.stride
            
            cx, cy  = (re_scale*point[0]+deta_x)/self.stride, (re_scale*point[1]+deta_y)/self.stride
            gaussian_map = draw_truncate_gaussian(heatmap_gt[1, :, :], (cx, cy), self.radius, self.radius)
            draw_gaussian(heatmap_posweight[1, :, :], (cx, cy), self.posweight_radius)

            int_cx, int_cy = int(cx), int(cy)
            if(int_cx>=self.fm_width or int_cy>=self.fm_height):
                continue
            heatmap_off_mask[:, int_cy, int_cx] = 1
            heatmap_offset[0, int_cy, int_cx] = cx - int_cx
            heatmap_offset[1, int_cy, int_cx] = cy - int_cy
            tag_offset[0, int_cy, int_cx]     = bx - int_cx
            tag_offset[1, int_cy, int_cx]     = by - int_cy
            # paf = generate_constant_paf([self.fm_height,self.fm_width,3], np.array([cx,cy]), np.array([bx,by]))
            # tag_offset += paf
            
        if 0:
            base_name = info_item.split('/')[-1]
            cv2.imwrite(f"imgs/{base_name}_gx.jpg", heatmap_gt[0]*255)
            cv2.imwrite(f"imgs/{base_name}_gy.jpg", heatmap_gt[1]*255)
            cv2.imwrite(f"imgs/{base_name}_pafx.jpg", tag_offset[0]*255)
            cv2.imwrite(f"imgs/{base_name}_pafy.jpg", tag_offset[1]*255)
            wimage = newImage.transpose(1, 2, 0)
            wimage = (wimage*self.rgb_std+self.rgb_mean)*255
            wimage = wimage.astype(np.int8)
            for point in tx_points:
                cv2.circle(wimage, (int(re_scale*point[0]+deta_x), int(re_scale*point[1]+deta_y)), 5, (0, 0, 255), -1)
            for point in ty_points:
                cv2.circle(wimage, (int(re_scale*point[0]+deta_x), int(re_scale*point[1]+deta_y)), 5, (0, 0, 255), -1)
            cv2.imwrite(f"imgs/{base_name}_or.jpg", wimage)

        return torch.tensor(newImage), heatmap_gt, heatmap_posweight, heatmap_offset, heatmap_off_mask, tag_offset ,len(x_points) + len(y_points)

def Get_dataloader(cfg):
    train_dir  = cfg.DataSet['Train_dir']
    train_json = cfg.DataSet['Train_json']
    val_dir    = cfg.DataSet['Val_dir']
    val_json   = cfg.DataSet['Val_json']

    train_data = LDataset(train_dir, train_json, cfg, 'train')
    val_data   = LDataset(val_dir, val_json, cfg, 'val')

    train_loader = DataLoader(dataset=train_data, batch_size=cfg.DataSet['batch_size'], shuffle=True, num_workers =cfg.DataSet['num_worker'], pin_memory=True)
    val_loader   = DataLoader(dataset=val_data, batch_size=cfg.DataSet['batch_size'], shuffle=False, num_workers =cfg.DataSet['num_worker'], pin_memory=True)

    Data_dict = {}
    Data_dict['Train_loader'] = train_loader
    Data_dict['Val_loader']   = val_loader

    return Data_dict
