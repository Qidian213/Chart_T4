import os
import cv2
import json
import math
import torch
import numpy as np
import torch.nn.functional as F
from models import Get_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def dist_points(point_a, point_b):
    dist = ((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)**0.5
    return dist
    
def oks_nms(dscores, dpoints, img_path, or_h, or_w, THRESHOLD=0.025):
    diag = ((or_h ** 2) + (or_w ** 2)) ** 0.5
    dth  = THRESHOLD * diag
    
    Tscores, Tpoints = [], []
    for ind, (scores, points) in enumerate(zip(dscores, dpoints)):
        scores = np.array(scores)
        points  = np.array(points)

        sc_indexs = scores.argsort()[::-1]

        scores = scores[sc_indexs]
        points = points[sc_indexs]

        sc_keep = []
        point_keep = []

        flags = [0] * len(scores)
        for index, sc in enumerate(scores):
            if flags[index] != 0:
                continue

            sc_keep.append(scores[index])
            point_keep.append(points[index])

            for j in range(index + 1, len(scores)):
                if flags[j] == 0 and dist_points(points[index], points[j]) < dth:
                    flags[j] = 1

        point_keep = np.array(point_keep)
        if(ind ==0 and len(point_keep)>0):
            ys = point_keep[:, 1]
            if(np.var(ys) <1.5):
                y_mean = np.mean(ys)
                new_points = []
                for point in point_keep:
                    new_points.append([point[0], y_mean])
                point_keep = np.array(new_points)

        if(ind ==1 and len(point_keep)>0):
            xs = point_keep[:, 0]
            if(np.var(xs) <1.5):
                x_mean = np.mean(xs)
                new_points = []
                for point in point_keep:
                    new_points.append([x_mean, point[1]])
                point_keep = np.array(new_points)
        point_keep = list(point_keep)
        
        Tscores.append(sc_keep)
        Tpoints.append(point_keep)
    return Tscores, Tpoints

def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image
        
def detect(models, image, img_path, threshold=0.3):
    mean = [0.408, 0.447, 0.47]
    std  = [0.289, 0.274, 0.278]
    or_h, or_w = image.shape[:2]
    
    image = pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)  ## C,H,W

    torch_image = torch.from_numpy(image)[None]
    torch_image = torch_image.cuda()

    Tscores, Tpoints = [[],[]], [[],[]]
    for model in models:
        out_dict = model(torch_image)
        hms      = out_dict['heatmap']
        offset   = out_dict['reg'].cpu().squeeze().data.numpy()
        # hms      = F.interpolate(hms, size=(image.shape[1]//2, image.shape[2]//2),mode='bilinear',align_corners=False)
        # offset   = F.interpolate(offset, size=(image.shape[1]//2, image.shape[2]//2),mode='bilinear',align_corners=False)
        # offset   = offset.cpu().squeeze().data.numpy()

        for ind in range(2):
            hm      = hms[0:1, ind:ind+1]
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]
            
            scores  = scores.squeeze()
            indices = indices.squeeze()
            ys      = list((indices / hm_width).int().data.numpy())
            xs      = list((indices % hm_width).int().data.numpy())
            scores  = list(scores.data.numpy())

            stride = 4
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                px = (cx + offset[0,cy,cx])* stride
                py = (cy + offset[1,cy,cx])* stride
                
                Tscores[ind].append(score)
                Tpoints[ind].append([px, py])

    return oks_nms(Tscores, Tpoints, img_path, or_h, or_w)

def detectFlip(models, image, threshold=0.3):
    mean = [0.408, 0.447, 0.47]
    std  = [0.289, 0.274, 0.278]
    or_h, or_w = image.shape[:2]
    
    image = pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    pd_h, pd_w = image.shape[:2]
    image_or   = image
    image_flip = cv2.flip(image, 1)
    image_or   = image_or.transpose(2, 0, 1)  ## C,H,W
    image_flip = image_flip.transpose(2, 0, 1)  ## C,H,W

    Tscores, Tpoints = [[],[]], [[],[]]
    
    torch_image = torch.from_numpy(image_or)[None]
    torch_image = torch_image.cuda()
    for model in models:
        out_dict = model(torch_image)
        hms      = out_dict['heatmap']
        offset   = out_dict['reg'].cpu().squeeze().data.numpy()
        
        for ind in range(2):
            hm      = hms[0:1, ind:ind+1]
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]
            
            scores  = scores.squeeze()
            indices = indices.squeeze()
            ys      = list((indices / hm_width).int().data.numpy())
            xs      = list((indices % hm_width).int().data.numpy())
            scores  = list(scores.data.numpy())

            stride = 4
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                px = (cx + offset[0,cy,cx])* stride
                py = (cy + offset[1,cy,cx])* stride
                
                Tscores[ind].append(score)
                Tpoints[ind].append([px, py])

    torch_image = torch.from_numpy(image_flip)[None]
    torch_image = torch_image.cuda()
    for model in models:
        out_dict = model(torch_image)
        hms      = out_dict['heatmap']
        offset   = out_dict['reg'].cpu().squeeze().data.numpy()
        
        for ind in range(2):
            hm      = hms[0:1, ind:ind+1]
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]
            
            scores  = scores.squeeze()
            indices = indices.squeeze()
            ys      = list((indices / hm_width).int().data.numpy())
            xs      = list((indices % hm_width).int().data.numpy())
            scores  = list(scores.data.numpy())

            stride = 4
            for cx, cy, score in zip(xs, ys, scores):
                if score < threshold:
                    break

                px = (cx + offset[0,cy,cx])* stride
                py = (cy + offset[1,cy,cx])* stride
                
                Tscores[ind].append(score)
                Tpoints[ind].append([pd_w - px, py])
                
    return oks_nms(Tscores, Tpoints, or_h, or_w)
    
class Cfg_Opts1(object):
    def __init__(self,):
        self.Model_Set                  = {}
        self.Model_Set['Model_name']    = 'DLA_34'

class Cfg_Opts2(object):
    def __init__(self,):
        self.Model_Set                  = {}
        self.Model_Set['Model_name']    = 'DLA_34'
        
if __name__ == "__main__":
    cfg_1   = Cfg_Opts1()
    model_1 = Get_model(cfg_1)
    model_1.eval()
    model_1.cuda()
    model_1.load_param("work_space/DLA_34_2020-10-09-20-13-28/Epoch_best.pth")

    cfg_2   = Cfg_Opts2()
    model_2 = Get_model(cfg_2)
    model_2.eval()
    model_2.cuda()
    model_2.load_param("work_space/DLA_34_2020-10-26-11-24-01/Epoch_best.pth")

    Val_Json  = json.load(open('/data/Dataset/Chart/Task4/TTest.json','r'))
    Data_dir  = '/data/Dataset/Chart/Task4/'
    Save_dir  = 'results/'
    Json_dirs = ['Synthetic_Json_Task4', 'PMC_Json_Task4']
    Img_dirs  = ['Synthetic_Img_Task4', 'PMC_Img_Task4']
    Re_Axis   = ['Horizontal box','Grouped horizontal bar','Stacked horizontal bar', 'horizontal bar', 'horizontal interval']
    for ind, json_dir in enumerate(Json_dirs):
        Json_subs  = os.listdir(Data_dir + json_dir)
        for sub_dir in Json_subs:
            if not os.path.exists(Save_dir + json_dir + '/' + sub_dir):
                os.makedirs(Save_dir + json_dir + '/' + sub_dir)

            Json_files = os.listdir(Data_dir + json_dir + '/' + sub_dir)
            for json_file in Json_files:
                json_path = Data_dir + json_dir + '/' + sub_dir + '/' + json_file

                if(ind == 0):
                    img_path  = Data_dir + Img_dirs[ind] + '/' + sub_dir + '/' + json_file.replace('json', 'png')
                else:
                    img_path  = Data_dir + Img_dirs[ind] + '/' + sub_dir + '/' + json_file.replace('json', 'jpg')
                    
                if(ind == 0 and (Img_dirs[ind] + '/' + sub_dir + '/' + json_file.replace('json', 'png') not in Val_Json.keys())):
                    continue
                if(ind == 1 and (Img_dirs[ind] + '/' + sub_dir + '/' + json_file.replace('json', 'jpg') not in Val_Json.keys())):
                    continue
                    
                json_data = json.load(open(json_path,'r'))
                Need_match = []
                roles = json_data['task3']['output']['text_roles']
                ctype = json_data['task4']['input']['task1_output']['chart_type']
                boxes = json_data['task4']['input']['task2_output']['text_blocks']
                gt_xs = json_data['task4']['output']['axes']["x-axis"]
                gt_ys = json_data['task4']['output']['axes']["y-axis"]

                for id, role in enumerate(roles):
                    if(role['role'] == "tick_label"):
                        Need_match.append(boxes[id])
                        
                image = cv2.imread(img_path)
                or_h, or_w  = image.shape[:2]
                Tscores, Tpoints = detect([model_1, model_2], image, img_path)

                axes = {}
                axes['x-axis'] = []
                axes['y-axis'] = []
                axes['x-axis-2'] = []
                axes['y-axis-2'] = []
                
                for box in Need_match:
                    id  = box['id']
                    if(ind == 0):
                        bb  = box['bb']
                    else:
                        bb  = box['polygon']
                    xp  = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0
                    yp  = (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0
 
                    mcls  = 0
                    mdist = 10000.0
                    mpx   = 0
                    mpy   = 0
                    for point in Tpoints[0]:
                        dist = ((point[0] - xp)**2+(point[1] - yp)**2)
                        if(dist < mdist):
                            mdist = dist
                            mcls  = 0
                            mpx   = point[0]
                            mpy   = point[1]
                    for point in Tpoints[1]:
                        dist = ((point[0] - xp)**2+(point[1] - yp)**2)
                        if(dist < mdist):
                            mdist = dist
                            mcls  = 1
                            mpx   = point[0]
                            mpy   = point[1]
                        
                    if(ctype in Re_Axis):
                        mcls = 1 - mcls
                        
                    if(mcls == 0):
                        if('Synthetic' in img_path):
                            axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                        else:
                            if(ctype in Re_Axis):
                                if(mpx <= or_w/2):
                                    axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                                else:
                                    axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                            else:
                                if(mpy >= or_h/2):
                                    axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                                else:
                                    axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                        cv2.circle(image, (int(mpx), int(mpy)), 2, (0, 255, 0), -1)
                    else:
                        if('Synthetic' in img_path):
                            axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                        else:
                            if(ctype in Re_Axis):
                                if(mpy >= or_h/2):
                                    axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                                else:
                                    axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                            else:
                                if(mpx <= or_w/2):
                                    axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                                else:
                                    axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                                    
                        cv2.circle(image, (int(mpx), int(mpy)), 2, (0, 0, 255), -1)
                        
                    cv2.line(image, (int(xp), int(yp)), (int(mpx), int(mpy)), [0,125,255],1)
                        
                json_data['task4']["output"]["axes"] = axes
                
                for point in gt_xs:
                    cv2.circle(image, (int(point['tick_pt']['x']), int(point['tick_pt']['y'])), 2, (0, 255, 255), -1)
                for point in gt_ys:
                    cv2.circle(image, (int(point['tick_pt']['x']), int(point['tick_pt']['y'])), 2, (255, 0, 0), -1)

                cv2.imwrite('outs/' + img_path.split('/')[-1],image)
                json.dump(json_data, open(Save_dir + json_dir + '/' + sub_dir + '/' + json_file, 'w'), indent=4)
