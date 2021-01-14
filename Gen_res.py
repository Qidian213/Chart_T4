import os
import cv2
import json
import math
import torch
import numpy as np
import torch.nn.functional as F
from models import Get_model
from cfg import Cfg_Opts

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r
    
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
        
def detect(model, image, threshold=0.3):
    mean = [0.408, 0.447, 0.47]
    std  = [0.289, 0.274, 0.278]

    image = pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    torch_image = torch_image.cuda()

    out_dict = model(torch_image)
    hms      = out_dict['heatmap']
    offset   = out_dict['reg'].cpu().squeeze().data.numpy()

    Tscores, Tpoints = [], []
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
        tscores, tpoints = [], []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            px = (cx + offset[0,cy,cx])* stride
            py = (cy + offset[1,cy,cx])* stride
            tscores.append(score)
            tpoints.append([px, py])
        Tscores.append(tscores)
        Tpoints.append(tpoints)
    return Tscores, Tpoints

if __name__ == "__main__":
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/FPHR18_2020-10-03-17-42-06/Epoch_best.pth")
    
    Val_Json  = json.load(open('/data/Dataset/Chart/Chart/ATest.json','r'))
    Data_dir  = '/data/Dataset/Chart/Chart/'
    Save_dir  = 'results/'
    Json_dirs = ['Synthetic_Json_Task4', 'PMC_Json_Task4']
    Img_dirs  = ['Synthetic_Img_Task4', 'PMC_Img_Task4']
    Re_Axis   = ['horizontal_bar', 'horizontal_interval']
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
                boxes = json_data['task3']['input']['task2_output']['text_blocks']
                gt_xs = json_data['task4']['output']['axes']["x-axis"]
                gt_ys = json_data['task4']['output']['axes']["y-axis"]

                for id, role in enumerate(roles):
                    if(role['role'] == "tick_label"):
                        Need_match.append(boxes[id])
                        
                image = cv2.imread(img_path)
                h, w  = image.shape[:2]
                Tscores, Tpoints = detect(model, image)

                axes = {}
                axes['x-axis'] = []
                axes['y-axis'] = []
                
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
                        
                    if(ind == 1 and sub_dir in Re_Axis):
                        mcls = 1 - mcls
                        
                    if(mcls == 0):
                        if('Synthetic' in img_path):
                            axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                        else:
                            if(ind == 1 and sub_dir in Re_Axis):
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
                            if(ind == 1 and sub_dir in Re_Axis):
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
                    cv2.circle(image, (int(point['tick_pt']['x']), int(point['tick_pt']['y'])), 2, (255, 0, 0), -1)
                for point in gt_ys:
                    cv2.circle(image, (int(point['tick_pt']['x']), int(point['tick_pt']['y'])), 2, (255, 0, 0), -1)

                cv2.imwrite('outs/' + img_path.split('/')[-1],image)
                json.dump(json_data, open(Save_dir + json_dir + '/' + sub_dir + '/' + json_file, 'w'), indent=4)
