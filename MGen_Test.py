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
    # Stacked vertical bar 62
    # Horizontal box 69
    # Stacked horizontal bar 65
    # Vertical box 65
    # Grouped vertical bar 59
    # Line 66
    # Grouped horizontal bar 66
    # Scatter 76

    # line 264
    # scatter 95
    # vertical bar 225
    # vertical box 71
    # horizontal bar 71

    cfg_1   = Cfg_Opts1()
    model_1 = Get_model(cfg_1)
    model_1.eval()
    model_1.cuda()
    model_1.load_param("work_space/DLA_34_2020-10-09-20-13-28/Epoch_best.pth")

    cfg_2   = Cfg_Opts2()
    model_2 = Get_model(cfg_2)
    model_2.eval()
    model_2.cuda()
    model_2.load_param("work_space/DLA_34_2020-10-27-09-36-04/Epoch_best.pth")

#### PMC
    PMC_Test_Img_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/images/'
    PMC_Test_Json_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/annotations/'
    PMC_Test_Cls_Dir  = '/data/Dataset/Chart/Test_Task3_output/PMC_pred_json/'
    PMC_Test_Box_Dir  = '/data/Dataset/Chart/Test_graph_legend_json/'
    PMC_Test_Out_Dir  = '/data/zzg/ICPR_Chart/AE_Chart/results/PMC_Test/'
    
    Ctype_Dict = {}
    Re_PMC = ['horizontal bar']
    PMC_Test_Json_Files = os.listdir(PMC_Test_Json_Dir)
    for json_file in PMC_Test_Json_Files:
        print(json_file)
        Img_path  = PMC_Test_Img_Dir + json_file.replace('json', 'jpg')
        Json_path = PMC_Test_Json_Dir + json_file
        Cls_Path  = PMC_Test_Cls_Dir + json_file
        Box_Path  = PMC_Test_Box_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))
        Cls_data  = json.load(open(Cls_Path,'r'))
        Box_data  = json.load(open(Box_Path,'r'))
        
        Task1_ctype = Json_data['task4']['input']['task1_output']['chart_type']  ###  ctype
        Task2_boxes = Json_data['task4']['input']['task2_output']['text_blocks']
        Task3_roles = Cls_data['task3']['output']['text_roles']

        Need_match = []
        if('graph_box' in Box_data.keys()):
            x1 ,y1, x2, y2 = Box_data['graph_box']
            for id, role in enumerate(Task3_roles):
                if(role['role'] == "tick_label"):
                    tbox = Task2_boxes[id]
                    bb = tbox['polygon']
                    xp = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0
                    yp = (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0
                    if(x1+5 < xp and xp< x2-5 and y1+5<yp and yp < y2-5):
                        continue
                    Need_match.append(Task2_boxes[id])
        else:
            for id, role in enumerate(Task3_roles):
                if(role['role'] == "tick_label"):
                    Need_match.append(Task2_boxes[id])
            
        Image = cv2.imread(Img_path)
        or_h, or_w  = Image.shape[:2]
        
        Tscores, Tpoints = detect([model_2], Image, Img_path)

        axes = {}
        axes['x-axis'] = []
        axes['y-axis'] = []
        axes['x-axis-2'] = []
        axes['y-axis-2'] = []
                
        for box in Need_match:
            id = box['id']
            bb = box['polygon']
            xp = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0
            yp = (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0

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
                        
            if(Task1_ctype in Re_PMC):
                mcls = 1 - mcls
                
            if(mcls == 0):
                if(Task1_ctype in Re_PMC):
                    if(mpx <= or_w/2):
                        axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                else:
                    if(mpy >= or_h/2):
                        axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                cv2.circle(Image, (int(mpx), int(mpy)), 2, (0, 255, 0), -1)
            else:
                if(Task1_ctype in Re_PMC):
                    if(mpy >= or_h/2):
                        axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                else:
                    if(mpx <= or_w/2):
                        axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                            
                cv2.circle(Image, (int(mpx), int(mpy)), 2, (0, 0, 255), -1)
                
            cv2.line(Image, (int(xp), int(yp)), (int(mpx), int(mpy)), [0,0,255],1)
        
        T4_out_dict = {}
        T4_out_dict['axes'] = axes
        Json_data['task4']["output"] = T4_out_dict
                
        cv2.imwrite('outs/' + Img_path.split('/')[-1],Image)
        
        json.dump(Json_data, open(PMC_Test_Out_Dir + json_file, 'w'), indent=4)

    print("*"*32)
#### SYN
    SYN_Test_Img_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_3_4_5/Charts/'
    SYN_Test_Json_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_3_4_5/Inputs/'
    SYN_Test_Cls_Dir  = '/data/Dataset/Chart/Test_Task3_output/Syn_pred_json/'
    SYN_Test_Box_Dir  = '/data/Dataset/Chart/Test_graph_legend_json/'
    SYN_Test_Out_Dir  = '/data/zzg/ICPR_Chart/AE_Chart/results/SYN_Test/'
    
    Ctype_Dict = {}
    Re_SYN = ["Grouped horizontal bar", "Stacked horizontal bar", "Horizontal box"]

    SYN_Test_Json_Files = os.listdir(SYN_Test_Json_Dir)
    for json_file in SYN_Test_Json_Files:
        print(json_file)
        Img_path  = SYN_Test_Img_Dir + json_file.replace('json', 'png')
        Json_path = SYN_Test_Json_Dir + json_file
        Cls_Path  = SYN_Test_Cls_Dir + json_file
        Box_Path  = SYN_Test_Box_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))
        Cls_data  = json.load(open(Cls_Path,'r'))
        Box_data  = json.load(open(Box_Path,'r'))
        
        Task1_ctype = Json_data['task1_output']['chart_type']  ###  ctype
        Task2_boxes = Json_data['task2_output']['text_blocks']
        Task3_roles = Cls_data['task3']['output']['text_roles']
       # Box_Graph   = Box_data['graph_box']
        
        Need_match = []
        if('graph_box' in Box_data.keys()):
            x1 ,y1, x2, y2 = Box_data['graph_box']
            for id, role in enumerate(Task3_roles):
                if(role['role'] == "tick_label"):
                    tbox = Task2_boxes[id]
                    bb = tbox['bb']
                    xp = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0
                    yp = (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0
                    if(x1+5 < xp and xp< x2-5 and y1+5<yp and yp < y2-5):
                        continue
                    Need_match.append(Task2_boxes[id])
        else:
            for id, role in enumerate(Task3_roles):
                if(role['role'] == "tick_label"):
                    Need_match.append(Task2_boxes[id])
                        
        Image = cv2.imread(Img_path)
        or_h, or_w  = Image.shape[:2]
        
        Tscores, Tpoints = detect([model_2], Image, Img_path)

        axes = {}
        axes['x-axis'] = []
        axes['y-axis'] = []
        axes['x-axis-2'] = []
        axes['y-axis-2'] = []
                
        for box in Need_match:
            id = box['id']
            bb = box['bb']
            xp = (bb['x0'] + bb['x1'] + bb['x2'] + bb['x3'])/4.0
            yp = (bb['y0'] + bb['y1'] + bb['y2'] + bb['y3'])/4.0

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
                        
            if(Task1_ctype in Re_SYN):
                mcls = 1 - mcls
                
            if(mcls == 0):
                if(Task1_ctype in Re_SYN):
                    if(mpx <= or_w/2):
                        axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                else:
                    if(mpy >= or_h/2):
                        axes['x-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['x-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                cv2.circle(Image, (int(mpx), int(mpy)), 2, (0, 255, 0), -1)
            else:
                if(Task1_ctype in Re_SYN):
                    if(mpy >= or_h/2):
                        axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                else:
                    if(mpx <= or_w/2):
                        axes['y-axis'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                    else:
                        axes['y-axis-2'].append({"id": id, "tick_pt": {"x": int(mpx), "y": int(mpy)}})
                            
                cv2.circle(Image, (int(mpx), int(mpy)), 2, (0, 0, 255), -1)
                
            cv2.line(Image, (int(xp), int(yp)), (int(mpx), int(mpy)), [0,0,255],1)
            
        cv2.imwrite('outs/' + Img_path.split('/')[-1],Image)
        
        T4_out_dict = {}
        T4_out_dict['task4'] = {}
        #T4_out_dict['task4']['input'] = null
        T4_out_dict['task4']['name'] = "Axes Analysis"
        T4_out_dict['task4']['output'] = {}
        T4_out_dict['task4']['output']['axes'] = axes
        
        json.dump(T4_out_dict, open(SYN_Test_Out_Dir + json_file, 'w'), indent=4)
        