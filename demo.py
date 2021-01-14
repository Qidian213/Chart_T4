import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from models import Get_model
from cfg import Cfg_Opts

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def detect(model, image, key, threshold=0.3):
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
    tags     = out_dict['tag'].cpu().squeeze().data.numpy()
   
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
       # offset  = offset.cpu().squeeze().data.numpy() ###

        stride = 4
        tscores, tpoints = [], []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            px = (cx + offset[0,cy,cx])* stride
            py = (cy + offset[1,cy,cx])* stride
            tagx = (cx + tags[0,cy,cx])* stride
            tagy = (cx + tags[1,cy,cx])* stride
            
            tscores.append(score)
            tpoints.append([px, py, tagx, tagy])
        Tscores.append(tscores)
        Tpoints.append(tpoints)
    return Tscores, Tpoints

def detect_image(model, file, key):
    image = cv2.imread(file)
    Tscores, Tpoints = detect(model, image, key)

    for point in Tpoints[0]:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    for point in Tpoints[1]:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

    return image

if __name__ == "__main__":
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/DLA_34_2020-10-25-16-16-40/Epoch_best.pth")

    #["horizontal bar", "Grouped horizontal bar", "Stacked horizontal bar", "Horizontal box"]

    json_data = json.load(open('/data/Dataset/Chart/Task4/ATest.json','r'))
    for key in json_data.keys():
       # if('hbox' in key):
            print(key)
            image     = detect_image(model, '/data/Dataset/Chart/Task4/' + key, key)
            data_dict = json_data[key]
            x_axis    = data_dict['x-axis']
            y_axis    = data_dict['y-axis']
            for point in x_axis:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
            for point in y_axis:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
            cv2.imwrite('outs/' + key.split('/')[-1],image)

# if __name__ == "__main__":
    # cfg   = Cfg_Opts()
    # model = Get_model(cfg)
    # model.eval()
    # model.cuda()
    # model.load_param("work_space/DLA_34_2020-10-22-20-35-38/Epoch_best.pth")

  # #  Img_dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_3_4_5/Charts/'
    # Img_dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/images/'
    # Img_files = os.listdir(Img_dir)
    # for key in Img_files:
        # print(key)
        # image     = detect_image(model, Img_dir + key, key)

        # # for point in x_axis:
            # # cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
        # # for point in y_axis:
            # # cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        # cv2.imwrite('outs/' + key,image)
        