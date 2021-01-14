# COPYRIGHT University of Buffalo 2019.  Used with permission
# Written by Bhargava Urala
import os
import sys
import cv2
import json
import numpy as np

LOW_THRESHOLD = 0.01
HIGH_THRESHOLD = 0.02

def extract_tick_point_pairs(js, file_json, gt_flag):
    def get_coords(tpp):
        ID = tpp['id']
        x, y = tpp['tick_pt']['x'], tpp['tick_pt']['y']
        if ID is None or ID == 'null':
            print(ID)
        return (ID, (x, y))
    if('PMC' in file_json and gt_flag):
        axes = js['task6']['input']['task4_output']['axes']
    else:
        axes = js['task4']['output']['axes']
    tpp_x = [get_coords(tpp) for tpp in axes['x-axis']]
    tpp_x = {ID: coords for ID, coords in tpp_x if ID is not None}
    tpp_y = [get_coords(tpp) for tpp in axes['y-axis']]
    tpp_y = {ID: coords for ID, coords in tpp_y if ID is not None}
    return tpp_x, tpp_y

def get_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.linalg.norm([x1 - x2, y1 - y2])

def get_distance_score(distance, low, high):
    if distance <= low:
        return 1.
    if distance >= high:
        return 0.
    return 1. - ((distance - low) / (high - low))

def get_axis_score(gt, res, lt, ht):
    if len(gt) == 0 and len(res) == 0:
        return 1.
    score = 0.
    for ID, gt_coords in gt.items():
        if ID not in res:
            continue
        distance = get_distance(gt_coords, res[ID])
        score += get_distance_score(distance, lt, ht)
    return score

def eval_task4(Gt_jsons, Pd_jsons, Img_files):
    total_recall    = 0.
    total_precision = 0.

    for ind in range(len(Gt_jsons)):
        gt_json  = Gt_jsons[ind]
        pd_json  = Pd_jsons[ind]
        img_file = Img_files[ind]

        gt  = json.load(open(gt_json,'r'))
        gt_x, gt_y = extract_tick_point_pairs(gt, gt_json, gt_flag = True)
        
        res = json.load(open(pd_json,'r'))
        res_x, res_y  = extract_tick_point_pairs(res, pd_json, gt_flag=False)
        
        h, w, _       = cv2.imread(img_file).shape
        
        diag          = ((h ** 2) + (w ** 2)) ** 0.5
        lt, ht        = LOW_THRESHOLD * diag, HIGH_THRESHOLD * diag
        score_x       = get_axis_score(gt_x, res_x, lt, ht)
        score_y       = get_axis_score(gt_y, res_y, lt, ht)
        recall_x      = score_x / len(gt_x) if len(gt_x) > 0 else 1.
        recall_y      = score_y / len(gt_y) if len(gt_y) > 0 else 1.
        precision_x   = score_x / len(res_x) if len(res_x) > 0 else 1.
        precision_y   = score_y / len(res_y) if len(res_y) > 0 else 1.
        precision_x   = 0. if len(gt_x) == 0 and len(res_x) > 0 else precision_x
        precision_y   = 0. if len(gt_y) == 0 and len(res_y) > 0 else precision_y
        total_recall += (recall_x + recall_y) / 2.
        total_precision += (precision_x + precision_y) / 2.
        
        # pc = (precision_x + precision_y) / 2
        # if( pc <0.9):
            # print(img_file, precision_x, precision_y, pc)
    
    if(len(Gt_jsons)):
        total_recall    /= len(Gt_jsons)
        total_precision /= len(Gt_jsons)
    
    if total_recall  == 0 and total_precision == 0:
        f_measure = 0
    else:
        f_measure = 2 * total_recall * total_precision / (total_recall + total_precision)
        
    print('Average Recall:', total_recall)
    print('Average Precision:', total_precision)
    print('Average F-Measure:', f_measure)

if __name__ == '__main__':
    # Grouped horizontal bar 66
    # Scatter 56
    # Stacked vertical bar 70
    # Line 66
    # Vertical box 67
    # Stacked horizontal bar 67
    # Grouped vertical bar 73
    # Horizontal box 62

    # scatter 95
    # vertical bar 225
    # line 264
    # horizontal bar 71
    # vertical box 71

    Val_Json  = json.load(open('/data/Dataset/Chart/Task4/TTest.json','r'))
    
    Sy_list  = ['Grouped horizontal bar', 'Scatter', 'Stacked vertical bar', 'Line', 'Vertical box', 'Stacked horizontal bar', 'Grouped vertical bar', 'Horizontal box', 'Synthetic']
    Pmc_list = ['scatter', 'vertical bar', 'line', 'horizontal bar', 'vertical box', 'PMC']

    GT_JSONS = []
    PD_JSONS = []
    IMG_FILE = []
    for cls in Sy_list:
        Gt_jsons  = []
        Pd_jsons  = []
        Img_files = []

        for key in Val_Json.keys():
            chart_type = Val_Json[key]['chart_type']
            if(chart_type==cls and 'Synthetic' in key):
                Img_files.append('/data/Dataset/Chart/Task4/' + key)
                Gt_jsons.append('/data/Dataset/Chart/Task4/' + key.replace('png', 'json').replace('Synthetic_Img_Test', 'Synthetic_Json_Test_Conv'))
                Pd_jsons.append('results/' + key.replace('png', 'json').replace('Synthetic_Img_Test', 'Synthetic_Json_Test_Conv'))
        
        for key in Val_Json.keys():
            if('Synthetic'==cls and 'Synthetic' in key):
                Img_files.append('/data/Dataset/Chart/Task4/' + key)
                Gt_jsons.append('/data/Dataset/Chart/Task4/' + key.replace('png', 'json').replace('Synthetic_Img_Test', 'Synthetic_Json_Test_Conv'))
                Pd_jsons.append('results/' + key.replace('png', 'json').replace('Synthetic_Img_Test', 'Synthetic_Json_Test_Conv'))
        
        if('Synthetic'==cls ):
            GT_JSONS.extend(Gt_jsons)
            PD_JSONS.extend(Pd_jsons)
            IMG_FILE.extend(Img_files)
            
        print(f'-------------Synthetic-{cls}-{len(Gt_jsons)}-------------')
        eval_task4(Gt_jsons, Pd_jsons, Img_files)

    for cls in Pmc_list:
        Gt_jsons  = []
        Pd_jsons  = []
        Img_files = []

        for key in Val_Json.keys():
            chart_type = Val_Json[key]['chart_type']
            if(chart_type==cls and 'PMC' in key):
                Img_files.append('/data/Dataset/Chart/Task4/' + key)
                Gt_jsons.append('/data/Dataset/Chart/Task4/' + key.replace('jpg', 'json').replace('PMC_Img_Test', 'PMC_Json_Test'))
                Pd_jsons.append('results/' + key.replace('jpg', 'json').replace('PMC_Img_Test', 'PMC_Json_Test'))

        for key in Val_Json.keys():
            chart_type = Val_Json[key]['chart_type']
            if('PMC'==cls and 'PMC' in key):
                Img_files.append('/data/Dataset/Chart/Task4/' + key)
                Gt_jsons.append('/data/Dataset/Chart/Task4/' + key.replace('jpg', 'json').replace('PMC_Img_Test', 'PMC_Json_Test'))
                Pd_jsons.append('results/' + key.replace('jpg', 'json').replace('PMC_Img_Test', 'PMC_Json_Test'))
        
        if(cls == 'PMC'):
            GT_JSONS.extend(Gt_jsons)
            PD_JSONS.extend(Pd_jsons)
            IMG_FILE.extend(Img_files)
        print(f'-------------PMC-{cls}-{len(Gt_jsons)}-------------')
        eval_task4(Gt_jsons, Pd_jsons, Img_files)

    print(f'------------Synthetic-PMC-{len(GT_JSONS)}-------------')
    eval_task4(GT_JSONS, PD_JSONS, IMG_FILE)
