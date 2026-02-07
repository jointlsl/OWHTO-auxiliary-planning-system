#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
    :Purpose: To detect landmarks for lower limb
    :Authors: Hongkaiz
    :Date   : 2025-09-13
    :Version: 0.1
    :Usage  : python landmark_detect.py
"""

import os
import sys
import cv2
import torch
import time
import json
import argparse
import numpy as np
from PIL   import Image

import __init__
from src.script.utils.kit import *
from src.script.networks  import get_net


# 参数
def parse_cmdline():
    """set up argparser for the program
    """
    parser = argparse.ArgumentParser(description='The Pipeline of Landmarks Detection')

    required = parser.add_argument_group('required arguments')
    required.add_argument("-image_path",    dest='image_path',  required=False,  default='test_data/demo1.png',   help='The input image file')
    required.add_argument("-save_path",     dest='save_path',   required=False,  default=None,  help='test_data/result')
    required.add_argument("-region",   dest='region', required=False,  default=1, type=int,  help='This parameter specifies the region of interest; 1=left knee,2=right knee')
    required.add_argument("-id",       dest='id',     required=False,  default='UNDEFINED',  help='patient ID')
    required.add_argument("-debug",    dest='debug',  required=False,  default=True, action='store_true',  help='debug mode')

    args = parser.parse_args()

    return args

## 读取影像
def readImage(path_to_img, size_tuple, is_normal=False):
    """Read image from path and return a numpy.ndarray in shape of cxwxh
    """
    img = Image.open(path_to_img)
    if os.path.splitext(path_to_img)[-1] == ".png":
        img = img.convert("RGB")

    origin_size = img.size
    img = img.resize(size_tuple)
    arr  = np.array(img)                        
    arr  = np.transpose(arr, (2, 1, 0)) 
    
    if is_normal:
        arr  = (arr - 46.2391)/57.6113
    arr  = arr.astype(np.float32)  

    return arr, origin_size

## 保存结果
def save_dict2json(dict_obj, name, Mycls=None):
    """save dict to json file
    """
    js_obj = json.dumps(dict_obj, cls=Mycls, indent=4)
    with open(name, 'w') as file_obj:
        file_obj.write(js_obj)

## 推理
def detect_pipeline(path_to_image, path_to_save=None, patient_id="UNDIFNE", target_size=(512, 512), patient_region=1, debug_opt=True):
    
    """landmarks detection pipeline
    """
    
    ##preprocess
    img_arr, origin_size = readImage(path_to_image, target_size)
    img_data = np.expand_dims(img_arr, axis=0)

    ##load model
    network_name = 'unet2d'
    device       = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") ##use cpu or gpu
    net_params   = dict({'in_channels': 3, 'out_channels': 14})
    print("load network name:{}".format(network_name))
    model = get_net(network_name)(** net_params)
    model.to(device)

    ###load weight for different region model
    cur_file_folder = os.path.dirname(os.path.abspath(__file__))
    if patient_region == 1:
        path_to_model = os.path.join(cur_file_folder, '../model/left/llimb.pt')
    elif patient_region == 2:
        path_to_model = os.path.join(cur_file_folder, '../model/right/llimb.pt')
    else:
        print('Run mode is not support!')
        sys.exit(0)
    ##onif
    path_to_model   = os.path.abspath(path_to_model)
    print('load weight file: {}'.format(path_to_model))
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img_tensor = torch.FloatTensor(img_data).to(device)
    heatmap_pred_tensor = model(img_tensor)
    heatmap_pred_arr    = heatmap_pred_tensor.detach().cpu().numpy()[0]
    
    ###get landmarks from heatmap
    landmark_pred_list  = getPointsFromHeatmap(heatmap_pred_arr)

    ##convert the predicted points to the original image size
    dict_summary = {}
    for idx, mypoint in enumerate(landmark_pred_list):
        pred_point = list(int(round(p*new/old)) for p, new, old in zip(mypoint, origin_size, img_arr[0].shape))
        dict_summary[idx+1] = pred_point
    ##onfor
    
    ## Save point information to json

    if path_to_save:
        if  patient_region == 1:
            path_to_json = os.path.join(path_to_save, patient_id + "_left.json")
        elif patient_region == 2:
            path_to_json = os.path.join(path_to_save, patient_id + "_right.json")
        else:
            print('Patient region is not support!')
            sys.exit(0)
    
        save_dict2json(dict_summary, path_to_json)

    ###debug output
    if debug_opt:
        image_arr = cv2.imread(path_to_image)

        for id, coord in dict_summary.items():
            cv2.circle(image_arr, coord, 2, (255, 0, 0), 1)
            
            text_size = 0.5
            if id in [3,4,5]:
                coord[1] = coord[1]-5
            
            if id in [6,7,8,9,10,11,12]:
                coord[1] = coord[1]+10
                text_size = 0.3

            cv2.putText(image_arr, str(id), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), 1)

        if  patient_region == 1:
            path_tmp_img = os.path.join(path_to_save, patient_id+'_left_debug.png')
        elif patient_region == 2:
            path_tmp_img = os.path.join(path_to_save, patient_id+'_right_debug.png')
        else:
            print('Patient region is not support!')
            sys.exit(0)

        print(path_tmp_img)
        cv2.imwrite(path_tmp_img, image_arr)


    return dict_summary

def main(args=None):

    ##start time
    st = time.time()

    ##prog args
    parser_args = parse_cmdline()

    detect_pipeline(parser_args.image_path, 
                    parser_args.save_path, 
                    "UNDIFNE", 
                    (512, 512), 
                    parser_args.region, 
                    parser_args.debug)

    ##end time
    et = time.time()
    t = et - st
    
    print('function running time:', t)    
##ondef
#

if __name__ == '__main__':
    main(sys.argv)