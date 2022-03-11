ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
# from re import I     # Reqular expression operations, IgnoreCase
WORK_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = osp.join(WORK_DIR, 'data')
IMG_PREFIX = 'image_2' 
ANN_PREFIX = 'label_2'

import cv2
from matplotlib import pyplot as plt 
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np

def get_detected_img(model, imgPath, score_threshold=0.3, is_print=True):
    img_array = cv2.imread(imgPath)
    # plt.imshow(draw_img)
    bbox_color = (  0,255,   0)   # Green
    text_color = (  0,  0, 255)   # Blur

    results = inference_detector(model, img_array)
    for result_ind, result in enumerate(results):
        if len(result)==0:
            continue
        result_filtered = result[ np.where(result[:, 4] > score_threshold)]
        
        for i in range(len(result_filtered)):
            # 좌상단 좌
            left = int(result_filtered[i, 0])
            top  = int(result_filtered[i, 1])
            # 우하단 좌표
            right  = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            cv2.rectangle(img_array, (left, top), (right, bottom), color=bbox_color, thickness=2)
            # Class Caption 
            caption = f"{labels_to_names_seq[result_ind]}: {result_filtered[i, 4]}"
            cv2.putText(img_array, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            
            if is_print:
                print(caption)
        return img_array

config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_tidy.py')
checkpoint_file = osp.join(WORK_DIR, 'tutorial_exps/latest.pth')

model = init_detector(config_file, checkpoint_file)
imgPath = osp.join(DATA_DIR, IMG_PREFIX,'000068.jpeg')

CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
labels_to_names_seq = {i:k for i, k in enumerate(CLASSES)}
draw_img = get_detected_img(model, imgPath, score_threshold=0.3, is_print=True)  
plt.figure(figsize=(4,4))#(15,10))
plt.imshow(draw_img)