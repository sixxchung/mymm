ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
WORK_DIR = os.path.dirname(os.path.realpath(__file__))

#config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_tidy.py')
#checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

DATA_DIR = osp.join(WORK_DIR, 'data')
IMG_PREFIX = 'image_2' 
ANN_PREFIX = 'label_2'

import cv2
imgPath = osp.join(DATA_DIR, IMG_PREFIX,'000068.jpeg')
img_array = cv2.imread(imgPath)

CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
cat2label = {k:i for i, k in enumerate(CLASSES)}
labels_to_names_seq = {i:k for i, k in enumerate(CLASSES)}

import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

import numpy as np
from mmcv import Config
from mmdet.models import build_detector
from mmdet.apis import train_detector

config_file = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_tidy.py')
cfg = Config.fromfile(config_file)

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.cfg = cfg

#  inference 시각화용 함수 생성. 
def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):
    # 모델 
    # 원본 이미지 array
    # filtering할 기준 class confidence score 

    # 인자로 들어온 image_array를 복사. 
    draw_img = img_array.copy()
    bbox_color=(0, 255, 0)
    text_color=(0, 0, 255)

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음. 
    # results는 80개의 2차원 array(shape=(오브젝트갯수, 5))를 가지는 list. 
    results = inference_detector(model, img_array)

    # 80개의 array원소를 가지는 results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
    # results 리스트의 위치 index가 바로 COCO 매핑된 Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
    for result_ind, result in enumerate(results):
        print(f"result_ind :: {result_ind}")
        print(f"result :: {result}")
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행. 
        if len(result) == 0:
            continue
        
        # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외. 
        result_filtered = result[np.where(result[:, 4] > score_threshold)]
        
        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
        for i in range(len(result_filtered)):
            # 좌상단, 우하단 좌표 추출. 
            left = int(result_filtered[i, 0])
            top = int(result_filtered[i, 1])
            right = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            caption = f"{labels_to_names_seq[result_ind]}:{result_filtered[i, 4]}"

            cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
        if is_print:
            print(caption)
    return draw_img

from matplotlib import pyplot as plt
plt.show(get_detected_img(model, img_array))
