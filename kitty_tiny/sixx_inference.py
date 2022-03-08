ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
# from cv2 import dnn_DetectionModel
WORK_DIR = os.path.dirname(os.path.realpath(__file__))
config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_tidy.py')
checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

DATA_DIR = osp.join(WORK_DIR, 'data')
IMG_PREFIX = 'image_2' 
ANN_PREFIX = 'label_2'

# imgPath = osp.join(DATA_DIR, IMG_PREFIX,'000068.jpeg')

# ### 4 Inference 
import os 
import os.path as osp
from re import I
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
# $ python tools/sixx_train.py configs/faster_rcnn_r50_fpn_1x_tidy.py 
from mmdet.datasets import build_dataset
from mmdet.models   import build_detector
from mmdet.apis     import train_detector


img = cv2.imread(image_file)

cfg = Config.fromfile('../configs/faster_rcnn_r50_fpn_1x_tidy.py')
model = build_detector(
    cfg.model, 
    train_cfg = cfg.get('train_cfg'),
    test_cfg  = cfg.get('test_cfg')
)

model.cfg = cfg
model = init_detector(config_file, checkpoint_file, device='cuda:0')

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)