ROOT_DIR = '/home/oschung_skcc/git'
import os
import os.path as osp
WORK_DIR = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )  # 'mymm/kitty_tiny'
#'/home/oschung_skcc/git/mymm/kitty_tiny'
DATA_DIR = osp.join(WORK_DIR, 'data')
IMG_PREFIX = 'image_2' 
ANN_PREFIX = 'label_2'

import sys
sys.path.append('/home/oschung_skcc/git/mmdetection') # ( os.path.dirname(os.path.abspath(__file__)) )
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom  import CustomDataset

import mmcv
import cv2
import numpy as np
import pandas as pd

# annotation xml 파일 파싱해서 bbox정보 추출
def get_bboxes_from_xml(annPath):
    annLines = mmcv.list_from_file(annPath)
    content  = [line.strip().split(' ') for line in annLines]
    bbox_names = [x[0] for x in content]
    bboxes = [ [float(info) for info in x[4:8]] for x in content]

    return bbox_names, bboxes

# imgFileNm = '000006'
# ann_file  = '/home/oschung_skcc/git/mymm/kitty_tiny/data/label_2/000006.txt'
@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
    
    def load_annotations(self, ann_file):
        # mmdetection 프레임웍이 Config에서 ann_file(path)인자를 찾아 파라미터로 사용.
        cat2label = {k:i for i, k in enumerate(self.CLASSES)}    
        annFileNm_list = mmcv.list_from_file(self.ann_file)

        data_info = []
        for imgFileNm in annFileNm_list:
            if imgFileNm is None:
                continue

            ### IMAGE metadata 
            imgBaseNm = str(imgFileNm)+'.jpeg'
            imgPath = osp.join(DATA_DIR, IMG_PREFIX, imgBaseNm)
            image = cv2.imread(imgPath)
            height, width = image.shape[:2]
            img_metaData = {
                'filename': imgBaseNm,  
                'width':  width, 
                'height': height 
            }

            ### Annotation metadata   
            annBaseNm = str(imgFileNm)+'.txt'
            annPath = osp.join(DATA_DIR, ANN_PREFIX, annBaseNm)
            if not osp.exists(annPath):
                continue
            elif os.stat(annPath).st_size==0:
                continue           

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            bbox_names, bboxes = get_bboxes_from_xml(annPath)
            for bboxNm, bbox in zip(bbox_names, bboxes):
                if bboxNm in cat2label:
                    gt_bboxes.append(bbox)
                    gt_labels.append(cat2label[bboxNm])
                else: 
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
            ann_metaData = {
                'bboxes':        np.array(gt_bboxes,        dtype=np.float32).reshape(-1, 4),
                'labels':        np.array(gt_labels,        dtype=np.compat.long),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.compat.long)
            }
            img_metaData.update(ann=ann_metaData)

            data_info.append(img_metaData)
        print(data_info[0])
        return data_info

import datetime
print(f"---registerd custom dataset by sixx on {datetime.datetime.now()}")