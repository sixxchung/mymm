ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
# from cv2 import dnn_DetectionModel
WORK_DIR = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )
# WORK_DIR = osp.join(ROOT_DIR, 'mymm/kitty_tiny')
DATA_DIR = osp.join(WORK_DIR, 'data')

IMG_PREFIX = 'image_2' 
ANN_PREFIX = 'label_2'

import sys
sys.path.append('/home/oschung_skcc/git/mmdetection')
# sys.path.append( os.path.dirname(os.path.abspath(__file__)) )
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
        # mmdetection프레임웍이 Config에서 ann_file인자를 찾아 파라미터로 사용.
        print("### sixx --- middle custome dataset ---")
        cat2label = {k:i for i, k in enumerate(self.CLASSES)}    
        annFileNm_list = mmcv.list_from_file(osp.join(DATA_DIR, self.ann_file))

        data_info = []    # 포맷 중립 데이터를 담을 list 객체
        for imgFileNm in annFileNm_list:
            if imgFileNm is None:
                continue
            # 개별 image에 대한 메타정보 및  annotation 정보 저장용
            imgBaseNm = str(imgFileNm)+'.jpeg'     # 경로제외하고, image파일명만
            imgPath =  osp.join(DATA_DIR, IMG_PREFIX, imgBaseNm)
            image = cv2.imread(imgPath)
            height, width = image.shape[:2]  # image로부터 너비, 높이를 집접구함. 
            data_imgMeta = {
                'filename': imgBaseNm,  
                'width':  width, 
                'height': height 
            }
             
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
                # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가
                if bboxNm in cat2label:
                    gt_bboxes.append(bbox)
                    gt_labels.append(cat2label[bboxNm])
                else:  # 그렇지 않으면 gt_bboxes_ignore와  gt_labels_ignore에 추가
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
            
            # data_imgMeta < image의 메타정보>에 data_annMeta 정보 추가 ('ann' key값으로 value값은 모두 np.array임)
            data_annMeta = {
                'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                'labels': np.array(gt_labels, dtype=np.compat.long),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.compat.long)
            }
            data_imgMeta.update(ann=data_annMeta)
            # data_info에 data_imgMeta를 추가하여, 전체 annotation 파일들에 대한 정보를 가지는 dict생성 
            data_info.append(data_imgMeta)
        return data_info