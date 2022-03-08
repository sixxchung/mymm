ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
#import modelindex

WORK_DIR = osp.join(ROOT_DIR, 'mymm/oxford')
DATA_DIR = osp.join(WORK_DIR, 'data')

IMG_PREFIX = 'images'
ANN_PREFIX = 'annotations/xmls'

# metaTrain0 = osp.join(DATA_DIR, ANN_PREFIX, 'trainval.txt')
# metaTest0  = osp.join(DATA_DIR, ANN_PREFIX, 'test.txt')

# metaTrain = osp.join(DATA_DIR, 'train.txt')
# metaValid = osp.join(DATA_DIR, 'valid.txt')
# metaTest  = osp.join(DATA_DIR, 'test.txt')

# iconfig_file    = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_oxford.py')
# checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# ann_file = osp.join(DATA_DIR, 'train.txt')
# imgNm = 'beagle_100'

# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread(imgFile)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# from bs4 import BeautifulSoup
# bs = BeautifulSoup(open(annFile), 'xml')
# print(bs.prettify())

# imgFileNm = 'Abyssinian_16'
# ann_file  = '/home/oschung_skcc/git/mymm/oxford/data/annotations/xmls/Abyssinian_16.xml'    # Path

import sys
sys.path.append(osp.join(ROOT_DIR, 'mmdetection'))
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom  import CustomDataset

import mmcv
import cv2
import numpy as np
import pandas as pd

# annotation xml 파일 파싱해서 bbox정보 추출
import xml.etree.ElementTree as ET
def get_bboxes_from_xml(annPath):
    xmlFileName = os.path.basename(annPath)
    tree = ET.parse(annPath)
    root = tree.getroot()

    bbox_names = []
    bboxes = []
    for obj in root.findall('object'):
        bbox_name = xmlFileName[:xmlFileName.rfind('_')]
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)

        bbox_names.append(bbox_name)
        bboxes.append([x1, y1, x2, y2])

    return bbox_names, bboxes

@DATASETS.register_module(force=True)
class OxfordDataset(CustomDataset):
    CLASSES = ('Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier')

    def load_annotations(self, ann_file):
        # mmdetection프레임웍이 Config에서 ann_file인자를 찾아 파라미터로 사용.
        print("### sixx --- middle custom dataset ---")
        cat2label = {k:i for i, k in enumerate(self.CLASSES)}    
        annFileNm_list = mmcv.list_from_file(self.ann_file)
        # imgFileNm = annFileNm_list[0]

        data_info = []    # 포맷 중립 데이터를 담을 list 객체
        for imgFileNm in annFileNm_list:
            print(f"====sixx====::{imgFileNm}") 
            if imgFileNm is None:
                continue
            # 개별 image에 대한 메타정보 및  annotation 정보 저장용
            imgBaseNm = str(imgFileNm)+'.jpg'     # 경로제외하고, image파일명만
            imgPath = osp.join(DATA_DIR, IMG_PREFIX, imgBaseNm)
            image = cv2.imread(imgPath)
            height, width = image.shape[:2]  # image로부터 너비, 높이를 집접구함. 
            data_imgMeta = {
                'filename': imgBaseNm,  
                'width':  width, 
                'height': height 
            }
            
            annBaseNm = str(imgFileNm)+'.xml'
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
            data_info.append(data_imgMeta)
        return data_info