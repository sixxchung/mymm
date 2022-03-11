
#import copy
import os
import os.path as osp
#import modelindex
ROOT_DIR = '/home/oschung_skcc/git'
WORK_DIR = osp.join(ROOT_DIR, 'mymm/oxford')
DATA_ROOT= osp.join(WORK_DIR, 'data')

IMG_PREFIX = osp.join(DATA_ROOT,'images')
ANN_PREFIX = osp.join(DATA_ROOT,'annotations','xmls')  

# metaTrain0 = osp.join(DATA_ROOT, 'annotations/trainval.txt')
# metaTest0  = osp.join(DATA_ROOT, 'annotations/test.txt')

# metaTrain = osp.join(DATA_ROOT, 'train.txt')
# metaValid = osp.join(DATA_ROOT, 'valid.txt')
# metaTest  = osp.join(DATA_ROOT, 'test.txt')

# iconfig_file    = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_oxford.py')
# checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# ann_file = osp.join(DATA_ROOT, 'train.txt')
# imgNm = 'beagle_100'

# imgFile = f'{IMG_PREFIX}/{imgNm}.jpg'
# annFile = f'{ANN_PREFIX}/{imgNm}.xml'

# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread(imgFile)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# from bs4 import BeautifulSoup
# bs = BeautifulSoup(open(annFile), 'xml')
# print(bs.prettify())

import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR) # '/home/oschung_skcc/git'
from mmdetection.mmdet.datasets.builder import DATASETS
from mmdetection.mmdet.datasets.custom  import CustomDataset

import mmcv
import cv2
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

# annotation xml 파일 파싱해서 bbox정보 추출
def get_bboxes_from_xml(annoXmlPath):
    xmlFileName = os.path.basename(annoXmlPath)
    tree = ET.parse(annoXmlPath)
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
        print("### sixx --- middle custome dataset ---")
        print('### sixx --- self.ann_file  :', self.ann_file)

        cat2label = {k:i for i, k in enumerate(self.CLASSES)}    
        ann_list = mmcv.list_from_file(self.ann_file)

        data_info = []    # 포맷 중립 데이터를 담을 list 객체
        for imgNm in ann_list:
            if imgNm is None:
                print("sixx None!!!")
                continue
            # 개별 image에 대한 메타정보 및  annotation 정보 저장용
            filename = str(imgNm)+'.jpg'     # 경로제외하고, image파일명만
            imgFile = f'{IMG_PREFIX}/{filename}'
            # print('### sixx --- imgFile  :', imgFile) 
            image = cv2.imread(imgFile)
            height, width = image.shape[:2]  # image로부터 너비, 높이를 집접구함. 
            data_imgMeta = {
                'filename': filename,  
                'width':  width, 
                'height': height 
            }
            
            xmlFileName = str(imgNm)+'.xml'
            annoXmlPath = osp.join(ANN_PREFIX, xmlFileName)
            annlines = mmcv.list_from_file(annoXmlPath)
            # if not osp.exists(annFilename)|os.stat(annFilename).st_size==0 :
            #     continue
            
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            bbox_names, bboxes = get_bboxes_from_xml(annoXmlPath)
            for bbox_nm, bbox in zip(bbox_names, bboxes):
                # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가
                if bbox_nm in cat2label:
                    gt_bboxes.append(bbox)
                    gt_labels.append(cat2label[bbox_nm])
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