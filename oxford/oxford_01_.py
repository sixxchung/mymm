##### CustomDataset을 수정하여 PetDataset 클래스를 생성
import copy
import glob

import mmcv
import cv2
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

import numpy as np
import pandas as pd
import os
import os.path as osp
import xml.etree.ElementTree as ET

baseDirectory = '/home/oschung_skcc/git/mmdetection/data/oxford/' 
imgDirectory = osp.join(baseDirectory, 'images/')
annoDirectory= osp.join(baseDirectory, 'annotations/') 

metaTrain = osp.join(baseDirectory, 'train.txt')
metaTest  = osp.join(baseDirectory, 'test.txt') 


pet_df = pd.read_csv(metaTrain, sep=' ', header=None, names=['img_name'])
pet_df['class_name'] = pet_df['img_name'].apply(lambda x: x[:x.rfind('_')])
PET_CLASSES = pet_df['class_name'].unique().tolist()
len(PET_CLASSES)

# annotation xml 파일 파싱해서 bbox정보 추출
def get_bboxes_from_xml(xmlFileName):
  print(xmlFileName)
  annoXmlPath = osp.join(annoDirectory, xmlFileName)
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
class PetDataset(CustomDataset):
    CLASSES = PET_CLASSES
    # ann_file : annotation정보를 가진 파일의 모든 목록 텍스트 파일
    # data_root='/home/oschung_skcc/git/mmdetection/data/oxford/'
    # ann_file='train.txt' 
    # img_prefix='images/'

    # def __init__(self, data_root, ann_file, img_prefix):
    #     self.data_root  = data_root
    #     self.ann_file   = osp.join(data_root, ann_file)
    #     self.img_prefix = osp.join(data_root, img_prefix)
    #     self.data_infos = self.load_annotations(self.ann_file)

    def load_annotations(self, ann_file):
        cat2label = {k:i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(self.ann_file)
        
        data_infos = []       # 포맷 중립 데이터를 담을 list 객체
        for image_id in image_list:
            imgFileName = f'{image_id}.jpg'
            imgPath = osp.join(self.img_prefix, imgFileName)
            
            image = cv2.imread(imgPath)
            height, width = image.shape[:2]
            data_info = {
                'filename': imgFileName,
                'width': width, 
                'height': height
            }

            label_prefix = self.img_prefix.replace('images', 'annotations')
            xmlFileName = f'{image_id}.xml'
            annoXmlPath = osp.join(label_prefix, xmlFileName)
            if not osp.exists(annoXmlPath):
                continue
            if os.stat(annoXmlPath).st_size==0:
                continue     

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            bbox_names, bboxes = get_bboxes_from_xml(xmlFileName) 
            for bbox_name, bbox in zip(bbox_names, bboxes):
                # bbox_name이 만약 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 
                #                           그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가

                # bbox_name이 CLASSES중에 반드시 하나 있어야 함. 안 그러면 FILTERING 되므로 주의 할것. 
                if bbox_name in cat2label:
                    gt_bboxes.append(bbox)
                    gt_labels.append(cat2label[bbox_name])                      # gt_labels에는 class id를 입력
                else:
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
            
            # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값을 np.array형태로 bbox의 좌표와 label값으로 생성. 
            data_anno = {
                'bboxes':        np.array(gt_bboxes,        dtype=np.float32).reshape(-1, 4),
                'labels':        np.array(gt_labels,        dtype=np.long),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
            }
            
            data_info.update(ann=data_anno)
            data_infos.append(data_info)
            #print(data_info)
        return data_infos



# 디버깅 용도로 생성한 클래스를 생성하고 data_infos를 10개만 추출하여 생성된 데이터 확인. 
# train_ds = PetDataset(data_root='/home/oschung_skcc/git/mmdetection/data/oxford', img_prefix='images', ann_file='train.txt')
# print(train_ds.data_infos[:2])

# cd '/home/oschung_skcc/git/mmdetection/mmdet/datasets'
# ln -s /home/oschung_skcc/git/mmdetection/my/oxford_01_.py  my_dataset.py