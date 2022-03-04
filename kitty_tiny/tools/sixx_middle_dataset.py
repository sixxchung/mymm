ROOT_DIR = '/home/oschung_skcc/git/'

import copy
import os
import os.path as osp

DATA_ROOT  = osp.join(ROOT_DIR, 'my')
IMG_PREFIX = 'kitty_tiny/data/image_2' 
ANN_PREFIX = IMG_PREFIX.replace('image_2', 'label_2')

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import mmcv
import cv2
import numpy as np

@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
    
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
            filename = str(imgNm)+'.jpeg'     # 경로제외하고, image파일명만
            imgFile = f'{DATA_ROOT}/{IMG_PREFIX}/{filename}'
            # print('### sixx --- imgFile  :', imgFile) 
            image = cv2.imread(imgFile)
            height, width = image.shape[:2]  # image로부터 너비, 높이를 집접구함. 
            data_imgMeta = {
                'filename': filename,  
                'width':  width, 
                'height': height 
            }
            annFilename = str(imgNm)+'.txt'
            annlines = mmcv.list_from_file(osp.join(DATA_ROOT, ANN_PREFIX, annFilename))
            content = [line.strip().split(' ') for line in annlines]
            bbox_names = [x[0] for x in content]
            bboxes = [ [float(info) for info in x[4:8]] for x in content]
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
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

            # data_info에 data_imgMeta를 추가하여, 전체 annotation 파일들에 대한 정보를 가지는 dict생성 
            data_info.append(data_imgMeta)
        return data_info