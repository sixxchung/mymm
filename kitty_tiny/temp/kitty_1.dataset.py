##---- <1> DATASET  중립 데이터형태로 변환하여 메모리 로드
import copy
import numpy as np
import os
import os.path as osp

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

import mmcv
import cv2

@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
    
    def load_annotations(self, ann_file):
        print("### --- from my customconfig file ---")
        print('### self.data_root :', self.data_root)
        print('### self.ann_file  :', self.ann_file)
        print('### self.img_prefix:', self.img_prefix)
        print('###      ann_file  :', ann_file)

        cat2label = {k:i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(osp.join(self.ann_file))

        data_info = []    # 포맷 중립 데이터를 담을 list 객체
        for image_id in image_list:
            filename = '{0:}/{1:}.jpeg'.format(self.img_prefix, image_id)
            image = cv2.imread(filename)
            height, width = image.shape[:2]  # image로부터 너비, 높이를 집접구함. 
            # 개별 image에 대한 메타정보 및  annotation 정보 저장용
            data_imgMeta = {
                'filename': str(image_id)+'.jpeg',  # 경로제외하고, image파일명만
                'width':  width, 
                'height': height 
            }
            label_prefix = self.img_prefix.replace('image', 'annotations')

            lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt'))
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [ [float(info) for info in x[4:8]] for x in content]

            # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'는 ignore로 별도 저장.
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
            
            # data_imgMeta < image의 메타정보>에 data_anno 정보 추가 ('ann' key값으로 value값은 모두 np.array임)
            data_anno = {
                'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                'labels': np.array(gt_labels, dtype=np.compat.long),
                'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array(gt_labels_ignore, dtype=np.compat.long)
            }
            data_imgMeta.update(ann=data_anno)

            # data_info에 data_imgMeta를 추가하여, 전체 annotation 파일들에 대한 정보를 가지는 dict생성 
            data_info.append(data_imgMeta)
        return data_info


### -------------------------------------------------------------------------------------
config_file = '/home/oschung_skcc/git/mmdetection/configs/_sixx/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/home/oschung_skcc/git/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

from mmcv import Config
cfg = Config.fromfile(config_file)
# print(cfg.pretty_text)

from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'KittyTinyDataset'
cfg.data_root = '/home/oschung_skcc/git/mmdetection/data/kitti_tiny/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'KittyTinyDataset'
cfg.data.train.data_root = '/home/oschung_skcc/git/mmdetection/data/kitti_tiny'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'images'

cfg.data.val.type = 'KittyTinyDataset'
cfg.data.val.data_root = '/home/oschung_skcc/git/mmdetection/data/kitti_tiny'
cfg.data.val.ann_file = 'valid.txt'
cfg.data.val.img_prefix = 'images'

cfg.data.test.type = 'KittyTinyDataset'
cfg.data.test.data_root = '/home/oschung_skcc/git/mmdetection/data/kitti_tiny'
cfg.data.test.ann_file = 'val.txt'
cfg.data.test.img_prefix = 'images'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 4
# pretrained 모델
cfg.load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = '../checkpoints/kitty_tutorial_exps'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')



#####----------
# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')


###--- <3> Train new detector
### 학습 수행 
from mmdet.datasets import build_dataset
from mmdet.models   import build_detector
from mmdet.apis     import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]
# print(datasets)
# datasets[0].CLASSES

model = build_detector(
    cfg.model, 
    train_cfg = cfg.get('train_cfg'),
    test_cfg  = cfg.get('test_cfg')
)
model.CLASSES = datasets[0].CLASSES

# 주의, config에 pretrained 모델 지정이 상대 경로로 설정됨 
# cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# 아래와 같이 %cd mmdetection 지정 필요. 
# %cd mmdetection 
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)