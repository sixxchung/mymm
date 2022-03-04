!pwd
!wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip
!unzip kitti_tiny.zip > /dev/null

from tabnanny import filename_only
    
##---- <1> DATASET
## 1) EDA
###--- support a new dataset   
####-- 중립 데이터형태로 변환하여 메모리 로드 

import mmcv
import modelindex
# 1-1) define CLASS
CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')   # Tuple
cat2label = {k:i for i, k in enumerate(CLASSES)}      # Tuple -> Dictionary
# {'Car': 0, 'Truck': 1, 'Pedestrian': 2, 'Cyclist': 3}

# 1-2) Read Annotation
workDir = '/home/oschung_skcc/git/mmdetection/my'
ann_file = 'input_image/kitti_tiny/train.txt'
image_list = mmcv.list_from_file(osp.join(workDir, ann_file))
# annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 
# 이 self.ann_file이 load_annotations()의 인자로 입력

img_prefix = 'input_image/kitti_tiny/training/image_2' 
label_prefix = img_prefix.replace('image_2', 'label_2')
image_id = '000006'
# 1-3) example 
fileNm = '{0:}/{1:}.jpeg'.format(img_prefix, image_id)
fileNm = f'{img_prefix}/{image_id}.jpeg'
# 각 annotation 파일을 line별로 읽어 list로 로드 
lines  = mmcv.list_from_file(os.path.join(workDir, label_prefix, str(image_id)+'.txt'))

import inspect
inspect.getsource(mmcv.list_from_file)  #^+Enter
# 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. 
# content는 list의 list형태임.
# ann 정보는 numpy array로 저장되나 
# 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.  
# strip() String앞뒤의 whitespace(공백 탭 엔터) 제거
content = [line.strip().split(' ') for line in lines]            # [x[0:] for x in content]
# 오브젝트의 클래스명은 bbox_names로 저장. 
bbox_nm = [x[0] for x in content]                             # print(bbox_names)
# bbox 좌표를 저장
bbox = [ [float(info) for info in x[4:8]] for x in content]    # print(bboxes)

gt_bboxes = [];  gt_labels = []
gt_bboxes_ignore = [];  gt_labels_ignore = []
if bbox_nm in 'Car':
    gt_bboxes.append(bbox)
    # gt_labels에는 class id를 입력
    gt_labels.append(cat2label[bbox_nm])
else:
    gt_bboxes_ignore.append(bbox)
    gt_labels_ignore.append(-1)

# ==============================================================================
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import mmcv
import cv2
import numpy as np
import copy


workDir = '/home/oschung_skcc/git/mmdetection/my'
# @DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. 
@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
    def load_annotations(self, ann_file):
        print("### --- from my customconfig file ---")
        # mmdetectio framework이 Config를 기반으로 ann_file인자를 찾아 파라미터로 사용.
        print('### self.data_root :', self.data_root)
        print('### self.ann_file  :', self.ann_file)
        print('### self.img_prefix:', self.img_prefix)
        print('###      ann_file  :', ann_file)

        cat2label = {k:i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(osp.join(workDir, self.ann_file))

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
            label_prefix = self.img_prefix.replace('image_2', 'label_2')

            lines = mmcv.list_from_file(osp.join(workDir, label_prefix, str(image_id)+'.txt'))
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

# dataset과 config의 상호작용
# customDataset객체를 mm Framework에 등록 (decorator)
# > config에 인자로 입력한 설정값으로 CustomDataset 객체생성 (build Dataset)
# > 변환부분은 직접작성
# ------------------------------------------------------------------------------
# Download checkpoints from url 
# !mim download mmdet --config faster_rcnn_r50_fpn_1x_coco \
#                     --dest 'checkpoints/faster_rcnn'
# !mv ./checkpoints/faster_rcnn/*.py ./config/faster_rcnn/

### Config 설정하고 Pretrained 모델 다운로드
import os
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file     = osp.join(workDir, 'config/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
checkpoint_file = osp.join(workDir, 'checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# img = 'input_image/kitti_tiny/training/image_2/000006.jpeg'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.3)

###--- <2>Modify the config
# ------------------------------------------------------------------------------
from mmcv import Config
from mmdet.apis import set_random_seed

cfg = Config.fromfile(config_file)
# print(cfg.pretty_text)

# dataset에 대한 환경 파라미터 수정. 
# content => input_image
### self.data_root : input_image/kitti_tiny/      # 가능하면 절대경로, 마지막 dash 주의 
### (사실 label_2폴더에 있지만,) 하나의 파일로 넣기 위해 목록파일을 지정. 
### self.ann_file  : input_image/kitti_tiny/train.txt       
### self.img_prefix: input_image/kitti_tiny/training/image_2  

###      ann_file  : input_image/kitti_tiny/train.txt
workDir = '/home/oschung_skcc/git/mmdetection/my/'

cfg.dataset_type = 'KittyTinyDataset'       # 'CocoDataset' 
cfg.data_root = 'input_image/kitti_tiny/'  # 'data/coco/'

# train, val, test dataset에 대한 
# type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'KittyTinyDataset'                # 'CocoDataset' 
cfg.data.train.data_root = osp.join(workDir,'input_image/kitti_tiny/')
cfg.data.train.ann_file = 'train.txt'                   # 'data/coco/annotations/instances_train2017.json'
cfg.data.train.img_prefix = 'training/image_2'          # 'data/coco/train2017/'

cfg.data.val.type = 'KittyTinyDataset'
cfg.data.val.data_root = osp.join(workDir,'input_image/kitti_tiny/')
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'training/image_2'

cfg.data.test.type = 'KittyTinyDataset'
cfg.data.test.data_root = osp.join(workDir, 'input_image/kitti_tiny/')
cfg.data.test.ann_file = 'val.txt'
cfg.data.test.img_prefix = 'training/image_2'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 4   # 80
# pretrained 모델 (경로확인)
cfg.load_from = osp.join(workDir, 'checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = osp.join(workDir, 'tutorial_exps')

# schedule 이나 default_runtime의 설정값 수정 
# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8          # 0.02
cfg.lr_config.warmup = None          # linear
cfg.log_config.interval = 10
# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'
# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'        # bbox
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12         # 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12  #1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

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


#### 4 Inference 
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

image_file = 'input_image/kitti_tiny/training/image_2/000068.jpeg'
img = cv2.imread(image_file)

model.cfg = cfg
model = init_detector(config_file, checkpoint_file, device='cuda:0')

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)