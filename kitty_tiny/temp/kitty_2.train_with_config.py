# dataset과 config의 상호작용
# customDataset객체를 mm Framework에 등록 (decorator)
# > config에 인자로 입력한 설정값으로 CustomDataset 객체생성 (build Dataset)
# > 변환부분은 직접작성
### Config 설정하고 Pretrained 모델 다운로드
import os
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

workDir = '/home/oschung_skcc/git/mmdetection/my/'
config_file     = osp.join(workDir, 'config/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
checkpoint_file = osp.join(workDir, 'checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
###--- <2>Modify the config
# ------------------------------------------------------------------------------
import mmcv
from os.path import osp
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