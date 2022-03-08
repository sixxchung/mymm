from my_dataset import MyDataset

'../mmdet/datasets/my_dataset.py'

# 디버깅 용도로 생성한 클래스를 생성하고 data_infos를 10개만 추출하여 생성된 데이터 확인. 
train_ds = PetDataset(data_root='/home/oschung_skcc/git/mmdetection/data/oxford',
                      img_prefix='images', ann_file='train.txt')
print(train_ds.data_infos[:2])
# Two ways:

# independent way:
# @DATASETS.register_module as a decorator to MyDataset in my_dataset.py
# custom_imports = dict(imports=['my_dataset'], allow_failed_imports=False) in your config file.

# Integrated way:
# @DATASETS.register_module as a decorator to MyDataset in mmdet/datasets/my_dataset.py
# from .my_dataset import MyDataset and extend __all__ with MyDataset

# __all__ ==> /home/oschung_skcc/git/mmdetection/mmdet/datasets/__init__.py


#---------------------------------------------------------------------------------
import os.path as osp
baseDirectory = "/home/oschung_skcc/git/mmdetection"

config_file     = osp.join(baseDirectory, 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
checkpoint_file = osp.join(baseDirectory, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# !wgecj -O checkpoill  httpijkdownload.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

from mmcv import Config
cfg = Config.fromfiㅐle(config_file)
print(cfg.pretty_text)

# import sys
# import numpy as np
# np.set_printoptions(threshold = sys.maxsize)
# print(cfg.pretty_text)

from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'PetDataset'
cfg.data_root = '/home/oschung_skcc/git/mmdetection/data/oxford/'#'/content/data/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'PetDataset'
cfg.data.train.data_root = '/home/oschung_skcc/git/mmdetection/data/oxford/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'images'

cfg.data.val.type = 'PetDataset'
cfg.data.val.data_root = '/home/oschung_skcc/git/mmdetection/data/oxford/'
cfg.data.val.ann_file = 'valid.txt'
cfg.data.val.img_prefix = 'images'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 37    # len(PET_CLASSES)
# pretrained 모델
cfg.load_from = checkpoint_file #'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리로 구글 Drive 설정. 
cfg.work_dir = '/home/oschung_skcc/git/mmdetection/my/pet_work_dir/'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None   # 'linear'
cfg.log_config.interval = 5
cfg.runner.max_epochs = 5

# 평가 metric 설정. 
cfg.evaluation.metric = 'mAP'
# 평가 metric 수행할 epoch interval 설정. 
cfg.evaluation.interval = 5
# 학습 iteration시마다 모델을 저장할 epoch interval 설정. 
cfg.checkpoint_config.interval = 5
# 학습 시 Batch size 설정(단일 GPU 별 Batch size로 설정됨)
cfg.data.samples_per_gpu = 4

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
cfg.lr_config.policy='step'
# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


custom_imports = dict(imports=['my_dataset'], allow_failed_imports=False)

import PetDataset
from mmdet.datasets import build_dataset
from mmdet.models   import build_detector
from mmdet.apis     import train_detector

datasets = [build_dataset(cfg.data.train)]


model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# BGR Image 사용 
img = cv2.imread('/content/data/images/Abyssinian_88.jpg')

model.cfg = cfg

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)

from mmdet.apis import show_result_pyplot

checkpoint_file = '/mydrive/pet_work_dir/epoch_5.pth'

# checkpoint 저장된 model 파일을 이용하여 모델을 생성, 이때 Config는 위에서 update된 config 사용. 
model_ckpt = init_detector(cfg, checkpoint_file, device='cuda:0')
# BGR Image 사용 
img = cv2.imread('/content/data/images/Abyssinian_88.jpg')
#model_ckpt.cfg = cfg

result = inference_detector(model_ckpt, img)
show_result_pyplot(model_ckpt, img, result, score_thr=0.3)