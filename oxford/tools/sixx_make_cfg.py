ROOT_DIR = '/home/oschung_skcc/git'
import os
import os.path as osp
WORK_DIR = osp.dirname( osp.dirname(osp.realpath(__file__)) ) #'mymm/oxford'

from mmcv import Config
cfg = Config.fromfile(osp.join(ROOT_DIR, "mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"))

config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_oxford.py')
checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

from mmdet.apis import set_random_seed

cfg.dataset_type = 'OxfordDataset'       # 'CocoDataset' 
cfg.data_root = WORK_DIR                 # 'data/coco/'

# train, val, test dataset에 대한 
# type, WORK_DIR, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'OxfordDataset'             # 'CocoDataset' 
cfg.data.train.data_root = WORK_DIR
cfg.data.train.ann_file = 'data/train.txt'    # 'data/coco/annotations/instances_train2017.json'
cfg.data.train.img_prefix = 'data/images'  # 'data/coco/train2017/'

cfg.data.val.type = 'OxfordDataset'
cfg.data.val.data_root = WORK_DIR
cfg.data.val.ann_file = 'data/valid.txt'
cfg.data.val.img_prefix = 'data/images'

cfg.data.test.type = 'OxfordDataset'
cfg.data.test.data_root = WORK_DIR
cfg.data.test.ann_file = 'data/test.txt'
cfg.data.test.img_prefix = 'data/images'

cfg.model.roi_head.bbox_head.num_classes = 37        #  class의 80 갯수 수정. 
cfg.load_from = checkpoint_file                      # pretrained 모델 (경로확인)
cfg.work_dir = osp.join(WORK_DIR, 'tutorial_exps')   # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 

# schedule 이나 default_runtime의 설정값 수정 
# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8          # 0.02
cfg.lr_config.warmup = None          # linear
cfg.log_config.interval = 12

cfg.runner.max_epochs = 12

# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'
# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'        # bbox
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12         # 1

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12  #1
cfg.data.sample_per_gpu = 4

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# print(f'Config:\n{cfg.pretty_text}')

###############################################################################
with open (config_file, 'w') as f:
    print(cfg.pretty_text, file=f)

import datetime
print(f"---created custom config file by sixx on {datetime.datetime.now()}")
print(f"{cfg.dataset_type} :: {osp.relpath(config_file)}")
