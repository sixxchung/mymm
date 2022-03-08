##---- <1> DATASET
## 1) EDA
###--- support a new dataset   
####-- 중립 데이터형태로 변환하여 메모리 로드 
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# !tar -xvf images.tar.gz -C /content/data
# !tar -xvf annotations.tar.gz -C /content/data

# !ls -al ~/git/mmdetection/my/input_image/oxford/images
import os
import os.path as osp
#import modelindex
ROOT_DIR = '/home/oschung_skcc/git'
WORK_DIR = osp.join(ROOT_DIR, 'mymm/oxford')
DATA_ROOT= osp.join(WORK_DIR, 'data')

IMG_PREFIX = osp.join(DATA_ROOT,'images')
ANN_PREFIX = osp.join(DATA_ROOT,'annotations','xmls')  

metaTrain0 = osp.join(DATA_ROOT, 'annotations/trainval.txt')
metaTest0  = osp.join(DATA_ROOT, 'annotations/test.txt')

metaTrain = osp.join(DATA_ROOT, 'train.txt')
metaValid = osp.join(DATA_ROOT, 'valid.txt')
metaTest  = osp.join(DATA_ROOT, 'test.txt')

config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_oxford.py')
checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

self_ann_file = osp.join(DATA_ROOT, 'train.txt')

imgNm = 'beagle_100'
imgFile = f'{IMG_PREFIX}/{imgNm}.jpg'
annFile = f'{ANN_PREFIX}/{imgNm}.txt'

# Display Image
imgFile = osp.join(IMG_PREFIX,'yorkshire_terrier_189.jpg')
from IPython.display import display, Image
display(Image(imgFile))

import cv2
import matplotlib.pyplot as plt
img = cv2.imread(imgFile)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Display Anno
from bs4 import BeautifulSoup
bs = BeautifulSoup(open(annFile), 'xml')
print(bs.prettify())






import mmcv

CLASSES = ('Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier')
cat2label = {k:i for i, k in enumerate(CLASSES)}    
imgList = mmcv.list_from_file(self_ann_file)

# annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을
# (mmdet/datasets/custom.py) 
# __init__(self, ann_file)로 입력 받고, 
# 이 self.ann_file이 load_annotations()의 인자로 입력


# 각 annotation 파일을 line별로 읽어 list로 로드 

# 전체 annlines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. 
# content는 list의 list형태임.
# ann 정보는 numpy array로 저장되나 
# 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.  
# strip() String앞뒤의 whitespace(공백 탭 엔터) 제거
annlines  = mmcv.list_from_file(osp.join(DATA_ROOT, annFile))
content = [line.strip().split(' ') for line in annlines]         # [x[0:] for x in content]
bbox_nm = [x[0] for x in content]                             # print(bbox_names)
bbox = [ [float(info) for info in x[4:8]] for x in content]   # print(bboxes)

gt_bboxes = []
gt_labels = []
gt_bboxes_ignore = []
gt_labels_ignore = []

if bbox_nm in 'Car':
    gt_bboxes.append(bbox)
    # gt_labels에는 class id를 입력
    gt_labels.append(cat2label[bbox_nm])
else:
    gt_bboxes_ignore.append(bbox)
    gt_labels_ignore.append(-1)

# ------------------------------------------------------------------------------
# Download checkpoints from url 
# !mim download mmdet --config faster_rcnn_r50_fpn_1x_coco \
#                     --dest 'checkpoints/faster_rcnn'
# !mv ./checkpoints/faster_rcnn/*.py ./config/faster_rcnn/


WORK_DIR = '/home/oschung_skcc/git/mmdetection/my/kitty_tiny'
config_file     = osp.join(WORK_DIR, 'configs/faster_rcnn_r50_fpn_1x_coco.py')
checkpoint_file = osp.join(WORK_DIR, 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')


import mmcv
mmcv.check_file_exist(config_file)




# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
#
# img = 'input_image/kitti_tiny/training/image_2/000006.jpeg'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.3)





# ==============================================================================
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import mmcv
import cv2
import numpy as np
import copy

DATA_ROOT  = '/home/oschung_skcc/git/mmdetection/my'
IMG_PREFIX = 'kitti_tiny/data/image_2' 
ANN_PREFIX = IMG_PREFIX.replace('image_2', 'label_2')

# @DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. 
@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
    CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
    def load_annotations(self, ann_file):
        # mmdetection프레임웍이 Config에서 ann_file인자를 찾아 파라미터로 사용.
        print("### --- from my customconfig file ---")
        print('### self.DATA_ROOT :', self.DATA_ROOT)
        print('### self.ann_file  :', self.ann_file)
        print('### self.IMG_PREFIX:', self.IMG_PREFIX)
        print('###      ann_file  :', ann_file)

        cat2label = {k:i for i, k in enumerate(self.CLASSES)}
        imgList = mmcv.list_from_file(osp.join(DATA_ROOT, self.ann_file))

        data_info = []    # 포맷 중립 데이터를 담을 list 객체
        for imgNm in imgList:
            # 개별 image에 대한 메타정보 및  annotation 정보 저장용
            filename = str(imgNm)+'.jpeg'     # 경로제외하고, image파일명만
            imgFile = f'{IMG_PREFIX}/{filename}'
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

