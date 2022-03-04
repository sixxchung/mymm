
###--- DOWMLOAD IMAGE5
# !pwd
# !wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip
# !unzip kitti_tiny.zip > /dev/null

#from tabnanny import filename_only
import os
import os.path as osp
myImgPath = '/home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/image_2/'
myAnnoPath= '/home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/label_2/'

os.listdir(myImgPath)
onlyfiles = [f for f in os.listdir(myImgPath) if osp.isfile(osp.join(myImgPath, f))]
onlyfiles.sort()
onlyfiles[0:110]

imgMyPath = '/home/oschung_skcc/git/mmdetection/data/kitty_tiny/images/'
annoMyPath= '/home/oschung_skcc/git/mmdetection/data/kitty_tiny/annotations/' 

metaMyTrain = '/home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/image_2/train.txt'
metaMyValid = '/home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/label_2/val.txt'

metaTrain = '/home/oschung_skcc/git/mmdetection/data/kitty_tiny/image/train.txt'
metaValid = '/home/oschung_skcc/git/mmdetection/data/kitti_tiny/annotations/valid.txt'

ln -s /home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/image_2/ images 
ln -s /home/oschung_skcc/git/mmdetection/my/input_image/kitti_tiny/training/label_2/ annotations
 '/home/oschung_skcc/git/mmdetection/data/kitti_tiny'



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
import os.path as osp
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