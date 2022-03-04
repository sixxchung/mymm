import xml.etree.ElementTree as ET

# annotation xml 파일 파싱해서 bbox정보 추출
def get_bboxes_from_xml(xml_file):
  # xml_file = '/home/oschung_skcc/git/mmdetection/data/oxford/annotations/Abyssinian_16.xml'
  tree = ET.parse(xml_file)
  root = tree.getroot()
  bbox_names = []
  bboxes = []
  # 파일내에 있는 모든 object Element를 찾음. 
  for obj in root.findall('object'):
    # obj = root.findall('object')[0] 
    bbox_name = obj.find('name').text
    xmlbox = obj.find('bndbox')
    x1 = int(xmlbox.find('xmin').text) 
    y1 = int(xmlbox.find('ymin').text)
    x2 = int(xmlbox.find('xmax').text)
    y2 = int(xmlbox.find('ymax').text)

    bbox_names.append(bbox_name)
    bboxes.append([x1, y1, x2, y2])

  return bbox_names, bboxes



import os.path as osp
#import copy
import mmcv
import numpy as np
import cv2
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom  import CustomDataset

import xml.etree.ElementTree as ET

metaMyTrain = '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/trainval.txt'
import pandas as pd
pet_df = pd.read_csv(metaMyTrain, sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
pet_df['class_name'] = pet_df['img_name'].apply(lambda x: x[:x.rfind('_')])
PET_CLASSES = pet_df['class_name'].unique().tolist()
# 디버깅 용도로 CustomDataset을 흉내낸 클래스 생성. 
class PetDataset_imsi():
  CLASSES = PET_CLASSES
  

  
  
  
  
  # 생성자 함수 생성. 
  #def __init__(self, data_root, ann_file, img_prefix):
    data_root = data_root
    ann_file = osp.join(data_root, ann_file)
    img_prefix = osp.join(data_root, img_prefix)
      
    data_infos = load_annotations(ann_file)

  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 
  # 이 self.ann_file이 load_annotations()의 인자로 입력
  def load_annotations(self, ann_file):
    cat2label = {k:i for i, k in enumerate(CLASSES)}
    image_list = mmcv.list_from_file(ann_file)
    # 포맷 중립 데이터를 담을 list 객체
    data_infos = []

    for image_id in image_list:
      # self.img_prefix는 images 가 입력될 것임. 
      image_id = image_list[6]
      imgFilename = f'{image_id}.jpg'
      imgPath = osp.join(img_prefix, imgFilename)
      
      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함. 
      image = cv2.imread(imgPath)
      height, width = image.shape[:2]
      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename에는 image의 파일명만 들어감(디렉토리는 제외)
      # 영상에는 data_info = {'filename': filename 으로 되어 있으나 filename은 image 파일명만 들어가는게 맞음. 
      data_info = {
          'filename': imgFilename, #str(image_id) + '.jpg', 
          'width': width, 
          'height': height
      }

      # 개별 annotation XML 파일이 있는 서브 디렉토리의 prefix 변환. 
      label_prefix = img_prefix.replace('images', 'annotations')

      # 개별 annotation XML 파일을 1개 line 씩 읽어서 list 로드. annotation XML파일이 xmls 밑에 있음에 유의
      #anno_xml_file = osp.join(label_prefix, 'xmls/'+str(image_id)+'.xml')
      xmlFileName = f'{image_id}.xml'
      anno_xml_file = osp.join(label_prefix, xmlFileName) 
      # 메타 파일에는 이름이 있으나 실제로는 존재하지 않는 XML이 있으므로 이는 제외. 
      if not osp.exists(anno_xml_file):
          continue
      
      # get_bboxes_from_xml() 를 이용하여 개별 XML 파일에 있는 이미지의 모든 bbox 정보를 list 객체로 생성. 
      # anno_dir = osp.join(label_prefix, 'xmls')
      anno_dir = label_prefix
      #bbox_names, bboxes = get_bboxes_from_xml(anno_dir, str(image_id)+'.xml')
      bbox_names, bboxes = get_bboxes_from_xml(anno_dir, xmlFileName)
      #print('#########:', bbox_names)
                  
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []
        
      # bbox별 Object들의 class name을 class id로 매핑. class id는 tuple(list)형의 CLASSES의 index값에 따라 설정
      for bbox_name, bbox in zip(bbox_names, bboxes):
        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        # bbox_name이 CLASSES중에 반드시 하나 있어야 함. 안 그러면 FILTERING 되므로 주의 할것. 
        if bbox_name in cat2label:
            gt_bboxes.append(bbox)
            # gt_labels에는 class id를 입력
            gt_labels.append(cat2label[bbox_name])
        else:
            gt_bboxes_ignore.append(bbox)
            gt_labels_ignore.append(-1)
      
      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값을 np.array형태로 bbox의 좌표와 label값으로 생성. 
      data_anno = {
        'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.array(gt_labels, dtype=np.long),
        'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
        'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      
      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
      data_info.update(ann=data_anno)
      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)
      #print(data_info)

    return data_infos


train_ds = PetDataset_imsi(data_root='/home/oschung_skcc/git/mmdetection/data/oxford',
 ann_file='train.txt', img_prefix='images')
print(train_ds.data_infos[:10])