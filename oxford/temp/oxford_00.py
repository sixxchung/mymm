

#from asyncio import subprocess
import os
import os.path as osp

#IMG_PREFIX = '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/images/'
#ANN_PREFIX = '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/xmls/'

os.listdir(IMG_PREFIX)
onlyfiles = [f for f in os.listdir(IMG_PREFIX) if osp.isfile(osp.join(IMG_PREFIX, f))]
onlyfiles.sort()
onlyfiles[0:110]

IMG_PREFIX = '/home/oschung_skcc/git/mmdetection/data/oxford/images/'
ANN_PREFIX = '/home/oschung_skcc/git/mmdetection/data/oxford/annotations/' 

#######
# import  re
# import numpy as np
# results = []
# for i in onlyfiles:
#     result = re.sub("[0-9]", "", i)
#     result = re.sub("_.jpg", "", result)
#     if result not in results:
#         results.append(result)
# print(results)

# ~/git/mmdetection/data/oxford $ 
# cd ~/git/mmdetection/data/oxford
# ln -s  /home/oschung_skcc/git/mmdetection/my/input_image/oxford/images  images
# ln -s  /home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/xmls annotations
# os.makedirs(IMG_PREFIX, exist_ok=True)
# cmd = 'ln -s '+IMG_PREFIX+' '+IMG_PREFIX
# os.system(cmd)
########


image_file = osp.join(IMG_PREFIX,'yorkshire_terrier_189.jpg')
#image_file = osp.realpath(image_file)
from IPython.display import display, Image
display(Image(image_file))

import cv2
import matplotlib.pyplot as plt
img = cv2.imread(image_file)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

'/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/xmls/yorkshire_terrier_96.xml'

xml_file =  osp.join(ANN_PREFIX+'yorkshire_terrier_189.xml')

from bs4 import BeautifulSoup
bs = BeautifulSoup(open(xml_file), 'xml')
pretty_xml = bs.prettify()
print(pretty_xml)

###########################################################################
metaTrain0 = '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/trainval.txt'
metaTest0  = '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/test.txt'

import pandas as pd
pet_df = pd.read_csv(metaTrain0, sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
pet_df['class_name'] = pet_df['img_name'].apply(lambda x: x[:x.rfind('_')])
pet_df.head()
# pet_df['class_id'].value_counts()
PET_CLASSES = pet_df['class_name'].unique().tolist()
print(PET_CLASSES)

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(pet_df, 
                        test_size=0.1, stratify=pet_df['class_id'], random_state=2021)
# train_df['class_id'].value_counts()
# valid_df['class_id'].value_counts()
train_df = train_df.sort_values(by='img_name')
valid_df = valid_df.sort_values(by='img_name')

metaTrain = '/home/oschung_skcc/git/mmdetection/data/oxford/train.txt'
metaValid = '/home/oschung_skcc/git/mmdetection/data/oxford/valid.txt'
#ann_file??? ???????????? ??????????????? ??????????????? ?????????????????? ?????? ?????? ??????????????? ???????????? ?????? ?????????. 
train_df['img_name'].to_csv(metaTrain, sep=' ', header=False, index=False)
valid_df['img_name'].to_csv(metaValid, sep=' ', header=False, index=False)

print( open(metaTrain, 'r').read() )


##################################################
#import glob
import xml.etree.ElementTree as ET
# annotation xml ?????? ???????????? bbox?????? ??????
def get_bboxes_from_xml_test(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbox_names = []
    bboxes = []
    # ???????????? ?????? ?????? object Element??? ??????. 
    for obj in root.findall('object'):

        bbox_name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)

        bbox_names.append(bbox_name)
        bboxes.append([x1, y1, x2, y2])
    return bbox_names, bboxes

get_bboxes_from_xml_test(xml_file)

# 1?????? annotation ???????????? bbox ?????? ??????. ???????????? object??? ?????? ?????? ?????? object??? name??? bbox ???????????? list??? ??????.
def get_bboxes_from_xml(anno_dir, xml_file):
    anno_xml_file = osp.join(anno_dir, xml_file)

    tree = ET.parse(anno_xml_file)
    root = tree.getroot()
    bbox_names = []
    bboxes = []
    # ???????????? ?????? ?????? object Element??? ??????. 
    for obj in root.findall('object'):
        #obj.find('name').text??? cat ?????? dog??? ??????     
        #bbox_name = obj.find('name').text
        # object??? ??????????????? ??????????????? ??????. 
        bbox_name = xml_file[:xml_file.rfind('_')]

        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)

        bbox_names.append(bbox_name)
        bboxes.append([x1, y1, x2, y2])
    return bbox_names, bboxes

