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

##EDA
image_file = osp.join(IMG_PREFIX,'yorkshire_terrier_189.jpg')
from IPython.display import display, Image
display(Image(image_file))

xml_file =  osp.join(ANN_PREFIX,'yorkshire_terrier_189.xml')
from bs4 import BeautifulSoup
bs = BeautifulSoup(open(xml_file), 'xml')
print(bs.prettify())

files = os.listdir(IMG_PREFIX)
# [file for file in files]   # same above

# for file in files:         # same above
#     print(file)
# for (root,dirs, files) in os.walk(IMG_PREFIX):
#     print(files)

#onlyFiles = [file for file in files if osp.isfile(osp.join(IMG_PREFIX, file)) ]
#onlyFiles.sort()

import re
import numpy as np
CLASSES = []
for file in files:
    result = re.sub("[0-9]", "", file)
    result = re.sub("_.jpg", "", result)
    if result not in CLASSES:
        CLASSES.append(result)
print(CLASSES)
# tuple(CLASSES)


import pandas as pd
pet_train_df = pd.read_csv(metaTrain0, sep=" ", header=None, 
                     names=['imgNm', 'class_id', 'etc1', 'etc2'] )
pet_train_df.head()
pet_train_df['class_name'] = pet_train_df['imgNm'].apply(lambda x: x[:x.rfind('_')])
PET_CLASSES = pet_train_df['class_name'].unique().tolist()
PET_CLASSES.sort()
print(PET_CLASSES)

pet_test_df = pd.read_csv(metaTest0, sep=" ", header=None, 
                     names=['imgNm', 'class_id', 'etc1', 'etc2'] ) 
pet_test_df.head()

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(pet_train_df, 
                        test_size=0.1, stratify=pet_train_df['class_id'], random_state=2021)
# train_df['class_id'].value_counts()
# valid_df['class_id'].value_counts()
train_df = train_df.sort_values(by='imgNm')
valid_df = valid_df.sort_values(by='imgNm')
test_df  = pet_test_df.sort_values(by='imgNm')


### SAVE Files : train.txt   valid.txt    test.txt
train_df['imgNm'].to_csv(metaTrain, sep='\n', index=False, header=None)
valid_df['imgNm'].to_csv(metaValid, sep='\n', index=False, header=None)
test_df['imgNm'].to_csv(metaTest,   sep='\n', index=False, header=None)
