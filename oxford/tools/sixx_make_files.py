ROOT_DIR = '/home/oschung_skcc/git'
import os
import os.path as osp
#import modelindex
WORK_DIR = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )  #'/home/oschung_skcc/git/mymm/oxford'

DATA_DIR= osp.join(WORK_DIR, 'data')
IMG_PREFIX = 'images'
ANN_PREFIX = 'annotations/xmls'  

metaTrain0 = osp.join(DATA_DIR, ANN_PREFIX, 'trainval.txt')
metaTest0  = osp.join(DATA_DIR, ANN_PREFIX, 'test.txt')

metaTrain = osp.join(DATA_DIR, 'train.txt')
metaValid = osp.join(DATA_DIR, 'valid.txt')
metaTest  = osp.join(DATA_DIR, 'test.txt')

##EDA
imgPath = osp.join(DATA_DIR, IMG_PREFIX,'yorkshire_terrier_189.jpg')
from IPython.display import display, Image
display(Image(imgPath))

xml_file =  osp.join(DATA_DIR, ANN_PREFIX,'yorkshire_terrier_189.xml')
from bs4 import BeautifulSoup
bs = BeautifulSoup(open(xml_file), 'xml')
print(bs.prettify())

files = os.listdir(osp.join(DATA_DIR,IMG_PREFIX))
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
pet_df_train = pd.read_csv(metaTrain0, sep=" ", header=None, 
                     names=['imgNm', 'class_id', 'etc1', 'etc2'] )
pet_df_train.head()
pet_df_train['class_name'] = pet_df_train['imgNm'].apply(lambda x: x[:x.rfind('_')])
PET_CLASSES = pet_df_train['class_name'].unique().tolist()
PET_CLASSES.sort()
print(PET_CLASSES)

pet_df_test = pd.read_csv(metaTest0, sep=" ", header=None, 
                     names=['imgNm', 'class_id', 'etc1', 'etc2'] ) 
pet_df_test.head()

from sklearn.model_selection import train_test_split
trainDf, validDf = train_test_split(pet_df_train, 
                        test_size=0.1, stratify=pet_df_train['class_id'], random_state=2021)
# train_df['class_id'].value_counts()
# valid_df['class_id'].value_counts()
trainDf = trainDf.sort_values(by='imgNm')
validDf = validDf.sort_values(by='imgNm')
testDf  = pet_df_test.sort_values(by='imgNm')

### SAVE Files : train.txt   valid.txt    test.txt
trainDf['imgNm'].to_csv(metaTrain, sep='\n', index=False, header=None)
validDf['imgNm'].to_csv(metaValid, sep='\n', index=False, header=None)
testDf['imgNm'].to_csv(metaTest,   sep='\n', index=False, header=None)
