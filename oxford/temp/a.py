ROOT_DIR = '/home/oschung_skcc/git'

import os
import os.path as osp
#import modelindex

WORK_DIR = osp.join(ROOT_DIR, 'mymm/oxford')

'
SSxXX

apaths=[
    '/oschung_skcc/git/mymm/oxford/data/annotations/xmls/Bombay_164.xml',


    '/home/oschung_skcc/git/mymm/oxford/data/annotations/xmls/Bengal_111.xml' ]

for i in apath:
    if not osp.exists(apath):
        print("out")
    else:
        if os.stat(apath).st_size==0:
            continue
    print("loop out")

for apth  in apaths