from xml.dom import minidom
import json
import os   
from argparse import ArgumentParser

parser = ArgumentParser()
### Required
parser.add_argument('target_json', help='filepath of target coco(json)')
parser.add_argument('output_cvat', help='filepath of output cvat(xml)')

args = parser.parse_args()
target_coco = args.target_json
output_cvat = args.output_cvat

print(f"convert {target_coco} to {output_cvat}")

xml_doc = minidom.Document()

xml_root = xml_doc.createElement('annotations') 
xml_doc.appendChild(xml_root)

# with open('./study/Python/coco-annotation.json', 'r') as f:
with open(target_coco, 'r') as f:
    cocoset = json.load(f)

lst_img = cocoset["images"]

for i in range(0,len(lst_img)):
    img_id = str(lst_img[i]['id'])
    img_name = str(lst_img[i]['file_name'])    
    
    xml_image = xml_doc.createElement('image')
    xml_image.setAttribute('id', img_id)
    xml_image.setAttribute('name', img_name)

    xml_root.appendChild(xml_image)

xml_output = xml_root.toprettyxml(indent ="\t") 
# print(xml_output)

lst_anno = cocoset["annotations"]
label="Cow"
occluded="0"

num_keypoint = 20
for anno in range(1,len(lst_anno)):
    image_id = str(lst_anno[anno]['image_id'])
    points = lst_anno[anno]['keypoints']
    
    el_image = xml_root.getElementsByTagName('image')
    
    for el in el_image:
        if el.getAttribute('id') == image_id:            
            str_points = ""
            for p in range(0,60,3):
                str_points += str(points[p])+","+str(points[p+1])+";"            
            xml_points = xml_doc.createElement('points')
            xml_points.setAttribute('label', label)
            xml_points.setAttribute('occluded', occluded)
            xml_points.setAttribute('points', str_points[0:-1])
            el.appendChild(xml_points)            
            xml_output = xml_root.toprettyxml(indent ="\t") 

with open(output_cvat, 'w') as f:
    f.write(xml_root.toxml())
