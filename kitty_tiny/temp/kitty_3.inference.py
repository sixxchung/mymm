#### 4 Inference 
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

image_file = 'input_image/kitti_tiny/training/image_2/000068.jpeg'
img = cv2.imread(image_file)

model.cfg = cfg
model = init_detector(config_file, checkpoint_file, device='cuda:0')

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)