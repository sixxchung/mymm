from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = './faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
img = '../../demo/demo.jpg'

model = init_detector(config_file, checkpoint_file, device=device)
result= inference_detector(model, img)
print(result)
show_result_pyplot(model, img, result, score_thr=0.3)

#import inspect
#inspect.getsource(show_result_pyplot)
#inspect.getsource(model.show_result)
