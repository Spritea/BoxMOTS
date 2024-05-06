from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path

# class: 1-object,2-shadow.
# original boxinst:mask ap:44.8 (car:0.601,ped:0.295)
# rate 0.1: mask ap:45.8 (car:0.622,ped:0.295)
# rate 0.2: mask ap:46.3 (car:0.631,ped:0.295)
# rate 0.3: mask ap:46.2 (car:0.630,ped:0.295)
# rate 0.4: mask ap:46.2 (car:0.629,ped:0.295)
# rate 0.5: mask ap:46.2 (car:0.628,ped:0.295)
# rate 0.6: mask ap:46.1 (car:0.627,ped:0.295)
# rate 0.7: mask ap:46.0 (car:0.626,ped:0.295)
# rate 0.8: mask ap:46.0 (car:0.626,ped:0.295)
# rate 0.9: mask ap:46.0 (car:0.625,ped:0.295)
# use object: mask ap:46.1 (car:0.627,ped:0.295)
#标注文件的路径及文件名，json文件形式
cocoGt=COCO('../../my_shadow_det/original_boxinst_result/val_label_gt/kitti_mots/val_in_trainval_gt_as_coco_instances.json')
# cocoGt = COCO('to_coco/test/insect-3.json')
#自己的生成的结果的路径及文件名，json文件形式
# result_file='../../my_shadow_det/origial_boxinst_result/inference/iter_0003999/coco_instances_results.json'
result_file="../../my_shadow_det/filter_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/imagenet_pretrain/long_epoch_seq_shuffle_fl_2_lr_0_01_bs_4_eval_500_no_color_sim_pretrain_weights_v2/iter_0008499/shadow_filter_result_rate_02.json"
cocoDt = cocoGt.loadRes(result_file)
# coco_eval = COCOeval(cocoGt, cocoDt, "bbox")
coco_eval = COCOeval(cocoGt, cocoDt, "segm")
# below choose one category to evaluate.
# coco_eval.params.catIds = 3
# below set max number of detections to be large enough.
# the maxDeys is same with mmdet codebase.
# changed at 2021/12/27.
coco_eval.params.maxDets=[100,300,1000]
print('total ap:')
# coco_eval.params.catIds=1
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
print('car ap:')
coco_eval.params.catIds=1
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
print('ped ap:')
coco_eval.params.catIds=2
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
# note the AP small/medium/large is different from the one on detectron2.