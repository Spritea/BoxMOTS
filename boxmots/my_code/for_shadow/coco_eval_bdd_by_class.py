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
# below is bdd.
# result_file='../../my_shadow_det/filter_result/bdd_data/boxinst/COCO_pretrain_strong/bs4/inference_final/by_class/shadow_filter_result_rate_02_car_truck_bus.json'
# 8 class means mask ap with car shadow removal:
# 17.835->18.066.
# 8 class means mask ap with all class shadow removal:
# 17.835->18.129.
# 8 class means mask ap with car+truck+bus shadow removal:
# 17.835->18.296
# car mask ap: 38.968->40.582
# ped mask ap: 21.524->21.498 bad.
# rider mask ap: 3.865->3.871
# truck mask ap: 23.649->24.717
# bus mask ap: 26.894->27.439
# motorcycle mask ap: 0.004->0.004
# bicycle mask ap: 9.941->8.794 bad
# below is bdd in reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train
# 8 class means mask ap with car+truck+bus shadow removal:
# 18.273->18.560
# car mask ap: 40.620->41.925
# ped mask ap: 21.940 
# rider mask ap: 3.742
# truck mask ap: 24.903->25.527
# bus mask ap: 27.308->27.389
# motorcycle mask ap: 0.000
# bicycle mask ap: 9.399
# below is bdd in reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train with pretrain weights.
# 8 class means mask ap with car+truck+bus shadow removal:
# 18.716->19.081
# car mask ap: 40.704->42.325
# ped mask ap: 21.951 
# rider mask ap: 3.918
# truck mask ap: 24.734->25.595
# bus mask ap: 30.597->30.673
# motorcycle mask ap: 0.001
# bicycle mask ap: 9.103
#标注文件的路径及文件名，json文件形式
# cocoGt=COCO('../../my_shadow_det/original_boxinst_result/val_label_gt/kitti_mots/val_in_trainval_gt_as_coco_instances.json')
cocoGt=COCO('../../my_shadow_det/original_boxinst_result/val_label_gt/bdd_data/val_seg_track.json')
#自己的生成的结果的路径及文件名，json文件形式
# result_file='../../my_shadow_det/origial_boxinst_result/inference/iter_0003999/coco_instances_results.json'
result_file='../../my_shadow_det/filter_result/bdd_data/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots/inference/iter_0014999/by_class/shadow_filter_result_rate_02_car_truck.json'
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
cat_dict={'pedestrian':1,'rider':2,'car':3,'truck':4,'bus':5,'motorcycle':7,'bicycle':8}
for cat in cat_dict.keys():
    print(f'{cat} ap:')
    coco_eval.params.catIds=cat_dict[cat]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()