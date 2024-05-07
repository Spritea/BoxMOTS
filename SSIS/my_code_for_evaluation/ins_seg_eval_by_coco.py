from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from my_cocoeval import COCOeval
from pathlib import Path
import json

# my_track_result/use_grounding_dino_det_use_sam/kitti_mots/seg_track_in_coco_json/seg_track_result.json
# mask-car AP: 61.8, AP50: 76.2
# mask-ped AP: 42.2, AP50: 67.3
# box-car AP: 63.7, AP50: 76.2
# box-ped AP: 55.2, AP50: 69.9

def main():
    cocoGt = COCO("my_dataset/KITTI_MOTS/annotations/val_in_trainval_gt_as_coco_instances.json")
    result_file = "../my_code_for_add_noise_to_shadow_det/my_shadow_det/combine_noisy_shadow_with_boxmots_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_ckpt_500_steps_4k/inference/iter_0002839/shadow_filter_result_rate_02_erode_kernel_11_iter_1.json"
    # a = cocoGt.getImgIds()
    # b = cocoGt.getCatIds()
    # with open(result_file) as f:
    #     anns = json.load(f)
    # annsImgIds = [ann['image_id'] for ann in anns]
    cocoDt = cocoGt.loadRes(result_file)
    # coco_eval = COCOeval(cocoGt, cocoDt, "bbox")
    coco_eval = COCOeval(cocoGt, cocoDt, "segm")
    # below choose one category to evaluate.
    # coco_eval.params.catIds = 3
    # below set max number of detections to be large enough.
    # the maxDeys is same with mmdet codebase.
    coco_eval.params.maxDets = [100, 300, 1000]
    print('total ap:')
    # coco_eval.params.catIds=1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('car ap:')
    coco_eval.params.catIds = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('ped ap:')
    coco_eval.params.catIds = 2
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # note the AP small/medium/large is different from the one on detectron2.


if __name__ == "__main__":
    main()