# @File: AdelaiDet/demo_for_MOSE/my_demo_simplified_no_args.py 
# @Author: cws 
# @Create Date: 2024/02/22
# @Desc: To infer on one seq of MOSE,
# and get the coco json, reid feat, and img output.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os

from detectron2.data.detection_utils import read_image

from demo_for_MOSE.my_predictor import VisualizationDemoReID
from adet.config import get_cfg

from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import pickle
import json
import numpy as np
from pycocotools import mask as cocomask


def deal_with_coco_data_img_one(predictions, img_path):
    coco_data_img_one = []
    instances = predictions["instances"]
    assert len(instances.pred_boxes)==len(instances.pred_masks)
    for inst_count, box_score_one in enumerate(instances.scores):
        box_score = box_score_one.cpu().numpy()
        pred_box = instances.pred_boxes[inst_count].tensor.cpu().numpy()
        pred_class = instances.pred_classes[inst_count].cpu().numpy()
        pred_mask = instances.pred_masks[inst_count].cpu().numpy()
        pred_mask = np.asfortranarray(pred_mask)
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask_rle = cocomask.encode(pred_mask)
        pred_mask_rle['counts']=pred_mask_rle['counts'].decode('UTF-8')
        inst_one = {}
        inst_one['image_id'] = Path(img_path).stem
        # map class from 0,1 to 1,2
        inst_one['category_id'] = (pred_class + 1).item()
        # box xyxy to xywh
        pred_box = pred_box.tolist()[0]
        pred_box[2] = pred_box[2] - pred_box[0]
        pred_box[3] = pred_box[3] - pred_box[1]
        inst_one['bbox'] = pred_box
        inst_one['score'] = box_score.item()
        inst_one['segmentation'] = pred_mask_rle
        coco_data_img_one.append(inst_one)
    return coco_data_img_one
    
def deal_with_reid_infer_data_img_one(predictions, reid_feats_pred, img_path):
    # copied and modified from class BDDReIDInferAndEvalInTrainClassSeven(DatasetEvaluator)
    # in /home/wensheng/code/boxmots_on_MOSE/AdelaiDet/my_code/my_evaluator.py
    # for ReID feat, need to get predicted boxes corrsponding to original img size.
    reid_infer_img_one = []
    instances = predictions["instances"]
    reid_feats_infer = reid_feats_pred["reid_feats"]
    for inst_count, reid_feats_infer_one in enumerate(reid_feats_infer):
        assert len(reid_feats_infer)==len(instances.pred_boxes),\
            "#reid_feats must be equal to #pred_boxes"
        pred_box_ori_size=instances.pred_boxes[inst_count]
        box_score=instances.scores[inst_count]
        pred_class=instances.pred_classes[inst_count]
        reid_feat=reid_feats_infer_one
        inst_one={}
        inst_one["img_info"]={"file_name": str(img_path)}
        inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
        inst_one["box_score"]=box_score.cpu().numpy()
        inst_one["pred_class"]=pred_class.cpu().numpy()
        inst_one["reid_feat"]=reid_feat.cpu().numpy()
        reid_infer_img_one.append(inst_one)
    return reid_infer_img_one

def pkl_write(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def write_json(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)
        
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # args.input = ["samples/0a7a3629"]
    # args.output = "test_out/0a7a3629"
    args.config_file = "../configs/BoxInst_ReID_One_Class_Infer_Right_Track/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k.yaml"
    args.opts = ["MODEL.WEIGHTS", "../training_dir/reid_one_class_infer_right_track/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k/BoxInst_MS_R_50_1x_kitti_mots/model_0002499.pth"]
    # args.config_file = "../configs/BoxInst_ReID_One_Class_Infer_Pair_Warp_ReID_Eval_In_Train_For_Infer/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_steps_4k_no_color_sim_pretrain_weights_v3_for_infer.yaml"
    # args.opts = ["MODEL.WEIGHTS", "../training_dir_hddd/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_steps_4k_no_color_sim_pretrain_weights_v3/BoxInst_MS_R_50_1x_kitti_mots/model_0002839.pth"]
    img_input_dir = "samples/train/0a7a3629"
    img_output_dir = img_input_dir.replace("samples", "data_out_no_pair_warp/img_out")
    reid_output_dir = img_input_dir.replace("samples", "data_out_no_pair_warp/reid_out")
    json_output_dir = img_input_dir.replace("samples", "data_out_no_pair_warp/coco_json_out")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(reid_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    cfg = setup_cfg(args)
    demo = VisualizationDemoReID(cfg)
    
    img_list = natsorted(list(Path(img_input_dir).glob("*.jpg")))
    reid_infer_seq_one = []
    coco_data_seq_one = []
    for img_one_path in tqdm(img_list):
        # use PIL, to be consistent with evaluation
        # img_one_path = "samples/train/0a7a3629/00004.jpg"
        img = read_image(img_one_path, format="BGR")
        predictions, visualized_output, reid_feature = demo.run_on_image(img)
        reid_infer_img_one = deal_with_reid_infer_data_img_one(predictions=predictions, reid_feats_pred=reid_feature, img_path=img_one_path)
        reid_infer_seq_one.extend(reid_infer_img_one)
        coco_data_img_one = deal_with_coco_data_img_one(predictions=predictions, img_path=img_one_path)
        coco_data_seq_one.extend(coco_data_img_one)
        img_outpath = os.path.join(img_output_dir, Path(img_one_path).stem+".png")
        visualized_output.save(img_outpath)
    reid_outpath = os.path.join(reid_output_dir, "reid_infer_out.pkl")
    json_outpath = os.path.join(json_output_dir, "coco_instances_results.json")
    pkl_write(reid_infer_seq_one, reid_outpath)
    write_json(coco_data_seq_one, json_outpath)
    



