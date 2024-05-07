# @File: SSIS/demo/demo_my_data_for_MOSE.py 
# @Author: cws 
# @Create Date: 2024/02/26
# @Desc: Get shadow det result for MOSE.
# Note that each seq has a json result,
# not all seqs have one json result like kitti/bdd.

import argparse
import multiprocessing as mp
import os
import cv2

from predictor import VisualizationDemo
from adet.config import get_cfg
import torch

import json
import numpy as np
import pycocotools.mask as mask_util
from natsort import os_sorted
from tqdm import tqdm
from pathlib import Path


def write_json(data, filepath):
    with open(filepath,'w') as f:
        json.dump(data,f,indent=2)

def get_img_my_id_from_filename(file_name):
    file_name_parts = file_name.split('/')
    img_my_id = int(file_name_parts[-2])*10000 + \
        int(file_name_parts[-1].split('.')[-2])
    return img_my_id

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

def instance_to_coco_json_result_MOSE(instances, img_path):
    pred_masks = np.asfortranarray(np.transpose(
        instances.pred_masks, (1, 2, 0)).astype(np.uint8))
    pred_boxes = instances.pred_boxes
    masks_rle = mask_util.encode(pred_masks)
    img_id = Path(img_path).stem
    result_img_one = []
    
    for ctr, mask_rle_one in enumerate(masks_rle):
        # convert type bytes to string.
        counts_one = mask_rle_one['counts'].decode('UTF-8')
        size_one = mask_rle_one['size']
        seg_one = {'size':size_one,'counts':counts_one}
        # xyxy to xywh.
        # need to change np.float32 to float,
        # so json can do serialization.
        box_one = [float(pred_boxes.tensor[ctr][0]), float(pred_boxes.tensor[ctr][1]),
                   float(pred_boxes.tensor[ctr][2]-pred_boxes.tensor[ctr][0]),
                   float(pred_boxes.tensor[ctr][3]-pred_boxes.tensor[ctr][1])]
        score_one = float(instances.scores[ctr])
        class_one = instances.pred_classes[ctr]
        result_inst_one = {'image_id':img_id,'category_id':class_one,'bbox':box_one,
                           'score':score_one,'segmentation':seg_one,}
        result_img_one.append(result_inst_one)
    return result_img_one

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
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
    args.config_file = "../configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml"
    args.input = "../my_dataset_for_MOSE/MOSE_train_car_seq"
    img_out_dir = "my_vis_result_MOSE/MOSE_train_car_seq/"
    json_out_dir = "my_json_result_MOSE/MOSE_train_car_seq/"
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)

    cfg = setup_cfg(args)
    img_dir_list = [os.path.join(args.input, path) 
                  for path in os.listdir(args.input) if not path.startswith('.')]
    img_dir_list = os_sorted(img_dir_list)
    
    demo = VisualizationDemo(cfg)    
    for img_dir_one in tqdm(img_dir_list):
        imgs_seq_one = [os.path.join(img_dir_one, 'img1', path) 
                        for path in os.listdir(img_dir_one+'/img1') if not path.startswith('.')]
        imgs_seq_one = os_sorted(imgs_seq_one)
        
        reqult_seq_one = []
        seq_id = Path(img_dir_one).stem
        img_out_seq_dir = os.path.join(img_out_dir, seq_id)
        os.makedirs(img_out_seq_dir,exist_ok=True)
        for path in tqdm(imgs_seq_one, leave=False):
            img = cv2.imread(path)
            with torch.no_grad():
                instances, visualized_output = demo.run_on_image(img)

            if instances is None:
                # this means no object/shadow found in the img.
                continue
            result_img_one = instance_to_coco_json_result_MOSE(instances, path)
            reqult_seq_one.extend(result_img_one)
            
            # below output img.
            out_filename = os.path.join(img_out_seq_dir, Path(path).name)
            visualized_output.save(out_filename)
        # below output json.
        json_out_seq_dir = os.path.join(json_out_dir, seq_id)
        os.makedirs(json_out_seq_dir, exist_ok=True)
        json_out_seq_path = os.path.join(json_out_seq_dir, 'shadow_instance_result.json')
        write_json(reqult_seq_one, json_out_seq_path)
    print('kk')
