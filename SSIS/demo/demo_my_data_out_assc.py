# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import skimage.io as io

# from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
import torch

import json
import numpy as np
import pycocotools.mask as mask_util

# constants
WINDOW_NAME = "COCO detections"


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


def instance_to_coco_json_result(instances, img_path):
    pred_masks = np.asfortranarray(np.transpose(
        instances.pred_masks, (1, 2, 0)).astype(np.uint8))
    pred_boxes = instances.pred_boxes
    masks_rle = mask_util.encode(pred_masks)
    img_id = get_img_my_id_from_filename(img_path)
    pred_associations = instances.pred_associations
    result_img_one = []
    
    for ctr in range(len(masks_rle)):
        # convert type bytes to string.
        counts_one = masks_rle[ctr]['counts'].decode('UTF-8')
        size_one = masks_rle[ctr]['size']
        seg_one = {'size':size_one,'counts':counts_one}
        # xyxy to xywh.
        # need to change np.float32 to float,
        # so json can do serialization.
        box_one = [float(pred_boxes.tensor[ctr][0]), float(pred_boxes.tensor[ctr][1]),
                   float(pred_boxes.tensor[ctr][2]-pred_boxes.tensor[ctr][0]),
                   float(pred_boxes.tensor[ctr][3]-pred_boxes.tensor[ctr][1])]
        score_one = float(instances.scores[ctr])
        class_one = instances.pred_classes[ctr]
        pred_associate_one = pred_associations[ctr]
        result_inst_one = {'image_id':img_id,'category_id':class_one,'bbox':box_one,
                           'score':score_one,'segmentation':seg_one,'pred_associations':pred_associate_one}
        result_img_one.append(result_inst_one)
    return result_img_one


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", default="./",
                        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="./my_vis_result/kitti_mots_train/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

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
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if os.path.exists(args.output) == False:
        os.mkdir(args.output)
    cfg = setup_cfg(args)
    args.input = [os.path.join(args.input, path)
                  for path in os.listdir(args.input)]
    demo = VisualizationDemo(cfg)
    result_all_img = []
    if args.input:
        if os.path.isdir(args.input[0]):
            # args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
            args.input = [os.path.join(
                seq_one, fname) for seq_one in args.input for fname in os.listdir(seq_one)]
            args.input.sort()
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            torch.cuda.empty_cache()
            img = cv2.imread(path)
            start_time = time.time()
            with torch.no_grad():
                instances, visualized_output = demo.run_on_image(img)

            if instances is None:
                # this means no object/shadow found in the img.
                continue
            result_img_one = instance_to_coco_json_result(instances, path)
            result_all_img.extend(result_img_one)
             
            if args.output:

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    seq_id = path.split('/')[-2]
                    img_out_dir = os.path.join(args.output,seq_id)
                    os.makedirs(img_out_dir,exist_ok=True)
                    # out_filename = os.path.join(
                    #     args.output, os.path.basename(path))
                    out_filename = os.path.join(
                        img_out_dir, os.path.basename(path))
                else:
                    assert len(
                        args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                
        # save json to disk.
        json_out_dir = './my_json_result/kitti_mots_train/'
        os.makedirs(json_out_dir,exist_ok=True)
        json_out_path = json_out_dir+'shadow_instance_result_out_assc.json'
        with open(json_out_path,'w') as f:
            json.dump(result_all_img,f,indent=2)
