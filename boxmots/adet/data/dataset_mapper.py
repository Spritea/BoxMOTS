import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations, annotations_to_instances_reid,
                              annotations_to_instances_reid_right_track,bdd_annotations_to_instances_reid_right_track,
                              bdd_annotations_to_instances_reid_right_track_eval,bdd_train_videos_id,
                              ytvis_2019_annotations_to_instances_reid_right_track, ytvis_2019_img_filename_to_full_info_dict)

from pathlib import Path
import json
import os


"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)

def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m

def get_one_car_shadow(file_name, ori_img_h, ori_img_w):
    shadow_file_path = "my_shadow_det/my_shadow_result/my_json_result/kitti_mots_train/shadow_instance_result_out_assc.json"
    with open(shadow_file_path,'r') as f:
        shadow_data_one = json.load(f)
    img_my_id = get_img_my_id_from_filename(file_name)
    # category_id: 0-object,1-shadow.
    shadow_img_one = list(filter(lambda x:x['image_id']==img_my_id 
                                 and x['category_id']==1,shadow_data_one))
    # below merge shadow.
    shadow_mask_img_one = [x['segmentation'] for x in shadow_img_one]
    if len(shadow_mask_img_one)>=1:
        if len(shadow_mask_img_one)>1:
            shadow_mask_img_one = [maskUtils.merge(shadow_mask_img_one,intersect=False)]
        shadow_all_img_one = maskUtils.decode(shadow_mask_img_one[0])
    elif len(shadow_mask_img_one)==0:
        # make a all 0 shadow mask when there is no detected shadow.
        shadow_all_img_one = np.zeros((ori_img_h,ori_img_w),dtype=np.uint8)
    return shadow_all_img_one

def get_one_flow(file_name, ori_img_h, ori_img_w):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_id = Path(file_name).parent.name
    img1_id = Path(file_name).stem
    img2_id = str(int(img1_id)+1).zfill(6)
    flow_file_name = '_'.join([seq_id, img1_id, img2_id])+'.npy'
    flow_file_dir = "my_opt_flow_results/KITTI_MOTS/train_in_trainval"
    flow_file_path = flow_file_dir+'/'+seq_id + \
        '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    if not Path(flow_file_path).exists():
        # for the last img of each seq.
        img0_id = str(int(img1_id)-1).zfill(6)
        flow_file_name = '_'.join([seq_id, img1_id, img0_id])+'.npy'
        flow_file_path = flow_file_dir+'/'+seq_id + \
            '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_one_flow_inverse(file_name, ori_img_h, ori_img_w):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_id = Path(file_name).parent.name
    img1_id = Path(file_name).stem
    img2_id = str(int(img1_id)+1).zfill(6)
    # flow_file_name = '_'.join([seq_id, img1_id, img2_id])+'.npy'
    flow_file_name = '_'.join([seq_id, img2_id, img1_id])+'.npy'
    flow_file_dir = "my_opt_flow_results/KITTI_MOTS/train_in_trainval"
    flow_file_path = flow_file_dir+'/'+seq_id + \
        '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    if not Path(flow_file_path).exists():
        # for the last img of each seq.
        img0_id = str(int(img1_id)-1).zfill(6)
        flow_file_name = '_'.join([seq_id, img1_id, img0_id])+'.npy'
        flow_file_path = flow_file_dir+'/'+seq_id + \
            '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_one_flow_gauss_noise(file_name, ori_img_h, ori_img_w, flow_noise_std):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_id = Path(file_name).parent.name
    img1_id = Path(file_name).stem
    img2_id = str(int(img1_id)+1).zfill(6)
    flow_file_name = '_'.join([seq_id, img1_id, img2_id])+'.npy'
    flow_file_dir = "my_optical_flow_with_gauss_noise_hdda/KITTI_MOTS/train_in_trainval"
    flow_file_path = flow_file_dir+'/'+seq_id + \
        f'/ckpt_sintel_np_offset_mat_bidirect_with_gauss_noise_std_{flow_noise_std}/'+flow_file_name
    if not Path(flow_file_path).exists():
        # for the last img of each seq.
        img0_id = str(int(img1_id)-1).zfill(6)
        flow_file_name = '_'.join([seq_id, img1_id, img0_id])+'.npy'
        flow_file_path = flow_file_dir+'/'+seq_id + \
            f'/ckpt_sintel_np_offset_mat_bidirect_with_gauss_noise_std_{flow_noise_std}/'+flow_file_name
    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_one_flow_constant_noise(file_name, ori_img_h, ori_img_w, constant_noise_val):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_id = Path(file_name).parent.name
    img1_id = Path(file_name).stem
    img2_id = str(int(img1_id)+1).zfill(6)
    flow_file_name = '_'.join([seq_id, img1_id, img2_id])+'.npy'
    flow_file_dir = "my_optical_flow_with_constant_noise/KITTI_MOTS/train_in_trainval"
    flow_file_path = flow_file_dir+'/'+seq_id + \
        f'/ckpt_sintel_np_offset_mat_bidirect_with_constant_noise_val_{constant_noise_val}/'+flow_file_name
    if not Path(flow_file_path).exists():
        # for the last img of each seq.
        img0_id = str(int(img1_id)-1).zfill(6)
        flow_file_name = '_'.join([seq_id, img1_id, img0_id])+'.npy'
        flow_file_path = flow_file_dir+'/'+seq_id + \
            f'/ckpt_sintel_np_offset_mat_bidirect_with_constant_noise_val_{constant_noise_val}/'+flow_file_name
    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_bdd_one_flow(file_name, ori_img_h, ori_img_w):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_id = Path(file_name).parent.name
    img1_id_num = Path(file_name).stem.split('-')[2]
    img2_id_num = str(int(img1_id_num)+1).zfill(7)
    img1_id = seq_id+'-'+img1_id_num
    img2_id = seq_id+'-'+img2_id_num
    flow_file_name = '_'.join([seq_id, img1_id, img2_id])+'.npy'
    flow_file_dir = "my_opt_flow_results_hdda/BDD_MOTS/train"
    flow_file_path = flow_file_dir+'/'+seq_id + \
        '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    if not Path(flow_file_path).exists():
        # for the last img of each seq.
        img0_id_num = str(int(img1_id_num)-1).zfill(7)
        img0_id = seq_id+'-'+img0_id_num
        flow_file_name = '_'.join([seq_id, img1_id, img0_id])+'.npy'
        flow_file_path = flow_file_dir+'/'+seq_id + \
            '/ckpt_sintel_np_offset_mat_bidirect/'+flow_file_name
    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_ytvis_2019_one_flow(file_name, ori_img_h, ori_img_w):
    # by cws. to get flow of the former img in one pair.
    # if the img is the last img of one seq, get the flow of (last_img,last_img_but_one)
    seq_name = Path(file_name).parent.name
    name_parts = Path(file_name).parts
    img_filename = os.path.join(name_parts[-2],name_parts[-1])
    img_name = Path(img_filename).stem
    
    # check whether the img is the last img of one seq.
    is_video_last = ytvis_2019_img_filename_to_full_info_dict[img_filename]['is_video_last_frame']
    
    if not is_video_last:
        # get the next img of the current img in the same video.
        next_frame_img_filename = ytvis_2019_img_filename_to_full_info_dict[img_filename]['next_img']
        next_frame_img_name = Path(next_frame_img_filename).stem
        img_for_flow = next_frame_img_name
    else:
        # get the previous img of the current img in the same video.
        # in fact, for this case where the 2 images are from different videos,
        # optical flow loss is not computed in the training process,
        # check line 318 (if gt_instances[ctr*2].img_my_id[0]+1 == gt_instances[ctr*2+1].img_my_id[0]:) in adet/modeling/condinst_reid_one_class_infer_pair_warp_reid_eval_in_train/dynamic_mask_head_pair_warp.py,
        # or line 318 (if gt_instances[ctr*2].img_my_id[0]+1 == gt_instances[ctr*2+1].img_my_id[0]:) in adet/modeling/condinst_reid_one_class_infer_bdd_pair_warp_reid_eval_in_train/dynamic_mask_head_pair_warp.py.
        # note the img_my_id is based on both seq id and img id (for kitti and bdd) or frame id (for ytvis 2019),
        # and the img_my_id is consecutive only in the same video,
        # so we can decide whether two images are consecutive in one video by checking the img_my_id.
        prev_frame_img_filename = ytvis_2019_img_filename_to_full_info_dict[img_filename]['prev_img']
        prev_frame_img_name = Path(prev_frame_img_filename).stem
        img_for_flow = prev_frame_img_name

    flow_file_dir = "my_optical_flow_results_ytvis_2019/YouTube_VIS_2019/train"
    flow_file_path = os.path.join(flow_file_dir, seq_name, 'ckpt_sintel_np_offset_mat_bidirect', f'{seq_name}_{img_name}_{img_for_flow}.npy')

    with open(flow_file_path, 'rb') as f:
        flow_data_one = np.load(f)
        # [1,2,H,W]->[H,W,2]
        flow_data_one = np.transpose(np.squeeze(
            flow_data_one, axis=0), axes=(1, 2, 0))
        # FIXME: np.resize use repeat, may do harm,
        # but for now don't know how to upsample coordinates.
        flow_data_one = np.resize(np.ascontiguousarray(
            flow_data_one), (ori_img_h, ori_img_w, 2))
    return flow_data_one

def get_img_my_id_from_filename(file_name):
    file_name_parts = file_name.split('/')
    img_my_id = int(file_name_parts[-2])*10000 + int(file_name_parts[-1].split('.')[-2])
    return img_my_id

def get_bdd_img_my_id_from_filename(file_name):
    bdd_train_video_id_dict={z['name']:z['id'] for z in bdd_train_videos_id}
    folder=Path(file_name).stem.rsplit('-',1)[0]
    img_id_in_seq=Path(file_name).stem.rsplit('-',1)[1]
    seq_id=bdd_train_video_id_dict[folder]
    # below make img_my_id unique across seqs by adding seq_id to the img number.    
    img_my_id=seq_id*10000+int(img_id_in_seq)
    # file_name_parts = file_name.split('/')
    # img_my_id = int(file_name_parts[-2])*10000 + int(file_name_parts[-1].split('.')[-2])
    return img_my_id

def get_ytvis_2019_img_my_id_from_filename(file_name):
    # img_my_id is constructed in a way that
    # consecutive img_my_id means consecutive frames in the same video.
    # if two images are from different videos, the img_my_id is not consecutive.
    name_parts = Path(file_name).parts
    # img_filename format: 'e00baaae9b/00140.jpg'.
    img_filename = os.path.join(name_parts[-2],name_parts[-1])
    
    frame_id = ytvis_2019_img_filename_to_full_info_dict[img_filename]['frame_id']
    video_id = ytvis_2019_img_filename_to_full_info_dict[img_filename]['video_id']
    
    img_my_id = video_id*10000+int(frame_id)
    return img_my_id


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict


class DatasetMapperWithBasisReID(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDRightTrack(DatasetMapper):
    """
    This uses right track label. The track label in DatasetMapperWithBasisReID is wrong,
    since the original track label is only unique in one seq, not unique across seqs. 
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDEval(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            # need annotations for ReID eval.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            # below no return, since we need to get instances information.
            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDEvalRightTrack(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            # need annotations for ReID eval.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            # below no return, since we need to get instances information.
            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict


class DatasetMapperWithBasisReIDOptFlow(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            flow_one_img = get_one_flow(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                flow_after_transform = transforms.apply_image(flow_one_img)
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict


class DatasetMapperWithBasisReIDOptFlowRightAug(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            flow_one_img = get_one_flow(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDOptFlowRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            flow_one_img = get_one_flow(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDOptFlowInverseRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            flow_one_img = get_one_flow_inverse(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDOptFlowGaussNoiseRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON
        self.flow_noise_std = cfg.OPTICAL_FLOW_WITH_GAUSS_NOISE.GAUSS_NOISE_STD

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            # flow_one_img = get_one_flow(
            #     dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            flow_one_img = get_one_flow_gauss_noise(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'], self.flow_noise_std)
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDOptFlowConstantNoiseRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON
        self.constant_noise_val = cfg.OPTICAL_FLOW_WITH_CONSTANT_NOISE.CONSTANT_NOISE_VAL

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid_right_track(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            # below for opt flow.
            # flow_one_img = get_one_flow(
            #     dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            flow_one_img = get_one_flow_constant_noise(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'], self.constant_noise_val)
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class DatasetMapperWithBasisReIDCarShadow(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            image_id = dataset_dict['image_id']
            instances = annotations_to_instances_reid(
                image_id, annos, image_shape, mask_format=self.instance_mask_format
            )
            #below for car shadow.
            shadow_one_img = get_one_car_shadow(dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                shadow_transforms = T.TransformList(transforms)
                shadow_after_transform = shadow_transforms.apply_image(shadow_one_img)
            else:
                shadow_after_transform = transforms.apply_image(shadow_one_img)
            # [H,W] -> [1,1,H,W]
            shadow_after_transform = np.expand_dims(shadow_after_transform, axis=(0,1))
            # need to make shadow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            shadow_after_transform = np.repeat(shadow_after_transform, len(instances), axis=0)
            instances.shadow_data = torch.from_numpy(shadow_after_transform)
            
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class BDDDatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format='bitmask'
            )
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
    
class BDDDatasetMapperWithBasisReID(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = bdd_annotations_to_instances_reid_right_track(
                file_name, annos, image_shape, mask_format='bitmask'
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class BDDDatasetMapperWithBasisReIDEval(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            # need annotations for ReID eval.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            # below no return, since we need to get instances information.
            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = bdd_annotations_to_instances_reid_right_track_eval(
                file_name, annos, image_shape, mask_format='bitmask'
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class BDDDatasetMapperWithBasisReIDOptFlowRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = bdd_annotations_to_instances_reid_right_track(
                file_name, annos, image_shape, mask_format='bitmask'
            )
            # below for opt flow.
            flow_one_img = get_bdd_one_flow(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_bdd_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class YTVIS2019DatasetMapperWithBasisReID(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = ytvis_2019_annotations_to_instances_reid_right_track(
                file_name, annos, image_shape, mask_format='bitmask'
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class YTVIS2019DatasetMapperWithBasisReIDEval(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            # need annotations for ReID eval.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            # below no return, since we need to get instances information.
            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = ytvis_2019_annotations_to_instances_reid_right_track(
                file_name, annos, image_shape, mask_format='bitmask'
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

class YTVIS2019DatasetMapperWithBasisReIDOptFlowRightAugRightTrack(DatasetMapper):
    """
    By cws: This uses correct aug for optical flow data in training stage.
    For downsample, first sample the tracking offset with the step as downsample rate,
    and this step is the same with the one on img.
    Then the tracking offset itself needs to be divided by the downsample rate,
    since the img is downsampled, and the coordinate range changed.
    For horizontal flip, first get a reverse order alone the horizontal axis,
    and this step is the same with the one on img.
    Then the tracking offset should time -1.0 on the horizontal axis,
    since the moving direction along the horizontal axis is reversed.
    Mainline: first get the corresponding pixel based on the transformed img,
    then consider the influence the transform make on the tracking offset itself.
    Ref on RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py,
    and upflow8 on https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py#L80.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        self.cfg = cfg

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            if cfg.INPUT.IS_ROTATE:
                self.augmentation.insert(
                    1,
                    T.RandomRotation(angle=[-30, 30], sample_style="range")
                )
                logging.getLogger(__name__).info(
                    "Rotation used in training: " + str(self.augmentation[1])
                )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

        # decide whether reid is on.
        # self.reid_on=cfg.MODEL.CONDINSTREID.REID_BRANCH.ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.cfg.INPUT.IS_ROTATE:
            augmentation = self.augmentation[2:]
            pp = np.random.rand()
            if pp < 0.5:
                augmentation = [self.augmentation[0]] + augmentation
            pp1 = np.random.rand()
            if pp1 < 0.5:
                augmentation = [self.augmentation[1]] + augmentation

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        if image.shape[1] == 0 or image.shape[0] == 0:
            print(dataset_dict)
            raise e
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # instances = annotations_to_instances(
            #     annos, image_shape, mask_format=self.instance_mask_format
            # )
            # below for reid.
            file_name = dataset_dict['file_name']
            instances = ytvis_2019_annotations_to_instances_reid_right_track(
                file_name, annos, image_shape, mask_format='bitmask'
            )
            # below for opt flow.
            flow_one_img = get_ytvis_2019_one_flow(
                dataset_dict['file_name'], dataset_dict['height'], dataset_dict['width'])
            if isinstance(transforms, (tuple, list)):
                opt_flow_transforms = T.TransformList(transforms)
                flow_after_transform = opt_flow_transforms.apply_image(
                    flow_one_img)
            else:
                transform_list = transforms.transforms
                for trans_one in transform_list:
                    flow_one_img = trans_one.apply_image(flow_one_img)
                    if type(trans_one).__name__ == 'ResizeTransform':
                        resize_scale = trans_one.new_h/trans_one.h
                        flow_one_img = flow_one_img*resize_scale
                    elif type(trans_one).__name__ == 'HFlipTransform':
                        # below times (w,h,2) with (2,), uses broadcast,
                        # and the (2,) becomes (w,h,2).
                        flow_one_img = flow_one_img*[-1.0, 1.0]
                    elif type(trans_one).__name__ == 'NoOpTransform':
                        pass
                    else:
                        assert False, 'Wrong data aug!'
                flow_after_transform = flow_one_img
            # [H,W,2] -> [1,2,H,W]
            flow_after_transform = np.expand_dims(
                np.transpose(flow_after_transform, (2, 0, 1)), axis=0)
            # need to make flow_after_transform the same length with len(instances),
            # so that it could be added to instances class as attribute.
            flow_after_transform = np.repeat(
                flow_after_transform, len(instances), axis=0)
            instances.flow_data = torch.from_numpy(flow_after_transform)
            
            # record aug to instance for optical fow pair consistency.
            flip_aug_dict = {'NoOpTransform':0, 'HFlipTransform':1}
            flip_aug = type(transform_list[1]).__name__
            instances.flip_aug = torch.from_numpy(np.array([flip_aug_dict[flip_aug]]*len(instances)))
        
            # If this img is paired with another img,
            # then the augmentation must use the same one.
            # It's mainly about the RandomFlip transformation,
            # otherwise, the optical flow with augmentation is wrong.
            # record img_my_id to judge whether the 2 images are consecutive.
            img_my_id = get_ytvis_2019_img_my_id_from_filename(dataset_dict['file_name'])
            instances.img_my_id = torch.from_numpy(np.array([img_my_id]*len(instances)))

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
