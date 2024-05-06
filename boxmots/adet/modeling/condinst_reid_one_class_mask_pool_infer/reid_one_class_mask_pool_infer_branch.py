from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform
from adet.utils.comm import aligned_bilinear

# below by cws.
from detectron2.layers import ROIAlign
from my_code.my_loss import TripletLoss
INF = 100000000
import numpy as np
import bisect
import logging
logger = logging.getLogger(name="adet.trainer")

# this one class means take car and ped as one class.
def build_reid_branch(cfg, input_shape):
    return ReIDBranchOneClassMaskPool(cfg, input_shape)


class ReIDBranchOneClassMaskPool(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        
        # input_channel->128->32.
        # backbone+roi pooling+1 conv 3*3+avgpool+2 fc.
        # or backbone+mask pooling+1 conv 3*3+2 fc.
        self.in_features = ['p3']
        # FCOS.FPN_STRIDES[0] is for 'p3'.
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES[0]
        reid_feat_dim=cfg.MODEL.CONDINSTREID.REID_BRANCH_OUT_CHANNELS
        self.reid_loss_weight=cfg.MODEL.CONDINSTREID.REID_LOSS_WEIGHT
        norm='BN'
        fc_1_channel=128
        
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        conv_block = conv_with_kaiming_uniform(norm, activation=True)
        self.conv_1=conv_block(feature_channels[self.in_features[0]],feature_channels[self.in_features[0]],3,1)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc_1=nn.Sequential(nn.Linear(feature_channels[self.in_features[0]],fc_1_channel),
                                nn.ReLU(inplace=True))
        # TODO: or take it as a classification task, like JDE/FairMOT,
        # that would be easier for dataloader.
        self.fc_2=nn.Linear(fc_1_channel,reid_feat_dim)
        # note spatial_scale should <1.
        self.roi_align=ROIAlign((7,7),spatial_scale=1/self.fpn_strides,sampling_ratio=0,aligned=True)
        self.trip_loss=TripletLoss()
    
    
    def forward(self, features, gt_instances=None,pred_masks=None,pred_instances=None):
        """
        pred_masks is for mask pooling.
        """
        def mask_pool_one_img(feat_map,pred_masks,mask_threshold=0.5):
            pred_masks_float = (pred_masks > mask_threshold).float()
            product=torch.mul(feat_map,pred_masks_float)
            # avg size:[n,256],
            # n is the number of instances in one img.
            for pre_mask_one in pred_masks:
                assert (pre_mask_one>mask_threshold).any(),"can not use full black pred_mask!"
            avg=torch.sum(product,dim=(2,3))/torch.sum(pred_masks_float,dim=(2,3))
            return avg
        
        if self.training:
            # TODO: use P3,P4,P5 features together.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            
            trackid_list_nest=[x.track_ids for x in gt_instances]
            trackid_list=[x for y in trackid_list_nest for x in y]
            track_id_tensor=torch.tensor(trackid_list)
            
            mask_th=0.5
            # gt_inds is the corresponding gt index of pred_masks/pred_boxes.
            gt_inds=pred_instances.gt_inds
            gt_inds_in_use=torch.unique(gt_inds)
            mask_dict_instance={}
            # get the corresponding pred_mask for each gt mask.
            # note gt_inds may not contain all gt masks in one img,
            # like pred_inst would correspond to only one of two overlapping gt objects,
            # the smaller area one in fcos.
            for gt_id_instance_one in gt_inds_in_use:
                ind=(gt_inds==gt_id_instance_one).nonzero(as_tuple=True)[0]
                pred_masks_one_id=pred_masks[ind,:,:,:]
                pred_mask_avg=torch.squeeze(torch.mean(pred_masks_one_id,dim=0))
                # note pred_mask_avg could be all<mask threshold,
                # which is a full black binary map, then it should be removed,
                # otherwise it would generate nan in mask_pool_one_img function.
                if not (pred_mask_avg>mask_th).any():
                    logger.info("This is a full black pred mask! It's not used in mask pool reid!")
                    continue
                # gt_id_instance_one.item() is to get the python version data as dict key,
                # otherwise the key would be tensor, which is hard to use.
                mask_dict_instance[gt_id_instance_one.item()]=pred_mask_avg
            
            # below combine masks of one image to one tensor.
            gt_count=[len(x) for x in gt_instances]
            gt_num_acc=np.cumsum(gt_count)
            mask_dict_img={}
            for gt_id_instance_one,pred_mask_instance_one in mask_dict_instance.items():
                img_id=bisect.bisect_right(gt_num_acc,gt_id_instance_one)
                if img_id not in mask_dict_img:
                    mask_dict_img[img_id]=[pred_mask_instance_one]
                else:
                    mask_dict_img[img_id].append(pred_mask_instance_one)
            # convert list to tensor and do mask pool.
            x_vec_list=[]
            for k in mask_dict_img.keys():
                # shape: [n,1,h,w]
                mask_tensor_img_one=torch.stack(mask_dict_img[k],dim=0).unsqueeze(dim=1)
                x_vec_img_one=mask_pool_one_img(x_conv_1[k,:,:,:].unsqueeze(dim=0),mask_tensor_img_one,mask_threshold=mask_th)
                # if len(x_vec_img_one)!=gt_count[k]:
                #     print('kk')
                x_vec_list.append(x_vec_img_one)
            
            x_vec=torch.cat(x_vec_list,dim=0)
            x_fc_1=self.fc_1(x_vec)
            x_fc_2=self.fc_2(x_fc_1)
            track_id_tensor_in_use=track_id_tensor[list(mask_dict_instance.keys())]
            # if len(x_fc_2)!=len(track_id_tensor):
            #     print('kk')

            # if torch.isnan(x_fc_2).any():
            #     print('kk')
                
            # below for loss.
            self.trip_loss.to(x_vec.device)
            loss=self.trip_loss(x_fc_2,track_id_tensor_in_use)
            
            losses={}
            losses['loss_reid']=self.reid_loss_weight*loss
            return losses
        
        # below for ReID infer.
        # the input is boxes produced by detection branch.
        else:
            # TODO: need to filter the pred box whose reid feat has Nan when DeepSORT use the reid_feat.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            
            # use pred_boxes to judge whether the img has results,
            # since pred_boxes is paired with mask in the coco out json file,
            # but can't use (pred_masks > mask_threshold).any()==False to
            # judge whether the img doesn't have mask output,
            # since for one img that doesn't have bbox output,
            # the pred_masks would still have several pixels
            # that have probability value>mask_threshold,
            # so we use pred_boxes to judge whether the img has results.
            if pred_instances.pred_boxes.tensor.nelement()==0:
                reid_out={}
                reid_out["reid_feats"]=torch.empty(0).to(x_conv_1.device)
                return reid_out
            else:
                # note the input instances are predicted masks, not gt instances.
                mask_threshold=0.5
                pred_masks_float = (pred_masks > mask_threshold).float()
                product=torch.mul(x_conv_1,pred_masks_float)
                # avg size:[7,256]
                avg=torch.sum(product,dim=(2,3))/torch.sum(pred_masks_float,dim=(2,3))

                x_vec=avg

                x_fc_1=self.fc_1(x_vec)
                x_fc_2=self.fc_2(x_fc_1)
                
                reid_out={}
                reid_out["reid_feats"]=x_fc_2
                
                return reid_out

