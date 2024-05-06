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
import torch.nn.functional as F

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
        def mask_pool_one_img_with_nan(feat_map,pred_masks,mask_threshold=0.5):
            pred_masks_float = (pred_masks > mask_threshold).float()
            product=torch.mul(feat_map,pred_masks_float)
            # avg size:[n,256],
            # n is the number of instances in one img.
            # for pre_mask_one in pred_masks:
            #     assert (pre_mask_one>mask_threshold).any(),"can not use full black pred_mask!"
            avg=torch.sum(product,dim=(2,3))/torch.sum(pred_masks_float,dim=(2,3))
            return avg
        
        if self.training:
            # TODO: use P3,P4,P5 features together.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            
            trackid_list_nest=[x.track_ids for x in gt_instances]
            trackid_list=[x for y in trackid_list_nest for x in y]
            track_id_tensor=torch.tensor(trackid_list)
            
            gt_count=[len(x) for x in gt_instances]
            
            mask_th=0.5
            x_vec_list=[]
            inst_use_id=[]
            for img_ctr,img_one in enumerate(gt_instances):
                # shape: [n,1,h,w]
                mask_tensor_img_one=img_one.gt_bitmasks.unsqueeze(dim=1)
                mask_tensor_img_one_resize=F.interpolate(mask_tensor_img_one,size=x_conv_1.shape[-2:],
                                                         mode='nearest')
                # mask_tensor_img_one_resize could contain full black mask,
                # when the gt mask in mask_tensor_img_one is small, like only 7 pixels.
                mask_valid_img_one=[]
                for inst_ctr,mask_inst_one in enumerate(mask_tensor_img_one_resize):
                    if (mask_inst_one>mask_th).any():
                        mask_valid_img_one.append(mask_inst_one)
                        inst_id_in_batch=sum(gt_count[:img_ctr])+inst_ctr
                        inst_use_id.append(inst_id_in_batch)
                    else:
                        logger.info("This is a full black gt mask after resize! It's not used in mask pool reid!")
                mask_valid_img_one_tensor=torch.stack(mask_valid_img_one,dim=0)
                x_vec_img_one=mask_pool_one_img(x_conv_1[img_ctr,:,:,:].unsqueeze(dim=0),mask_valid_img_one_tensor,mask_threshold=mask_th)
                x_vec_list.append(x_vec_img_one)
            
            x_vec=torch.cat(x_vec_list,dim=0)
            x_fc_1=self.fc_1(x_vec)
            x_fc_2=self.fc_2(x_fc_1)
            self.trip_loss.to(x_vec.device)
            track_id_tensor_in_use=track_id_tensor[inst_use_id]
            loss=self.trip_loss(x_fc_2,track_id_tensor_in_use)
            
            losses={}
            losses['loss_reid']=self.reid_loss_weight*loss
            return losses
        
        # below for ReID infer.
        # the input is boxes produced by detection branch.
        else:
            # TODO: in condinst test process, there could be full black pred mask,
            # need to deal with the situation.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            
            mask_th=0.5
            x_vec_list=[]
            inst_use_id=[]
            
            if gt_instances[0].gt_boxes.tensor.nelement()==0:
                x_fc_2=torch.tensor(float('nan')).to(x_conv_1.device)
                return x_fc_2
            # in fact, the batchsize=1 in inference,
            # so the for loop is not really needed in the case.
            for img_ctr,img_one in enumerate(gt_instances):
                # shape: [n,1,h,w]
                mask_tensor_img_one=img_one.gt_bitmasks.unsqueeze(dim=1)
                mask_tensor_img_one_resize=F.interpolate(mask_tensor_img_one,size=x_conv_1.shape[-2:],
                                                        mode='nearest')
                # FIXME: mask_tensor_img_one_resize could contain full black mask,
                # when the gt mask in mask_tensor_img_one is small, like only 7 pixels,
                # need to filter the pred box whose reid feat has Nan when DeepSORT use the reid_feat.
                x_vec_img_one=mask_pool_one_img_with_nan(x_conv_1[img_ctr,:,:,:].unsqueeze(dim=0),mask_tensor_img_one_resize,mask_threshold=mask_th)
                x_vec_list.append(x_vec_img_one)
            
            x_vec=torch.cat(x_vec_list,dim=0)
            x_fc_1=self.fc_1(x_vec)
            x_fc_2=self.fc_2(x_fc_1)
            
            return x_fc_2

