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

# this one class means take car and ped as one class.
def build_reid_branch(cfg, input_shape):
    return ReIDBranchOneClass(cfg, input_shape)


class ReIDBranchOneClass(nn.Module):
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

    def forward(self, features, gt_instances=None,pred_instances=None):
        """
        pred_instances is for rois.
        """
        if self.training:
            # TODO: use P3,P4,P5 features together.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            # roi_pooling to vector, then go to fc layers.
            # prepare rois for roi align,
            # rois shape need to be B*5, and the 1st column is the img order in one batch.
            rois_list=[]
            trackid_list=[]
            for img_order_per_batch,inst_per_img in enumerate(gt_instances):
                boxes_per_img=inst_per_img.gt_boxes.tensor
                index_column=torch.ones((len(boxes_per_img),1),dtype=boxes_per_img.dtype,device=boxes_per_img.device)*img_order_per_batch
                rois_per_img=torch.cat((index_column,boxes_per_img),dim=1)
                rois_list.append(rois_per_img)
                # below get track_id.
                trackid_per_img=inst_per_img.track_ids
                trackid_list.extend(trackid_per_img)
            rois_tensor=torch.cat(rois_list,dim=0)
            track_id_tensor=torch.tensor(trackid_list)
            
            x_roi=self.roi_align(x_conv_1,rois_tensor)
            x_pool=self.avg_pool(x_roi)
            x_vec=torch.flatten(x_pool,1)        

            x_fc_1=self.fc_1(x_vec)
            x_fc_2=self.fc_2(x_fc_1)

            # below for loss.
            self.trip_loss.to(x_vec.device)
            loss=self.trip_loss(x_fc_2,track_id_tensor)
            
            losses={}
            losses['loss_reid']=self.reid_loss_weight*loss
            return losses
        
        # below for ReID infer.
        # the input is boxes produced by detection branch.
        else:
            # TODO: use P3,P4,P5 features together.
            feature_use=features[self.in_features[0]]
            x_conv_1=self.conv_1(feature_use)
            # roi_pooling to vector, then go to fc layers.
            # prepare rois for roi align,
            # rois shape need to be B*5, and the 1st column is the img order in one batch.
            rois_list=[]
            # note the input instances are predicted instances, not gt instances.
            # make pred_instances iterable.
            # if pred_instances.pred_boxes.tensor.nelement()==0:
            #     print('kk')
            pred_instances=[pred_instances]
            for img_order_per_batch,inst_per_img in enumerate(pred_instances):
                boxes_per_img=inst_per_img.pred_boxes.tensor
                index_column=torch.ones((len(boxes_per_img),1),dtype=boxes_per_img.dtype,device=boxes_per_img.device)*img_order_per_batch
                rois_per_img=torch.cat((index_column,boxes_per_img),dim=1)
                rois_list.append(rois_per_img)
            rois_tensor=torch.cat(rois_list,dim=0)
            
            x_roi=self.roi_align(x_conv_1,rois_tensor)
            x_pool=self.avg_pool(x_roi)
            x_vec=torch.flatten(x_pool,1)        

            x_fc_1=self.fc_1(x_vec)
            x_fc_2=self.fc_2(x_fc_1)
            
            reid_out={}
            reid_out["reid_feats"]=x_fc_2
            
            return reid_out

