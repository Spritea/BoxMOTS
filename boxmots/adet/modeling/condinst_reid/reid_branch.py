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


def build_reid_branch(cfg, input_shape):
    return ReIDBranch(cfg, input_shape)


class ReIDBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        
        # input_channel->128->32.
        # backbone+roi pooling+1 conv 3*3+avgpool+2 fc.
        # or backbone+mask pooling+1 conv 3*3+2 fc.
        self.in_features = ['p3']
        # FCOS.FPN_STRIDES[0] is for 'p3'.
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES[0]
        reid_feat_dim=cfg.MODEL.CONDINSTREID.REID_BRANCH_OUT_CHANNELS
        norm='BN'
        fc_1_channel=128
        
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        conv_block = conv_with_kaiming_uniform(norm, activation=True)
        self.conv_1=conv_block(feature_channels[self.in_features[0]],feature_channels[self.in_features[0]],3,1)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc_1_car=nn.Sequential(nn.Linear(feature_channels[self.in_features[0]],fc_1_channel),
                                nn.ReLU(inplace=True))
        self.fc_1_ped=nn.Sequential(nn.Linear(feature_channels[self.in_features[0]],fc_1_channel),
                                nn.ReLU(inplace=True))
        # TODO: or take it as a classification task, like JDE/FairMOT,
        # that would be easier for dataloader.
        self.fc_2_car=nn.Linear(fc_1_channel,reid_feat_dim)
        self.fc_2_ped=nn.Linear(fc_1_channel,reid_feat_dim)
        self.roi_align=ROIAlign((7,7),spatial_scale=self.fpn_strides,sampling_ratio=0,aligned=True)
        self.trip_loss=TripletLoss()

    def forward(self, features, gt_instances=None):
        # TODO: use P3,P4,P5 features together.
        feature_use=features[self.in_features[0]]
        x_conv_1=self.conv_1(feature_use)
        # roi_pooling to vector, then go to fc layers.
        # prepare rois for roi align,
        # rois shape need to be B*5, and the 1st column is the img order in one batch.
        rois_list=[]
        trackid_list=[]
        cat_list=[]
        for img_order_per_batch,inst_per_img in enumerate(gt_instances):
            boxes_per_img=inst_per_img.gt_boxes.tensor
            index_column=torch.ones((len(boxes_per_img),1),dtype=boxes_per_img.dtype,device=boxes_per_img.device)*img_order_per_batch
            rois_per_img=torch.cat((index_column,boxes_per_img),dim=1)
            rois_list.append(rois_per_img)
            # below get track_id.
            trackid_per_img=inst_per_img.track_ids
            trackid_list.extend(trackid_per_img)
            # below get category id.
            cat_per_img=inst_per_img.gt_classes
            cat_list.extend(cat_per_img)
        rois_tensor=torch.cat(rois_list,dim=0)
        track_id_tensor=torch.tensor(trackid_list)
        
        x_roi=self.roi_align(x_conv_1,rois_tensor)
        x_pool=self.avg_pool(x_roi)
        x_vec=torch.flatten(x_pool,1)
        
        bool_index=torch.tensor(cat_list,device=cat_list[0].device,dtype=torch.bool)
        # 0-false is car, 1-true is pedestrian.
        x_vec_car=x_vec[~bool_index]
        x_vec_ped=x_vec[bool_index]
        track_id_car=track_id_tensor[~bool_index]
        track_id_ped=track_id_tensor[bool_index]
        
        loss=torch.tensor(0,dtype=x_vec.dtype,device=x_vec.device)
        self.trip_loss.to(x_vec.device)
        if len(x_vec_car)>0:
            x_fc_1_car=self.fc_1_car(x_vec_car)
            x_fc_2_car=self.fc_2_car(x_fc_1_car)
            loss_reid_car=self.trip_loss(x_fc_2_car,track_id_car)
            loss=loss+loss_reid_car
        if len(x_vec_ped)>0:
            x_fc_1_ped=self.fc_1_ped(x_vec_ped)
            x_fc_2_ped=self.fc_2_ped(x_fc_1_ped)
            loss_reid_ped=self.trip_loss(x_fc_2_ped,track_id_ped)
            loss=loss+loss_reid_ped

        losses={}
        losses['loss_reid']=loss
        return losses
    
    # def forward_train():
    #     pass
    # def forward_test():
    #     pass

