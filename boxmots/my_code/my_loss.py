import torch
from torch import nn

import numpy as np
import bisect

import logging
logger = logging.getLogger(name="adet.trainer")
# the file is from FairMOT code.
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # for norm to same scale.
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # below by cws.
        # need to decide whether there is no positive sample(except the anchor itself), or no negative sample.
        diag_false=torch.zeros(n,dtype=torch.bool)
        mask_copy_all_neg=mask.clone()
        mask_copy_all_neg.diagonal().copy_(diag_false)
        # judge whether all values are False, or all values are True.
        if (mask_copy_all_neg==False).all():
            # remove + between strings to avoid lazy formatting rule by pylint.
            logger.warning("\n************************************************************\n"
                           "ReID Warning: Negative pairs only in this batch. No use!!!\n"
                           "************************************************************")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        if (mask==True).all():
            logger.warning("\n************************************************************\n"
                           "ReID Warning: Positive pairs only in this batch. No use!!!\n"
                           "************************************************************")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class TripletLossSemHard(nn.Module):
    """Triplet loss with semi-hard negative and all positive.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLossSemHard, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # for norm to same scale.
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # below by cws.
        # need to decide whether there is no positive sample(except the anchor itself), or no negative sample.
        diag_false=torch.zeros(n,dtype=torch.bool)
        mask_copy_all_neg=mask.clone()
        mask_copy_all_neg.diagonal().copy_(diag_false)
        # judge whether all values are False, or all values are True.
        if (mask_copy_all_neg==False).all() or (mask==True).all():
            logger.warning("\n***************************************\n"
                           "ReID Warning:This batch is no use!!!\n"
                           "***************************************")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class PairTripletLoss(nn.Module):
    """PairTripletLoss limits the triplet choice to 2 consecutive frames.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(PairTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, inst_num_by_img):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        def get_img_id_per_instance(inst_num_by_img):
            # below get img id for each instance,
            # so we can know which img follows the current img.
            gt_num_acc=np.cumsum(inst_num_by_img)
            img_id_by_inst=[]
            for inst_id in range(len(targets)):
                img_id=bisect.bisect_right(gt_num_acc,inst_id)
                img_id_by_inst.append(img_id)
            return np.array(img_id_by_inst)
        
        def check_pos_neg(pair_targets):
            pair_n = pair_targets.size(0)
            pair_mask = pair_targets.expand(pair_n, pair_n).eq(pair_targets.expand(pair_n, pair_n).t())
            # below by cws.
            # need to decide whether there is no positive sample(except the anchor itself), or no negative sample.
            diag_false=torch.zeros(pair_n,dtype=torch.bool)
            mask_copy_all_neg=pair_mask.clone()
            mask_copy_all_neg.diagonal().copy_(diag_false)
            # judge whether all values are False, or all values are True.
            if (mask_copy_all_neg==False).all():
                # only neg samples.
                return 1
            if (pair_mask==True).all():
                # only pos samples.
                return 2
            # if have both pos and neg sample, return 0.
            return 0
        
        # def print_for_check_pair(check_result):
        #     if check_result==1:
        #         # remove + between strings to avoid lazy formatting rule by pylint.
        #         logger.info("ReID Info: Negative pairs only in this pair. Skip.")
        #     if check_result==2:
        #         logger.info("ReID Info: Positive pairs only in this pair. Skip.")
        
        def get_dist_ap_an(n,dist,mask):
            dist_ap, dist_an = [], []
            # For each anchor, find the hardest positive and negative
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            return dist_ap,dist_an

        
        n = inputs.size(0)
        # for norm to same scale.
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        # first check all instancs in the batch.
        check_all=check_pos_neg(targets)
        if check_all==1:
            # remove + between strings to avoid lazy formatting rule by pylint.
            logger.warning("\n************************************************************\n"
            "ReID Warning: Negative pairs only in this batch. No use!!!\n"
            "************************************************************")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        if check_all==2:
            logger.warning("\n************************************************************\n"
                "ReID Warning: Positive pairs only in this batch. No use!!!\n"
                "************************************************************")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        
        img_id_by_inst=get_img_id_per_instance(inst_num_by_img)
        dist_ap_all=[]
        dist_an_all=[]
        check_pair_flag=[]
        for i in range(len(inst_num_by_img)-1):
            cur_inst_img_id=i
            next_img_id=cur_inst_img_id+1
            pair_index=np.where((img_id_by_inst==cur_inst_img_id) | (img_id_by_inst==next_img_id))
            # no below, it would only get len(pair_index) elements
            # with the first pair_index as x coordiante, and the second pair_index as y coordiante.
            # pair_dist=dist[pair_index,pair_index]
            pair_dist=dist[np.min(pair_index):(np.max(pair_index)+1),np.min(pair_index):(np.max(pair_index)+1)]
            pair_targets=targets[pair_index]
            pair_n=len(pair_targets)
            pair_mask = pair_targets.expand(pair_n, pair_n).eq(pair_targets.expand(pair_n, pair_n).t())
            check_pair=check_pos_neg(pair_targets)
            check_pair_flag.append(check_pair)
            if check_pair>0:
                # print_for_check_pair(check_pair)
                # check_pair_flag.append(0)
                continue
            # check_pair_flag.append(1)
            pair_dist_ap,pair_dist_an=get_dist_ap_an(pair_n,pair_dist,pair_mask)
            dist_ap_all.extend(pair_dist_ap)
            dist_an_all.extend(pair_dist_an)
        
        if np.all(np.array(check_pair_flag)>0):
            logger.warning("ReID Warning: No triplet in this batch for pair image. No Use!!!")
            loss=torch.tensor(0,dtype=inputs.dtype)
            return loss
        
        dist_ap = torch.cat(dist_ap_all)
        dist_an = torch.cat(dist_an_all)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss