from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform
from adet.utils.comm import aligned_bilinear

# below by cws.
import torch.nn.functional as F
import numpy as np

#FIXME: the cost volume tracking offset in training stage is wrong,
#because the img would be transformed in training,
#so cost colume tracking offset should be computed in test stage.

def build_cva_branch(cfg, input_shape):
    return CostVolumeBranch(cfg, input_shape)

# CVA is modified from TraDes.
class CostVolumeBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        
        self.cfg=cfg
        self.my_iter_count=0
        self.in_features = ['p3']
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.embed_dim = feature_channels[self.in_features[0]]
        # temparature makes the prob_distribution smoother.
        self.tempature = 5
        self.maxpool_stride2 = nn.MaxPool2d(2, stride=2)
        
    def save_flow(self,flow_imgs,gt_instances):
        # this is to save flow_imgs.
        np_img=flow_imgs.detach().clone().cpu().numpy()
        
    def CVA(self, embedding_prime, feat_prev, batch_size, h_c, w_c):
        # h = int(opt.output_h / 2)
        # w = int(opt.output_w / 2)
        h = h_c
        w = w_c
        off_template_w = np.zeros((h, w, w), dtype=np.float32)
        off_template_h = np.zeros((h, w, h), dtype=np.float32)
        for ii in range(h):
            for jj in range(w):
                for i in range(h):
                    off_template_h[ii, jj, i] = i - ii
                for j in range(w):
                    off_template_w[ii, jj, j] = j - jj
        self.m = np.reshape(
            off_template_w, newshape=(h * w, w))[None, :, :] * 2
        self.v = np.reshape(
            off_template_h, newshape=(h * w, h))[None, :, :] * 2

        # embedding_prime is the feat of current frame.
        embedding_prime = self.maxpool_stride2(embedding_prime)
        # (B, 128, H, W) -> (B, H*W, 128):
        embedding_prime = embedding_prime.view(
            batch_size, self.embed_dim, -1).permute(0, 2, 1)

        # embedding_prev = self.embedconv(feat_prev)
        # _embedding_prev = self.maxpool_stride2(embedding_prev)
        _embedding_prev = self.maxpool_stride2(feat_prev)
        _embedding_prev = _embedding_prev.view(batch_size, self.embed_dim, -1)
        # Cost Volume Map
        c = torch.matmul(embedding_prime, _embedding_prev)  # (B, H*W/4, H*W/4)
        c = c.view(batch_size, h_c * w_c, h_c, w_c)  # (B, H*W, H, W)

        c_h = c.max(dim=3)[0]  # (B, H*W, H)
        c_w = c.max(dim=2)[0]  # (B, H*W, W)
        c_h_softmax = F.softmax(c_h * self.tempature, dim=2)
        c_w_softmax = F.softmax(c_w * self.tempature, dim=2)
        v = torch.tensor(self.v, device=embedding_prime.device)  # (1, H*W, H)
        m = torch.tensor(self.m, device=embedding_prime.device)
        off_h = torch.sum(c_h_softmax * v, dim=2,
                          keepdim=True).permute(0, 2, 1)
        off_w = torch.sum(c_w_softmax * m, dim=2,
                          keepdim=True).permute(0, 2, 1)
        off_h = off_h.view(batch_size, 1, h_c, w_c)
        off_w = off_w.view(batch_size, 1, h_c, w_c)
        off_h = nn.functional.interpolate(off_h, scale_factor=2)
        off_w = nn.functional.interpolate(off_w, scale_factor=2)

        tracking_offset = torch.cat((off_w, off_h), dim=1)

        return c_h, c_w, tracking_offset

    def forward(self, features, gt_instances=None, pred_instances=None):
        """
        Get pixel level offset of 2 adjacent frames.
        """
        # if self.training:
        feature_use = features[self.in_features[0]]
        batch_size_all = feature_use.shape[0]
        batch_size = batch_size_all//2
        h_f = feature_use.shape[2]
        w_f = feature_use.shape[3]
        h_c = int(h_f / 2)
        w_c = int(w_f / 2)

        # embedding_prime is the current frame feat,
        # feat_prev is the previous frame feat.
        # eg, one batch: [1,2,5,6,9,10,13,14],
        # embedding_prime: [2,6,10,14], feat_prev: [1,5,9,13].

        # below extract embedding_prime abd feat_prev.
        embedding_prime = feature_use[1::2, :, :, :]
        feat_prev = feature_use[0::2, :, :, :]
        c_h, c_w, tracking_offset = self.CVA(
            embedding_prime, feat_prev, batch_size, h_c, w_c)
        return c_h, c_w, tracking_offset
