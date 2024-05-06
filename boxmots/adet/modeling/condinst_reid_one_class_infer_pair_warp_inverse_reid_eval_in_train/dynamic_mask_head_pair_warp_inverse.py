import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
import numpy as np
import bisect

import logging
logger = logging.getLogger(name="adet.trainer")

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        
        # my boxinst opt flow configs
        self.flow_thresh=cfg.MODEL.BOXINST.FLOW_THRESH
        self.flow_loss_factor=cfg.MODEL.BOXINST.FLOW_LOSS_FACTOR
        self.use_color_sim=cfg.MODEL.BOXINST.USE_COLOR_SIMILARITY
        self.use_flow_sim=cfg.MODEL.BOXINST.USE_FLOW_SIMILARITY
        
        # for pair loss.
        # default reduction='mean'.
        self.pairwarp_loss_fn = nn.BCELoss()
        self.pairwarp_loss_factor = cfg.MODEL.BOXINST.PAIRWARP_LOSS_FACTOR
        self.pairwarp_conf_th = cfg.MODEL.BOXINST.PAIRWARP_CONF_THRESH
        self.min_nonzero_num = 5
        self.last_loss = None

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    # below from
                    # adet/modeling/condinst_reid_one_class_mask_pool_infer/reid_one_class_mask_pool_infer_branch.py.
                    trackid_list_nest=[x.track_ids for x in gt_instances]
                    trackid_list=[x for y in trackid_list_nest for x in y]
                    mask_th=0.5
                    # gt_inds is the corresponding gt index of pred_masks/pred_boxes.
                    gt_inds=pred_instances.gt_inds
                    gt_inds_in_use=torch.unique(gt_inds)
                    mask_dict_instance={}
                    pred_masks=mask_scores
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
                            # logger.info("This is a full black pred mask! It's not used in mask pool reid!")
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
                            mask_dict_img[img_id]={trackid_list[gt_id_instance_one].item():pred_mask_instance_one}
                        else:
                            mask_dict_img[img_id].update({trackid_list[gt_id_instance_one].item():pred_mask_instance_one})


                    prev_img_mask=[]
                    cur_img_mask=[]
                    for k,v in mask_dict_img.items():
                        if k%2==0:
                            # swap prev_img_mask and cur_img_mask here for inverse.
                            # prev_img_mask.append(v)
                            cur_img_mask.append(v)
                        else:
                            # cur_img_mask.append(v)
                            prev_img_mask.append(v)
                    if len(cur_img_mask)>=2 and len(prev_img_mask)>=2 and len(cur_img_mask)==len(prev_img_mask):
                        item_one = list(prev_img_mask[0].values())[0]
                        downsample_img_h, downsample_img_w = item_one.shape
                        x_range_downsample = torch.arange(downsample_img_w).float().to(device=item_one.device)
                        y_range_downsample = torch.arange(downsample_img_h).float().to(device=item_one.device)
                        y_grid_downsample, x_grid_downsample = torch.meshgrid([y_range_downsample, x_range_downsample])
                        
                        pair_cur_inst_mask_list = []
                        for ctr, prev_img_mask_one in enumerate(prev_img_mask):
                            # gt_instances id:[0,1,2,3],
                            # prev_img_mask corresponds to [0,2], and cur_img_mask corresponds to [1,3].
                            # prev_img_mask[0]=gt_instances[0], prev_img_mask[1]=gt_instances[2].
                            if gt_instances[ctr*2].img_my_id[0]+1 == gt_instances[ctr*2+1].img_my_id[0]:
                                # need to be 2 consecutive frames.
                                cur_img_mask_one = cur_img_mask[ctr]
                                flow_data_img_one = gt_instances[ctr*2].flow_data
                                x_warped_coord = flow_data_img_one[0,0,:,:] + x_grid_downsample
                                y_warped_coord = flow_data_img_one[0,1,:,:] + y_grid_downsample
                                
                                # below deals with instance level mask.
                                for k in prev_img_mask_one.keys():
                                    if k in cur_img_mask_one.keys():
                                        # ensure same instance appears in both prev_im and cur_img.
                                        prev_inst_mask_one = prev_img_mask_one[k]
                                        cur_inst_mask_one = cur_img_mask_one[k]
                                        # cur_inst_mask_one_no_flip = cur_inst_mask_one
                                        if gt_instances[ctr*2].flip_aug[0] != gt_instances[ctr*2+1].flip_aug[0]:
                                            # cur_inst_mask_one_no_flip = cur_inst_mask_one
                                            cur_inst_mask_one = torch.flip(cur_inst_mask_one,dims=[1])
                                            
                                        obj_prev_coord = torch.nonzero(prev_inst_mask_one > self.pairwarp_conf_th)
                                        y_obj_warped_coord = [y_warped_coord[pair[0].item(), pair[1].item()] for pair in obj_prev_coord]
                                        x_obj_warped_coord = [x_warped_coord[pair[0].item(), pair[1].item()] for pair in obj_prev_coord]
                                        # flow_data_img_one_HWC = flow_data_img_one[0].permute([1,2,0])
                                        # flow_data_inst_one = [flow_data_img_one_HWC[pair[0].item(), pair[1].item()] for pair in obj_prev_coord]
                                        # deal with values that exceed the axis range.
                                        y_obj_warped_coord = [torch.tensor(0).float().to(device=item_one.device) if item<0 else (torch.tensor(downsample_img_h-1).float().to(device=item_one.device) if item>downsample_img_h-1 else item) for item in y_obj_warped_coord]
                                        x_obj_warped_coord = [torch.tensor(0).float().to(device=item_one.device) if item<0 else (torch.tensor(downsample_img_w-1).float().to(device=item_one.device) if item>downsample_img_w-1 else item) for item in x_obj_warped_coord]
                                        y_obj_warped_coord_int = [round(ele.item()) for ele in y_obj_warped_coord]
                                        x_obj_warped_coord_int = [round(ele.item()) for ele in x_obj_warped_coord]
                                        y_x_tuple = list(zip(y_obj_warped_coord_int,x_obj_warped_coord_int))
                                        # remove repeated warped coordinates and keep order,
                                        # list(set(y_x_tuple)) can't keep order.
                                        y_x_tuple = list(dict.fromkeys(y_x_tuple))
                                        
                                        # note that when do index on img, should use img[y_obj_warped_coord,x_obj_warped_coord],
                                        # since the first dim is height, and the second dim is width.
                                        # pair_cur_inst_mask_one=[cur_inst_mask_one[int(y_obj_warped_coord[ctr].item()), int(x_obj_warped_coord[ctr].item())] for ctr in range(len(y_obj_warped_coord))]
                                        pair_cur_inst_mask_one = [cur_inst_mask_one[item] for item in y_x_tuple]
                                        if torch.count_nonzero(torch.tensor(pair_cur_inst_mask_one) > self.pairwarp_conf_th) < self.min_nonzero_num:
                                            # ignore when the warped inst mask doesn't have positive label,
                                            # this usually happens on pedestrian, where optical flow tracking offset is not accurate enough,
                                            # and the pedestrian area is also small,  
                                            # which makes the chance that warped inst mask has overlap with cur inst mask lower.
                                            continue
                                        pair_cur_inst_mask_list.extend(pair_cur_inst_mask_one)
                    else:
                        logger.info(f'\n No Enough and Equal Number Masks in This Batch! Length of cur_img_mask: {len(cur_img_mask)}. Length of prev_img_mask: {len(prev_img_mask)}.')
                        pair_cur_inst_mask_list = []
                
                    if len(pair_cur_inst_mask_list)>0:
                        # below get loss_pairwarp
                        pred_logits = torch.tensor(pair_cur_inst_mask_list).to(device=pair_cur_inst_mask_list[0].device)
                        target_label = torch.ones(len(pair_cur_inst_mask_list)).to(device=pred_logits.device)
                        loss_pairwarp = self.pairwarp_loss_fn(pred_logits,target_label)
                        loss_pairwarp = loss_pairwarp*warmup_factor
                        loss_pairwarp = self.pairwarp_loss_factor*loss_pairwarp
                        
                        # loss_flow = loss_flow * warmup_factor
                        # loss_flow=self.flow_loss_factor*loss_flow
                        losses.update({"loss_pairwarp": loss_pairwarp})
                        self.last_loss = loss_pairwarp
                    else:
                        if self.last_loss is not None:
                            losses.update({"loss_pairwarp": self.last_loss})
                        
                    losses.update({
                        "loss_prj": loss_prj_term,
                    })
                    if self.use_color_sim:
                        losses.update({
                            "loss_pairwise": loss_pairwise,
                        })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances
