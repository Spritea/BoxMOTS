import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
import numpy as np
import bisect

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

def get_car_mask_per_img(pred_instances,mask_scores,gt_instances):
    # below from
    # adet/modeling/condinst_reid_one_class_mask_pool_infer/reid_one_class_mask_pool_infer_branch.py.
    mask_th = 0.5
    # gt_inds is the corresponding gt index of pred_masks/pred_boxes.
    gt_inds = pred_instances.gt_inds
    gt_inds_in_use = torch.unique(gt_inds)
    mask_dict_instance = {}
    pred_masks = mask_scores
    # get the corresponding pred_mask for each gt mask.
    # note gt_inds may not contain all gt masks in one img,
    # like pred_inst would correspond to only one of two overlapping gt objects,
    # the smaller area one in fcos.
    for gt_id_instance_one in gt_inds_in_use:
        ind = (gt_inds==gt_id_instance_one).nonzero(as_tuple=True)[0]
        pred_masks_one_id = pred_masks[ind,:,:,:]
        pred_mask_avg=torch.squeeze(torch.mean(pred_masks_one_id,dim=0))
        # note pred_mask_avg could be all<mask threshold,
        # which is a full black binary map, then it should be removed,
        # otherwise it would generate nan in mask_pool_one_img function.
        if not (pred_mask_avg>mask_th).any():
            # note that mask_dict_instance could be empty for one whole batch,
            # when the pred_mask_avg<mask_th for all pred_masks.
            # logger.info("This is a full black pred mask! It's not used in mask pool reid!")
            continue
        # gt_id_instance_one.item() is to get the python version data as dict key,
        # otherwise the key would be tensor, which is hard to use.
        mask_dict_instance[gt_id_instance_one.item()] = pred_mask_avg

    # below get gt inst category.
    # 0-car,1-ped.
    gt_class_list = []
    for gt_img_one in gt_instances:
        gt_class_list.append(gt_img_one.gt_classes)
    gt_class_tensor = torch.cat(gt_class_list,dim=0)
    mask_dict_instance_car = dict(filter(lambda item:gt_class_tensor[item[0]]==0,mask_dict_instance.items()))
    # below combine masks of one image to one tensor.
    gt_count=[len(x) for x in gt_instances]
    gt_num_acc=np.cumsum(gt_count)
    mask_dict_img={}
    for gt_id_instance_one,pred_mask_instance_one in mask_dict_instance_car.items():
        img_id=bisect.bisect_right(gt_num_acc,gt_id_instance_one)
        if img_id not in mask_dict_img:
            mask_dict_img[img_id]=[pred_mask_instance_one]
        else:
            mask_dict_img[img_id].append(pred_mask_instance_one)
    return mask_dict_img

            
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

        # my boxinst loss configs
        self.use_color_sim = cfg.MODEL.BOXINST.USE_COLOR_SIMILARITY
        self.shadow_rate_th = cfg.MODEL.BOXINST.SHADOW_RATE_THRESH 
        self.last_loss = None
        # default reduction='mean'.
        self.shadow_loss_fn = nn.BCELoss()
        self.shadow_loss_factor = cfg.MODEL.BOXINST.SHADOW_LOSS_FACTOR
        self.shadow_warm_up = cfg.MODEL.BOXINST.SHADOW_WARM_UP
        
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

                    # below for my shadow loss.
                    mask_th = 0.5
                    mask_dict_img = get_car_mask_per_img(pred_instances,mask_scores,gt_instances)
                    if len(mask_dict_img)>0:
                        shadow_all_inst = []
                        pred_mask_all_inst = []
                        for k,v in mask_dict_img.items():
                            for inst_one in v:
                                shadow_all_inst.append(gt_instances[k].shadow_data[0][0])
                                pred_mask_all_inst.append(inst_one)
                        
                        shadow_all_tensor = torch.stack(shadow_all_inst,dim=0)
                        pred_mask_all_tensor = torch.stack(pred_mask_all_inst,dim=0)
                        pred_binary_mask_all_tensor = (pred_mask_all_tensor>mask_th).float()
                        intersect_all_tensor = shadow_all_tensor*pred_binary_mask_all_tensor
                        shadow_rate = torch.count_nonzero(intersect_all_tensor,dim=(1,2))/torch.count_nonzero(pred_binary_mask_all_tensor,dim=(1,2))
                        shadow_use_id = torch.where((0<shadow_rate) & (shadow_rate<self.shadow_rate_th))
                        intersect_use_tensor = intersect_all_tensor[shadow_use_id]
                        pred_mask_use_tensor = pred_mask_all_tensor[shadow_use_id]
                        pred_mask_in_shadow = pred_mask_use_tensor[intersect_use_tensor>0]
                        
                        if len(pred_mask_in_shadow)>0:
                            labels = torch.zeros_like(pred_mask_in_shadow).to(device=pred_mask_in_shadow.device)
                            loss_shadow = self.shadow_loss_fn(pred_mask_in_shadow,labels)
                            if self.shadow_warm_up:
                                loss_shadow = loss_shadow*warmup_factor
                            loss_shadow = self.shadow_loss_factor*loss_shadow
                            losses.update({"loss_shadow":loss_shadow})
                            self.last_loss = loss_shadow
                        # note: cannot use below, ptherwise would get error
                        # RuntimeError: Trying to backward through the graph a second time, 
                        # but the buffers have already been freed. 
                        # Specify retain_graph=True when calling backward the first time.
                        # Since the intermediate results were removed after the first call to loss.backward(),
                        # then if we use the loss in last time as the loss for this time,
                        # loss.backward() cannot work properly, because these intermediate results were removed.
                        # elif self.last_loss is not None:
                        #     losses.update({"loss_shadow":self.last_loss})
                        
                    losses.update({
                        "loss_prj": loss_prj_term,
                        # "loss_pairwise": loss_pairwise,
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
