import numpy as np
import json

# this file my_code/search_for_result.py is to
# search for the checkpoint that has the best result.
def compare_base(metric_list):
    # the boxinst base we built in the draft table.
    boxinst_base = {'bbox/AP-car':75.87,'bbox/AP-pedestrian':61.31,
                    'segm/AP-car':59.19,'segm/AP-pedestrian':30.87}
    filter_list = []
    for item in metric_list:
        if item['bbox/AP-car']>boxinst_base['bbox/AP-car'] and item['bbox/AP-pedestrian']>boxinst_base['bbox/AP-pedestrian'] \
            and item['segm/AP-car']>boxinst_base['segm/AP-car'] and item['segm/AP-pedestrian']>boxinst_base['segm/AP-pedestrian']:
            filter_list.append(item)
    return filter_list

def sort_metric(metric_list,sort_key):
    metric_list_sorted = sorted(metric_list,key=lambda x:x[sort_key],reverse=True)
    print(f'Sort Key is: {sort_key}')
    for line_one in metric_list_sorted[:10]:
        print(line_one)
    return metric_list_sorted

def main():
    metric_path = "../training_dir_hddb/reid_one_class_infer_pair_warp_inverse_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_steps_4k_no_color_sim_pretrain_weights_v3_time_4/BoxInst_MS_R_50_1x_kitti_mots/metrics.json"
    line_list = []
    for line in open(metric_path, 'r'):
        line_list.append(json.loads(line))
    score_list = [x for x in line_list if 'bbox/AP' in x.keys()]

    key_names = ['bbox/AP-car', 'bbox/AP-pedestrian', 'segm/AP-car',
        'segm/AP-pedestrian', 'iteration']
    metric_list = []
    for eval_one in score_list:
        metric_dict = {}
        for k, v in eval_one.items():
            if k in key_names:
                metric_dict.update({k: v})
        metric_mean= (metric_dict['bbox/AP-car'] + metric_dict['bbox/AP-pedestrian'] + metric_dict['segm/AP-car'] + metric_dict['segm/AP-pedestrian'])/4.0
        metric_dict.update({'metric_mean':metric_mean})
        metric_list.append(metric_dict)
    
    metric_list = compare_base(metric_list)
    metric_list_sorted = sort_metric(metric_list,sort_key='bbox/AP-car')
    print('KK')
if __name__ == "__main__":
    main()