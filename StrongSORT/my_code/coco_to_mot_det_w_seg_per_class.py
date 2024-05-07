import json
import os
from tqdm import tqdm

# this file is from E:/code/Base/12_mots_data_process/coco_to_mot_det_w_seg_per_class.py.
# this converts coco json result to txt file with rle mask.


def process_seq_one_w_seg(seq_id, json_content_class_one, outpath):
    json_content_seq_one = list(filter(lambda x: (x['image_id']//10000) == seq_id, json_content_class_one))
    with open(outpath, 'w') as f_out:
        for item_one in json_content_seq_one:
            frame_id = int(str(item_one['image_id'])[-4:])
            tl_x, tl_y, width, height = int(item_one['bbox'][0]), int(item_one['bbox'][1]), int(item_one['bbox'][2]), int(item_one['bbox'][3])
            conf = format(item_one['score'], '.6f')
            img_h = item_one['segmentation']['size'][0]
            img_w = item_one['segmentation']['size'][1]
            rle = item_one['segmentation']['counts']
            line = [str(frame_id), '-1', str(tl_x), str(tl_y), str(width),
                    str(height), conf, str(img_h), str(img_w), rle]
            line_str = ','.join(line)+'\n'
            f_out.write(line_str)


if __name__ == "__main__":

    coco_result_path = "../my_data/reid_one_class_infer_pair_warp_right_track/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_no_color_sim_pretrain_weights_v3_time_7/BoxInst_MS_R_50_1x_kitti_mots/inference/iter_0007919/coco_instances_results.json"
    with open(coco_result_path, 'r') as f_in:
        json_content_all = json.load(f_in)

    # car:1,pedestrian:2.
    category_dict = {'car': 1, 'pedestrian': 2}
    train_in_trainval_seqmap = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
    val_in_trainval_seqmap = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    seqmap = val_in_trainval_seqmap

    outdir = "../my_data/reid_one_class_infer_pair_warp_right_track/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_no_color_sim_pretrain_weights_v3_time_7/BoxInst_MS_R_50_1x_kitti_mots/to_mots_txt/iter_0007919/mots_det_seg/val_in_trainval/"
    for cat, cat_id in category_dict.items():
        outdir_cat = outdir+cat+'/'
        os.makedirs(outdir_cat, exist_ok=True)
        json_content_class_one = list(
            filter(lambda x: x['category_id'] == cat_id, json_content_all))
        for seq_one in tqdm(seqmap):
            outpath = outdir_cat+str(seq_one).zfill(4)+'.txt'
            process_seq_one_w_seg(seq_one, json_content_class_one, outpath)

    print('kk')
