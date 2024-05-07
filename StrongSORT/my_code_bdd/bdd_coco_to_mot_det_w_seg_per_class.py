import json
import os
from tqdm import tqdm
from bdd_category_dict import bdd_val_img_id_dict

# this file is from E:/code/Base/12_mots_data_process/coco_to_mot_det_w_seg_per_class.py.
# this converts coco json result to txt file with rle mask.


def process_seq_one_w_seg(seq_id, json_content_class_one, outpath):
    json_content_seq_one = list(filter(lambda x: bdd_val_img_id_dict[x['image_id']]['video_id'] == seq_id, json_content_class_one))
    with open(outpath, 'w') as f_out:
        for item_one in json_content_seq_one:
            # frame_id = int(str(item_one['image_id'])[-4:])
            # frame_id needs to start from 1 to be the same with img filename for deep_sort.
            frame_id = bdd_val_img_id_dict[item_one['image_id']]['frame_id']+1
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

    coco_result_path = "../my_data_bdd/reid_one_class_infer_for_train/repeat_results/long_epoch_seq_shuffle_fl_2_lr_0_0001_eval_500_steps_4k/CondInst_MS_R_50_1x_kitti_mots/inference/iter_0020999/coco_instances_results.json"
    with open(coco_result_path, 'r') as f_in:
        json_content_all = json.load(f_in)

    # car:1,pedestrian:2.
    # category_dict = {'car': 1, 'pedestrian': 2}
    category_dict={'pedestrian':1,'rider':2,'car':3,'truck':4,'bus':5,'motorcycle':7,'bicycle':8}
    val_video_id_list=list(range(1,33))
    seqmap=val_video_id_list

    outdir = "../my_data_bdd/reid_one_class_infer_for_train/repeat_results/long_epoch_seq_shuffle_fl_2_lr_0_0001_eval_500_steps_4k/CondInst_MS_R_50_1x_kitti_mots/to_mots_txt/iter_0020999/mots_det_seg/val_set/"
    for cat, cat_id in category_dict.items():
        outdir_cat = outdir+cat+'/'
        os.makedirs(outdir_cat, exist_ok=True)
        json_content_class_one = list(
            filter(lambda x: x['category_id'] == cat_id, json_content_all))
        for seq_one in tqdm(seqmap):
            outpath = outdir_cat+str(seq_one).zfill(4)+'.txt'
            process_seq_one_w_seg(seq_one, json_content_class_one, outpath)

    print('kk')

