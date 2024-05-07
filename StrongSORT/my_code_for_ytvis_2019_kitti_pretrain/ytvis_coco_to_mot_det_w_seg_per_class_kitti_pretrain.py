# @File: StrongSORT/my_code_for_ytvis_2019/ytvis_coco_to_mot_det_w_seg_per_class_kitti_pretrain.py 
# @Author: cws 
# @Create Date: 2024/04/23
# @Desc: Convert the combined coco json result to txt file for each class in each video.
# Based on: my_code_for_ytvis_2019/ytvis_coco_to_mot_det_w_seg_per_class.py

import json
import os
from tqdm import tqdm
from pathlib import Path

val_img_filename_dict = json.load(open('val_img_filename_to_frame_id_video_id_dict.json', 'r'))

def process_seq_one_w_seg(seq_id, json_content_class_one, outpath):
    json_content_seq_one = list(filter(lambda x: str(Path(x['image_id']).parent) == seq_id, json_content_class_one))
    
    if len(json_content_seq_one) == 0:
        # this menas the whole seq got no object det for this class.
        # still needs to create the empty txt file for deepsort.
        with open(outpath, 'w') as f:
            pass
        return
    
    with open(outpath, 'w') as f_out:
        for item_one in json_content_seq_one:
            # the ytvis 2019 img filename will be renamed to be continuous, and start from 0000001.jpg,
            # like 0000001.jpg, 0000002.jpg, 0000003.jpg, etc.
            # frame_id needs to start from 1 to be the same with img filename for deep_sort.
            frame_id = val_img_filename_dict[item_one['image_id']]['frame_id']+1
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

    coco_result_path = "../my_data_for_ytvis_2019_kitti_pretrain/use_kitti_pretrain_model_selected_videos/youtube_vis_2019_out_pair_warp/valid_for_VIS/combined_result_all_videos/combined_coco_instances_results.json"
    with open(coco_result_path, 'r') as f_in:
        json_content_all = json.load(f_in)

    # car:1,pedestrian:2.
    # category_dict = {'car': 1, 'pedestrian': 2}
    category_dict = {'person': 2}
    
    # set the selected validation video id list.
    val_video_id_list = ["0b97736357", "1f1396a9ef", "4b1a561480", "06a5dfb511", "7a72130f21",
                       "30fe0ed0ce", "69c0f7494e", "b4e75872a0", "b005747fee", "bbbcd58d89",
                       "d1dd586cfd", "fb104c286f"]
    seqmap=val_video_id_list

    coco_result_dir = str(Path(coco_result_path).parent)
    outdir = coco_result_dir.replace("my_data_for_ytvis_2019_kitti_pretrain", "my_data_for_ytvis_2019_kitti_pretrain_track_result")
    outdir = os.path.join(outdir, "to_mots_txt/mots_det_seg/")
    
    # only handle person class.
    cat = 'person'
    cat_id = 2
    outdir_cat = outdir+cat+'/'
    os.makedirs(outdir_cat, exist_ok=True)
    
    json_content_class_one = list(filter(lambda x: x['category_id'] == cat_id, json_content_all))
    for cat, cat_id in tqdm(category_dict.items()):
        outdir_cat = outdir+cat+'/'
        os.makedirs(outdir_cat, exist_ok=True)
        json_content_class_one = list(
            filter(lambda x: x['category_id'] == cat_id, json_content_all))
        for seq_one in tqdm(seqmap, leave=False):
            outpath = outdir_cat+seq_one+'.txt'
            process_seq_one_w_seg(seq_one, json_content_class_one, outpath)

    print('kk')

