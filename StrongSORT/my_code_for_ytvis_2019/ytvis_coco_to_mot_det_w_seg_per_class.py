# @File: StrongSORT/my_code_for_ytvis_2019/ytvis_coco_to_mot_det_w_seg_per_class.py 
# @Author: cws 
# @Create Date: 2024/04/21
# @Desc: Convert the combined coco json result to txt file for each class in each video.
# Based on: my_code_bdd/bdd_coco_to_mot_det_w_seg_per_class.py.

import json
import os
from tqdm import tqdm
from pathlib import Path

val_img_filename_dict = json.load(open('val_img_filename_to_frame_id_video_id_dict.json', 'r'))

def load_txt(file_name):
    with open(file_name,'r') as f:
        txt_content=f.read().splitlines()
    return txt_content

def get_seqs_from_json(json_content_all):
    seq_list = []
    for item_one in json_content_all:
        img_filename = item_one['image_id']
        seq_id = str(Path(img_filename).parent)
        if seq_id not in seq_list:
            seq_list.append(seq_id)
    return seq_list

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

    coco_result_path = "../my_data_for_ytvis_2019/youtube_vis_2019_out_pair_warp/valid_for_VIS/combined_result_all_videos/combined_coco_instances_results.json"
    with open(coco_result_path, 'r') as f_in:
        json_content_all = json.load(f_in)

    # car:1,pedestrian:2.
    # category_dict = {'car': 1, 'pedestrian': 2}
    # class id in coco json result starts from 1.
    category_dict = {"person":1, "giant_panda":2, "lizard":3, "parrot":4, "skateboard":5,
            "sedan":6, "ape":7, "dog":8, "snake":9, "monkey":10, 
            "hand":11, "rabbit":12, "duck":13, "cat":14, "cow":15,
            "fish":16, "train":17, "horse":18, "turtle":19, "bear":20, 
            "motorbike":21, "giraffe":22, "leopard":23, "fox":24, "deer":25,
            "owl":26, "surfboard":27, "airplane":28, "truck":29, "zebra":30,
            "tiger":31, "elephant":32, "snowboard":33, "boat":34, "shark":35,
            "mouse":36, "frog":37, "eagle":38, "earless_seal":39, "tennis_racket":40}
    
    # load the full validation video id list.
    val_video_id_list = load_txt("valid_set_video_names_for_VIS.txt")
    seqmap=val_video_id_list

    coco_result_dir = str(Path(coco_result_path).parent)
    outdir = coco_result_dir.replace("my_data_for_ytvis_2019", "my_data_for_ytvis_2019_track_result")
    outdir = os.path.join(outdir, "to_mots_txt/mots_det_seg/")
    for cat, cat_id in tqdm(category_dict.items()):
        outdir_cat = outdir+cat+'/'
        os.makedirs(outdir_cat, exist_ok=True)
        json_content_class_one = list(
            filter(lambda x: x['category_id'] == cat_id, json_content_all))
        for seq_one in tqdm(seqmap, leave=False):
            outpath = outdir_cat+seq_one+'.txt'
            process_seq_one_w_seg(seq_one, json_content_class_one, outpath)

    print('kk')

