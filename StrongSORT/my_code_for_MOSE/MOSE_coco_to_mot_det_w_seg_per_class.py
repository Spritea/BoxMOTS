# @File: StrongSORT/my_code_for_MOSE/MOSE_coco_to_mot_det_w_seg_per_class.py 
# @Author: cws 
# @Create Date: 2024/02/24
# @Desc: Convert coco json to txt file.
# Only care about car det results, and ignore the ped det results.

import json
import os
from tqdm import tqdm
from natsort import os_sorted
from pathlib import Path

def load_json(file_name):
    with open(file_name,'r') as f:
        json_content=json.load(f)
    return json_content

def process_seq_one_w_seg(json_content_seq_one, outpath):
    # json_content_seq_one = list(filter(lambda x: (x['image_id']//10000) == seq_id, json_content_class_one))
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
    coco_result_dir = "../my_data_for_MOSE/MOSE_car_seq_out_pair_warp/valid_car_seq"
    outdir = coco_result_dir.replace("my_data_for_MOSE", "my_data_for_MOSE_track_result")
    # outdir = str(Path(outdir).parent)
    outdir = os.path.join(outdir, "to_mots_txt/mots_det_seg/car")
    os.makedirs(outdir, exist_ok=True)
    seq_list = os_sorted(os.listdir(coco_result_dir))
    seq_ped_det = []
    for seq_one in tqdm(seq_list):
        coco_result_seq_one = os.path.join(coco_result_dir, seq_one, "coco_json_out/coco_instances_results.json")
        outpath = os.path.join(outdir, seq_one+'.txt')
        if not os.path.isfile(coco_result_seq_one):
            # this menas the whole seq got no object det.
            with open(outpath, 'w') as f:
                continue
        json_content_seq_one = load_json(coco_result_seq_one)
        # car:1,pedestrian:2.
        json_content_car = list(filter(lambda x: x['category_id']==1, json_content_seq_one))
        if len(json_content_car)<len(json_content_seq_one):
            seq_ped_det.append(seq_one)
        process_seq_one_w_seg(json_content_car, outpath)
    print(f'ped detected in {len(seq_ped_det)}/{len(seq_list)} seqs')
    print(seq_ped_det)
    print('kk')
