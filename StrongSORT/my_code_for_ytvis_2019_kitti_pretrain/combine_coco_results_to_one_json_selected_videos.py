# @File: StrongSORT/my_code_for_ytvis_2019/combine_coco_results_to_one_json_selected_videos.py 
# @Author: cws 
# @Create Date: 2024/04/23
# @Desc: Combine separate coco json files to one json file.

import json
import os
from natsort import os_sorted
from tqdm import tqdm

def load_json(file_name):
    with open(file_name,'r') as f:
        json_content=json.load(f)
    return json_content

def save_json(json_content, file_name):
    with open(file_name, 'w') as f:
        json.dump(json_content, f, indent=2)

def process_seq_one_json(json_content_seq_one, seq_id):
    # set the image_id to the full image filename.
    # because we need to get the frame_id from the image_id 
    # in my_code_for_ytvis_2019/ytvis_coco_to_mot_det_w_seg_per_class.py.
    for item_one in json_content_seq_one:
        image_name = item_one['image_id']
        image_full_filename = seq_id + '/' + image_name + '.jpg'
        item_one['image_id'] = image_full_filename
    return json_content_seq_one

def main():
    coco_result_dir = "../my_data_for_ytvis_2019_kitti_pretrain/use_kitti_pretrain_model_selected_videos/youtube_vis_2019_out_pair_warp/valid_for_VIS/JPEGImages"
    out_path = coco_result_dir.replace("JPEGImages", "combined_result_all_videos/combined_coco_instances_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seq_list = os_sorted(os.listdir(coco_result_dir))
    
    seq_no_det_list = []
    
    for seq_one in tqdm(seq_list):
        coco_result_seq_one = os.path.join(coco_result_dir, seq_one, "coco_json_out/coco_instances_results.json")
        
        if not os.path.isfile(coco_result_seq_one):
            # this menas the whole seq got no object det.
            seq_no_det_list.append(seq_one)
            continue
        
        json_content_seq_one = load_json(coco_result_seq_one)
        # in fact, json_content_seq_one is modified inline,
        # so json_content_seq_one_modified is the same with json_content_seq_one after calling func process_seq_one_json().
        json_content_seq_one_modified = process_seq_one_json(json_content_seq_one, seq_one)
        if seq_one == seq_list[0]:
            json_content_all = json_content_seq_one_modified
        else:
            json_content_all += json_content_seq_one_modified
            
    save_json(json_content_all, out_path)
    print('no object detected in seqs: ', seq_no_det_list)
    print('kk')

if __name__ == "__main__":
    main()
