# @File: StrongSORT/my_code_for_track_plot/plot_track_by_txt.py
# @Author: cws 
# @Create Date: 2024/04/23
# @Desc: Draw the tracking results by the track txt files.

import json
import os
from pathlib import Path

import numpy as np
from natsort import os_sorted
from PIL import Image
from pycocotools import mask as cocomask
from tqdm import tqdm

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        # must remove the '\n' at the end of each line.
        lines = f.read().splitlines()
    return lines

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def apply_mask(image, mask, color, alpha=0.5, category=1):
    # the apply_mask function is modified from:
    # /home/wensheng/code/boxmots_on_MOSE/draw_track/mots_vis/visualize_mots.py.
    for c in range(3):
        image[:, :, c] = np.where(mask == category,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def draw_mask_on_image(img_dir, image_filename, mask, color):
    img_path = os.path.join(img_dir, image_filename)
    image = np.array(Image.open(img_path), dtype="float32") / 255
    image = apply_mask(image, mask, color, alpha=0.5, category=1)
    
    img_with_mask = (image*255).astype(np.uint8)
    img_with_mask = Image.fromarray(img_with_mask)
    return img_with_mask

def deal_with_one_seq(seq_one_txt, video_name, track_dict, img_dir, colors, out_dir_seq_one):
    obj_ids = [int(x.split(' ')[1]) for x in seq_one_txt]
    obj_ids = list(set(obj_ids))
    
    assert len(obj_ids) <= len(colors), 'The number of objects is more than the number of colors.'
    
    # process each track.
    for obj_ctr, obj_id in enumerate(obj_ids):
        obj_color = colors[obj_ctr]
        obj_one = [x for x in seq_one_txt if int(x.split(' ')[1]) == obj_id]
        obj_one = [x.split(' ') for x in obj_one]
        obj_one = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), int(x[4]), int(x[5]), x[6]] for x in obj_one]
        
        # process each frame of the track.
        for frame_one in obj_one:
            frame_id, obj_id, cat_id, score, frame_height, frame_width, rle_mask = frame_one
            assert cat_id == 1, 'Only process person class.'
            rle_full = {'size': [frame_height, frame_width], 'counts': rle_mask}
            binary_mask = cocomask.decode(rle_full)
            img_filename = track_dict[video_name][str(frame_id)]
            img = draw_mask_on_image(img_dir, img_filename, binary_mask, obj_color)
            out_path = os.path.join(out_dir_seq_one, Path(img_filename).stem+'.png')
            img.save(out_path)


def main():
    track_dict = load_json('track_dict.json')
    img_dir = "../dataset/youtube_vis_2019/valid_for_VIS/JPEGImages"
    
    track_dir = "../my_data_for_ytvis_2019_kitti_pretrain_track_result/use_kitti_pretrain_model_selected_videos/youtube_vis_2019_out_pair_warp/valid_for_VIS/combined_result_all_videos/to_mots_txt/mots_seg_track_w_mask_score/DeepSORT/mask_min_det_conf_04_min_det_conf_04_max_cos_0.6_no_kalman_gate/person"
    out_dir = "../my_data_for_ytvis_2019_kitti_pretrain_track_result_vis/use_kitti_pretrain_model_selected_videos/class_person/"
    track_txt_list = list(Path(track_dir).glob('*.txt'))
    track_txt_list = os_sorted([str(x) for x in track_txt_list])
    
    # color order: RGB.
    color_orange = [255/255.0, 165/255.0, 0]
    color_red = [1,0,0]
    color_green = [0,1,0]
    color_blue = [0,0,1]
    color_cyan = [0,1,1]
    color_magenta = [1,0,1]
    color_yellow = [1,1,0]
    colors = [color_orange, color_red, color_green, color_blue, color_cyan, color_magenta, color_yellow]
    
    for seq_one_txt_path in tqdm(track_txt_list):
        video_name = Path(seq_one_txt_path).stem
        out_dir_seq_one = os.path.join(out_dir, video_name)
        os.makedirs(out_dir_seq_one, exist_ok=True)
        
        seq_one_txt = load_txt(seq_one_txt_path)
        deal_with_one_seq(seq_one_txt, video_name, track_dict, img_dir, colors, out_dir_seq_one)
    print('kk')

if __name__ == '__main__':
    main()
