# @File: StrongSORT/my_code_for_track_plot/plot_track_by_json.py 
# @Author: cws 
# @Create Date: 2024/04/24
# @Desc: Plot the tracking results by the maskfreevis track json files.

import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
from pathlib import Path
from tqdm import tqdm


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

def deal_with_one_seq(video_data, video_name, track_dict, img_dir, colors, out_dir_seq_one, conf_thresh):
    category_labels = video_data['pred_labels']
    pred_scores = video_data['pred_scores']
    pred_rles = video_data['pred_rles']
    
    # person class id is 0 in maskfreevis json result.
    # only draw person class to compare with kitti pretrain method.
    person_track_ids = []
    for i, category_label in enumerate(category_labels):
        if category_label == 0 and pred_scores[i] > conf_thresh:
            person_track_ids.append(i) 
    
    if len(person_track_ids) == 0:
        # this means no person confident track result in the video.
        return
    
    person_confident_rles = [pred_rles[x] for x in person_track_ids]
    
    # process each reack.
    for obj_ctr, rle_track_one in enumerate(person_confident_rles):
        obj_color = colors[obj_ctr]
        
        # process each frame of the track.
        for frame_ctr, rle_frame_one in enumerate(rle_track_one):
            # ignore the frame if the frame does not have mask.
            if rle_frame_one is None:
                continue
            
            # because the frame_id in track_dict.json starts from 1.
            frame_id = frame_ctr + 1
            binary_mask = cocomask.decode(rle_frame_one)
            
            img_filename = track_dict[video_name][str(frame_id)]
            img = draw_mask_on_image(img_dir, img_filename, binary_mask, obj_color)
            
            out_path = os.path.join(out_dir_seq_one, Path(img_filename).stem+'.png')
            img.save(out_path)
    
    
def main():
    track_dict = load_json('track_dict.json')
    img_dir = "../dataset/youtube_vis_2019/valid_for_VIS/JPEGImages"
    
    json_result_file = "../my_data_for_ytvis_2019_maskfreevis/track_result_selected_val_videos/coco_box_only_r50_0425_json_output/val_set_selected_videos_predictions.json"
    out_dir = "../my_data_for_ytvis_2019_maskfreevis/track_result_selected_val_videos/coco_box_only_r50_0425_vis/class_person"
    json_data = load_json(json_result_file)
    
    # track conf thresh for drawing tracks.
    conf_thresh = 0.5
    # color order: RGB.
    color_pink = [255/255.0, 102/255.0, 178/255.0]
    color_red = [1,0,0]
    color_green = [0,1,0]
    color_blue = [0,0,1]
    color_cyan = [0,1,1]
    color_magenta = [1,0,1]
    color_yellow = [1,1,0]
    colors = [color_pink, color_red, color_green, color_blue, color_cyan, color_magenta, color_yellow]
    
    for video_name, video_data in tqdm(json_data.items()):
        out_dir_video = os.path.join(out_dir, video_name)
        os.makedirs(out_dir_video, exist_ok=True)
        
        deal_with_one_seq(video_data, video_name, track_dict, img_dir, colors, out_dir_video, conf_thresh)
    print('kk')

if __name__ == '__main__':
    main()
