# @File: my_code_for_ytvis_2019/build_val_dict_for_data_association.py
# @Author: cws 
# @Create Date: 2024/04/21
# @Desc: Build the img_filename to full info dict for data association in inference.
# The image_id in coco json result of inference demo is the image filename without seq id,
# so we need this dict to get the frame_id based on the image_id,
# when we convert the coco json result to txt files.
# This is needed in: StrongSORT/my_code_for_ytvis_2019/ytvis_coco_to_mot_det_w_seg_per_class.py.

import json

def load_json(json_file):    
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    val_json = "../dataset/youtube_vis_2019/annotations/valid.json"
    out_path = "val_img_filename_to_frame_id_video_id_dict.json"
    val_data = load_json(val_json)
    videos = val_data['videos']
    
    # build the image_file_name to {frame_id, video id} dict.
    img_dict_all = {}
    
    for video_one in videos:
        video_id = video_one['id']
        images = video_one['file_names']
        
        for frame_ctr, img_file_name in enumerate(images):
            # frame_id starts with 0.
            # video_id starts with 1.
            img_dict_all[img_file_name] = {'frame_id': frame_ctr, 'video_id': video_id}
            
    save_json(img_dict_all, out_path)
    print('kk')

if __name__ == '__main__':
    main()


