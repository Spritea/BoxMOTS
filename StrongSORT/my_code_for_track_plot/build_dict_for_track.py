# @File: StrongSORT/my_code_for_track_plot/build_dict_for_track.py
# @Author: cws 
# @Create Date: 2024/04/24
# @Desc: Build the dict for track plot.
# The dict maps video and frame id in track txt to the img filename.

import os
import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    json_path = "../my_code_for_ytvis_2019/val_img_filename_to_frame_id_video_id_dict.json"
    out_path = "track_dict.json"
    json_data = load_json(json_path)
    
    track_dict = {}
    for img_filename, v in json_data.items():
        video_name = img_filename.split('/')[0]
        frame_id = v['frame_id']
        
        # the frame_id in the track txt starts from 1.
        frame_id_track_txt = frame_id + 1
        
        if video_name not in track_dict:
            track_dict[video_name] = {frame_id_track_txt: img_filename}
        else:
            track_dict[video_name].update({frame_id_track_txt: img_filename})
    
    save_json(track_dict, out_path)    
    print('kk')

if __name__ == '__main__':
    main()
