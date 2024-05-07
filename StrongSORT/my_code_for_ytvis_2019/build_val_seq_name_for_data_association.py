# @File: StrongSORT/my_code_for_ytvis_2019/build_val_seq_name_for_data_association.py 
# @Author: cws 
# @Create Date: 2024/04/22
# @Desc: Build validation set video names for deepsort.
# This needed in opts_for_ytvis_2019.py.

import json
from natsort import os_sorted

def load_json(json_file):    
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_txt(data, txt_file):
    with open(txt_file, 'w') as f:
        f.write('\n'.join(data))

def main():
    valid_label_file = "../dataset/youtube_vis_2019/annotations/valid.json"
    valid_data = load_json(valid_label_file)
    
    videos = valid_data['videos']
    
    first_frame_filenames = [x['file_names'][0] for x in videos]
    video_names = [x.split('/')[0] for x in first_frame_filenames]
    video_names_sorted = os_sorted(video_names)
    
    save_txt(video_names_sorted, "valid_set_video_names_for_VIS.txt")
    print('kk')

if __name__ == '__main__':
    main()
