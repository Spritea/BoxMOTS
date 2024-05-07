# @File: StrongSORT/my_code_for_ytvis_2019/build_val_video_name_to_video_id_dict.py 
# @Author: cws 
# @Create Date: 2024/04/22
# @Desc: Build a dict to map video name to video id and video length for ytvis 2019 validation dataset.

import json

def load_json(json_file):
    with open(json_file,'r') as f:
        content=json.load(f)
    return content

def save_json(json_file,json_data):
    with open(json_file,'w') as f:
        json.dump(json_data,f,indent=2)

def main():
    valid_json_file = "../dataset/youtube_vis_2019/annotations/valid.json"
    out_path = "val_video_name_to_video_id_video_length_dict.json"
    valid_data = load_json(valid_json_file)
    
    videos = valid_data['videos']
    
    video_name2id_dict = {}
    for video in videos:
        video_id = video['id']
        video_name = video['file_names'][0].split('/')[0]
        video_length = video['length']
        
        video_name2id_dict[video_name] = {'video_id': video_id, 'video_length': video_length}
    
    save_json(out_path, video_name2id_dict)
    print('kk')

if __name__ == '__main__':
    main()
