# @File: StrongSORT/my_code_for_ytvis_2019/ytvis_generate_submission.py 
# @Author: cws 
# @Create Date: 2024/04/22
# @Desc: Generate submission json file for ytvis 2019 validation dataset.

# The submission json format:
# a list of dict, each dict has 4 keys: video_id, score, category_id, segmentations.
# Each dict in the list represents a unique object track.
# The segmentations is a list of dict, each dict has 2 keys: size, counts.
# Each dict in the segmentations list represents an object mask on that frame.
# The dict is None when there is no mask of that object on that frame.
# The frame order is based on the index of the dict in the segmentations list.
# The first index 0 of the dict corresponds to the first frame of the video.

import os
import json
from pathlib import Path
from natsort import os_sorted
import numpy as np
from tqdm import tqdm

def load_txt(txt_file):
    with open(txt_file,'r') as f:
        content=f.read().splitlines()
    return content

def load_json(json_file):
    with open(json_file,'r') as f:
        content = json.load(f)
    return content

def save_json(json_file, data):
    with open(json_file,'w') as f:
        json.dump(data, f)

def process_category_one_seq_one(seq_one_result, video_id, video_length, category_id):
    # txt format: [frame_index object_id category_id mask_score rle_size_height rle_size_width rle_counts].
    seq_one_result = [x.split(' ') for x in seq_one_result]
    seq_one_result = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), int(x[4]), int(x[5]), x[6]] for x in seq_one_result]
        
    # get the unique object id.
    object_id_list = list(set([x[1] for x in seq_one_result]))
    
    track_all_list = []
    # process the result for each object id.
    for object_id in object_id_list:
        object_id_result = [x for x in seq_one_result if x[1] == object_id]
        
        # sort the result based on the frame index.
        object_id_result = sorted(object_id_result, key=lambda x: x[0])
        
        # get the track score from the mask scores.
        score_list = [x[3] for x in object_id_result]
        track_score = float(np.array(score_list).mean())
        
        # get the segmentations.
        segmentations = []
        for ctr in range(video_length):
            frame_id = ctr + 1
            object_frame_one = [x for x in object_id_result if x[0] == frame_id]
            
            if len(object_frame_one) == 0:
                segmentations.append(None)
                continue
            
            # object_frame_one is a list of list,
            # so we need to get the only element of the list.
            object_frame_one = object_frame_one[0]
            rle_size = [object_frame_one[4], object_frame_one[5]]
            rle_counts = object_frame_one[6]
            
            segmentations.append({'size': rle_size, 'counts': rle_counts})
        
        # generate the submission json.
        track_one_dict = {'video_id': video_id, 'score': track_score, 'category_id': category_id, 'segmentations': segmentations}
        track_all_list.append(track_one_dict)
    return track_all_list
    
def main():
    seg_track_result_dir = "../my_data_for_ytvis_2019_track_result/youtube_vis_2019_out_pair_warp/valid_for_VIS/combined_result_all_videos/to_mots_txt/mots_seg_track_w_mask_score/DeepSORT/mask_min_det_conf_04_min_det_conf_04_max_cos_0.6_no_kalman_gate/"
    out_path = "../my_data_for_ytvis_2019_track_result/youtube_vis_2019_out_pair_warp/valid_for_VIS/combined_result_all_videos/valid_submission_result/mask_min_det_conf_04_min_det_conf_04_max_cos_0.6_no_kalman_gate/results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    category_list = os.listdir(seg_track_result_dir)
    
    video_name_dict = load_json('val_video_name_to_video_id_video_length_dict.json')
    category_dict = {"person":1, "giant_panda":2, "lizard":3, "parrot":4, "skateboard":5,
                     "sedan":6, "ape":7, "dog":8, "snake":9, "monkey":10, 
                     "hand":11, "rabbit":12, "duck":13, "cat":14, "cow":15,
                     "fish":16, "train":17, "horse":18, "turtle":19, "bear":20, 
                     "motorbike":21, "giraffe":22, "leopard":23, "fox":24, "deer":25,
                     "owl":26, "surfboard":27, "airplane":28, "truck":29, "zebra":30,
                     "tiger":31, "elephant":32, "snowboard":33, "boat":34, "shark":35,
                     "mouse":36, "frog":37, "eagle":38, "earless_seal":39, "tennis_racket":40}
    
    result_all = []
    for category_one in tqdm(category_list):
        category_one_dir = os.path.join(seg_track_result_dir, category_one)
        seq_list = list(Path(category_one_dir).glob('*.txt'))
        seq_list = os_sorted([str(x) for x in seq_list])
        
        for seq_one in seq_list:            
            seq_one_result = load_txt(seq_one)
            
            # ignore the empty txt result.
            if len(seq_one_result) == 0:
                continue
            
            video_id = video_name_dict[Path(seq_one).stem]['video_id']
            video_length = video_name_dict[Path(seq_one).stem]['video_length']
            category_id = category_dict[category_one]
            
            category_one_seq_one_result = process_category_one_seq_one(seq_one_result, video_id, video_length, category_id)
            result_all.extend(category_one_seq_one_result)
    
    result_all_sorted = sorted(result_all, key=lambda x: x['video_id'])
    save_json(out_path, result_all_sorted)
    print('kk')

if __name__ == '__main__':
    main()
