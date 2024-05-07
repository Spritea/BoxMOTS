import json

def load_json(json_file):
    with open(json_file,'r') as f:
        content=json.load(f)
    return content

json_file = "../dataset/youtube_vis_2019/valid_submission_sample/results.json"
json_data = load_json(json_file)
json_data_2 = load_json("../my_data_for_ytvis_2019_track_result/youtube_vis_2019_out_no_pair_warp/valid_for_VIS/combined_result_all_videos/valid_submission_result/results.json")
print('kk')