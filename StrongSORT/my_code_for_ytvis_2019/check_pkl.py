import numpy as np
import pickle

file_path = "../my_data_for_ytvis_2019/youtube_vis_2019_out_no_pair_warp/valid_for_VIS/JPEGImages/0a49f5265b/reid_out/reid_infer_out.pkl"
with open(file_path,'rb') as f:
    pkl_content=pickle.load(f)
print('kk')