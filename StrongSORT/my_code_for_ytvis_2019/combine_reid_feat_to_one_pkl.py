# @File: StrongSORT/my_code_for_ytvis_2019/combine_reid_feat_to_one_pkl.py 
# @Author: cws 
# @Create Date: 2024/04/21
# @Desc: Combine the reid feat of each video to one pkl file.

import pickle
import os
from natsort import os_sorted
from tqdm import tqdm

def load_pkl(file_path):
    with open(file_path,"rb") as f:
        pkl_content_ori=pickle.load(f)
    return pkl_content_ori

def save_pkl(pkl_content,file_path):
    with open(file_path,"wb") as f:
        pickle.dump(pkl_content,f)

def main():
    # below for no pair warp.
    # reid_result_dir = "../my_data_for_ytvis_2019/youtube_vis_2019_out_no_pair_warp/valid_for_VIS/JPEGImages"
    # below for pair warp.
    reid_result_dir = "../my_data_for_ytvis_2019/youtube_vis_2019_out_pair_warp/valid_for_VIS/JPEGImages"
    
    out_path = reid_result_dir.replace("JPEGImages", "combined_result_all_videos/combined_reid_infer_out.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    seq_list = os_sorted(os.listdir(reid_result_dir))
    
    seq_no_reid_feat_list = []
    
    for seq_one in tqdm(seq_list):
        reid_result_seq_one = os.path.join(reid_result_dir, seq_one, "reid_out/reid_infer_out.pkl")
        
        if not os.path.isfile(reid_result_seq_one):
            # this menas the whole seq got no reid result.
            seq_no_reid_feat_list.append(seq_one)
            continue
        
        pkl_content_seq_one = load_pkl(reid_result_seq_one)
        if seq_one == seq_list[0]:
            pkl_content_all = pkl_content_seq_one
        else:
            pkl_content_all += pkl_content_seq_one
    
    save_pkl(pkl_content_all, out_path)
    # no pair warp: no reid feat in seqs: ['3fc9e61451', 'df4750a8fe', 'eb49ce8027']
    # pair warp: no reid feat in seqs: ['9a38b8e463', '92fde455eb', 'df4750a8fe', 'eb49ce8027'].
    print('no reid feat in seqs: ', seq_no_reid_feat_list)
    print('kk')

if __name__=="__main__":
    main()
