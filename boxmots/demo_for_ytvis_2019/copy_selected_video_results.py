# @File: AdelaiDet/demo_for_ytvis_2019/copy_selected_video_results.py 
# @Author: cws 
# @Create Date: 2024/04/23
# @Desc: Copy the selected video results to a new folder for deepsort.

import os
import shutil

def main():
    # these videos are selected by hand based on the visual results.
    selected_videos = ["0b97736357", "1f1396a9ef", "4b1a561480", "06a5dfb511", "7a72130f21",
                       "30fe0ed0ce", "69c0f7494e", "b4e75872a0", "b005747fee", "bbbcd58d89",
                       "d1dd586cfd", "fb104c286f"]
    
    full_results_dir = "use_kitti_pretrain_model/youtube_vis_2019_out_pair_warp/valid_for_VIS/JPEGImages"
    
    for video_one in selected_videos:
        video_one_dir = os.path.join(full_results_dir, video_one)
        video_one_out_dir = os.path.join("use_kitti_pretrain_model_selected_videos/youtube_vis_2019_out_pair_warp/valid_for_VIS/JPEGImages", video_one)
        
        shutil.copytree(video_one_dir, video_one_out_dir)
    
    print('kk')

if __name__ == '__main__':
    main()
