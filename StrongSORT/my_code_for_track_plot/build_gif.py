# @File: StrongSORT/my_code_for_track_plot/build_gif.py 
# @Author: cws 
# @Create Date: 2024/04/24
# @Desc: Build gif from the track vis images.

import os
from pathlib import Path
import imageio
from tqdm import tqdm

def main():
    # img_dir = "../my_data_for_ytvis_2019_kitti_pretrain_track_result_vis/use_kitti_pretrain_model_selected_videos/class_person"
    img_dir = "../my_data_for_ytvis_2019_maskfreevis/track_result_selected_val_videos/coco_box_only_r50_0425_vis/class_person"
    out_dir = img_dir.replace('class_person', 'class_person_gif')
    os.makedirs(out_dir, exist_ok=True)
    
    video_list = os.listdir(img_dir)
    for video_name in tqdm(video_list):
        out_path = os.path.join(out_dir, video_name)
        
        video_path = os.path.join(img_dir, video_name)
        img_path_list = list(Path(video_path).glob('*.png'))
        img_path_list = sorted([str(x) for x in img_path_list])
        
        frame_list = []
        for img_one in img_path_list:
            frame_one = imageio.imread(img_one)
            frame_list.append(frame_one)

        imageio.mimsave(out_path + ".gif", frame_list, fps=5)
    print('kk')

if __name__ == '__main__':
    main()
