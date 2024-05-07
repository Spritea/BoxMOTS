# @File: StrongSORT/my_code_for_ytvis_2019/rename_img_filename_consecutive.py 
# @Author: cws 
# @Create Date: 2024/04/22
# @Desc: Rename the ytvis val set image filename to consecutive numbers to match frame id.
# This is needed for deepsort, because deepsort directly use the image filename to get the frame id.
# Add folder img1 between the seq folder and the image files, needed by deepsort.

import os
import shutil
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path

def main():
    val_img_dir = "../dataset/youtube_vis_2019/valid_for_VIS/JPEGImages"
    val_list = natsorted(os.listdir(val_img_dir))
    
    for val_one in tqdm(val_list):
        img_list = list(Path(os.path.join(val_img_dir, val_one)).glob('*.jpg'))
        img_list = natsorted([str(x) for x in img_list])
        
        dst_dir = os.path.join("../my_data_for_ytvis_2019", "youtube_vis_2019_valid_for_VIS_rename_for_deepsort", val_one, 'img1')
        os.makedirs(dst_dir, exist_ok=True)
        
        for ctr, img_one_src in enumerate(tqdm(img_list, leave=False)):
            frame_id = ctr + 1
            img_one_dst = os.path.join(dst_dir, f"{frame_id:05d}.jpg")
            shutil.copy2(img_one_src, img_one_dst)

    print('kk')

if __name__=="__main__":
    main()
