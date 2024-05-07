# @File: SSIS/my_code_for_add_noise_to_shadow_det/crop_shadow_vis_result.py 
# @Author: cws 
# @Create Date: 2024/02/28
# @Desc: Crop the shadow vis result.

from pathlib import Path
from natsort import os_sorted
from tqdm import tqdm
from PIL import Image
import os

def main():
    img_dir = "vis_shadow_result"
    img_list = list(Path(img_dir).glob("**/*.png"))
    img_list = os_sorted([str(x) for x in img_list])
    for img_src_one in tqdm(img_list):
        img = Image.open(img_src_one)
        w, h = img.size
        # left: 750, upper: 170,
        # right: 980, lower: 310.
        crop_area = (750, 170, 980, 310)
        img_cropped = img.crop(crop_area)
        path_splits = img_src_one.split('/')
        path_splits[1] = path_splits[1]+'_after_crop'
        out_path = '/'.join(path_splits)
        os.makedirs(Path(out_path).parent, exist_ok=True)
        img_cropped.save(out_path)
    print('kk')

if __name__=="__main__":
    main()
