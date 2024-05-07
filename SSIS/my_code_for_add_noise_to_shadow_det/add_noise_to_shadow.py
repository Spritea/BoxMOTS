# @File: AdelaiDet/my_code/for_shadow/add_noise_to_shadow.py 
# @Author: cws 
# @Create Date: 2024/02/13
# @Desc: Add noise to shadow detection result,
# by using erosion on the original shadow detection result.

import cv2
import json
import pycocotools.mask as cocomask
import numpy as np
import os
from tqdm import tqdm

def get_json_file(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content

def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)
        
def main():
    shadow_filepath = "my_shadow_det/my_shadow_result/my_json_result/kitti_mots_val/shadow_instance_result.json"
    json_outdir = "my_shadow_det/add_noise_to_shadow_det_kitti_mots_val"
    os.makedirs(json_outdir, exist_ok=True)
    shadow_content = get_json_file(shadow_filepath)
    iter_count = 1
    # erode kernel-3,5,7,9,11.
    kernel_size = 11
    kernel = np.ones((kernel_size,kernel_size), dtype=np.uint8)
    for ctr, shadow_inst_one in enumerate(tqdm(shadow_content)):
        # class: 1-object,2-shadow.
        # category_id: 0-object,1-shadow.
        if shadow_inst_one['category_id']==1:
            shadow_mask_img_one = shadow_inst_one['segmentation']
            # shadow_mask_img_one_decode type is np.uint8.
            shadow_mask_img_one_decode = cocomask.decode(shadow_mask_img_one)
            # cv2.imwrite('shadow_before_erode.png', shadow_mask_img_one_decode*255)
            shadow_mask_after_erode_decode = cv2.erode(shadow_mask_img_one_decode, kernel, iterations=iter_count)
            # cv2.imwrite('shadow_after_erode_kernel_11_iter_1.png', shadow_mask_after_erode*255)
            shadow_mask_after_erode_encode = cocomask.encode(np.asfortranarray(shadow_mask_after_erode_decode))
            # decode('UTF-8')-convert type bytes to string.
            shadow_mask_after_erode_encode['counts'] = shadow_mask_after_erode_encode['counts'].decode('UTF-8')
            # note that shadow_inst_one['segmentation'] = shadow_mask_after_erode_encode in fact 
            # already modifies the content of variable shadow_content.
            shadow_inst_one['segmentation'] = shadow_mask_after_erode_encode
            shadow_content[ctr] = shadow_inst_one
    json_name = f"shadow_instance_result_erode_kernel_{kernel_size}_iter_{iter_count}.json"
    json_full_path = os.path.join(json_outdir, json_name)
    write_json_file(shadow_content, json_full_path)
    print('kk')

if __name__=="__main__":
    main()
