# @File: SSIS/my_code_for_MOSE_shadow_combine/combine_shadow_MOSE.py 
# @Author: cws 
# @Create Date: 2024/02/26
# @Desc: Combine shadow for MOSE seq by seq.
# This file combine_shadow_MOSE.py is modified based on
# /home/wensheng/code/boxmots_on_MOSE/AdelaiDet/my_code/for_shadow/combine_shadow.py.

import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as cocomask
from natsort import os_sorted
from tqdm import tqdm


def get_json_file(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content

def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)
        
def deal_with_seq_one(boxinst_path, shadow_path, shadow_rate_th):
    boxinst_content = get_json_file(boxinst_path)
    shadow_content = get_json_file(shadow_path)
    img_id_list = [x['image_id'] for x in boxinst_content]
    img_id_list = list(dict.fromkeys(img_id_list))
    boxinst_filter_seq_one = []
    for img_id_one in tqdm(img_id_list, leave=False):
        # class: 1-object,2-shadow.
        # category_id: 0-object,1-shadow.
        # boxinst choose car class,shadow choose shadow class.
        boxinst_img_one = list(filter(lambda x:x['image_id']==img_id_one,
                                    boxinst_content))
        shadow_img_one = list(filter(lambda x:x['image_id']==img_id_one and x['category_id']==1,
                                    shadow_content))
        # below merge all shadow together.
        # cocomask.merge(xxx,intersect=false)-union,cocomask.merge(xxx,intersect=true)-intersect.
        shadow_mask_img_one = [x['segmentation'] for x in shadow_img_one]
        if len(shadow_mask_img_one)>1:
            shadow_mask_img_one = [cocomask.merge(shadow_mask_img_one,intersect=False)]
        if len(shadow_mask_img_one)>0:
            for ctr,boxinst_inst_one in enumerate(boxinst_img_one):
                # only for car class.
                if boxinst_inst_one['category_id']==1:
                    boxinst_mask_inst_one = boxinst_inst_one['segmentation']
                    intersect_area = cocomask.area(cocomask.merge([boxinst_mask_inst_one,shadow_mask_img_one[0]],intersect=True))
                    # if shadow is too large for the car, the shadow usually is wrong.
                    shadow_rate = intersect_area/cocomask.area(boxinst_mask_inst_one)
                    if intersect_area>0.0 and shadow_rate<shadow_rate_th:
                        boxinst_mask_inst_one_decode = cocomask.decode(boxinst_mask_inst_one)
                        # shadow_mask_img_one_decode type is np.uint8.
                        shadow_mask_img_one_decode = cocomask.decode(shadow_mask_img_one[0])
                        boxinst_mask_inst_one_decode[np.where(shadow_mask_img_one_decode>0)]=0
                        boxinst_mask_inst_one_encode = cocomask.encode(boxinst_mask_inst_one_decode)
                        # decode('UTF-8')-convert type bytes to string.
                        boxinst_mask_inst_one_encode['counts'] = boxinst_mask_inst_one_encode['counts'].decode('UTF-8')
                        boxinst_inst_one['segmentation'] = boxinst_mask_inst_one_encode
                        boxinst_img_one[ctr] = boxinst_inst_one
        boxinst_filter_seq_one.extend(boxinst_img_one)
    return boxinst_filter_seq_one

def main():
    boxinst_result_dir = "my_infer_result_MOSE/MOSE_car_seq_out_pair_warp/train_car_seq"
    shadow_result_dir = "../demo/my_json_result_MOSE/MOSE_train_car_seq"
    json_out_dir = "filter_result_MOSE/MOSE_car_seq_out_pair_warp/train_car_seq"
    seq_list = os_sorted(os.listdir(boxinst_result_dir))
    shadow_rate_th = 0.2
    seq_empty_det = []
    for seq_id in tqdm(seq_list):
        boxinst_filepath = os.path.join(boxinst_result_dir, seq_id, "coco_json_out/coco_instances_results.json")
        shadow_filepath = os.path.join(shadow_result_dir, seq_id, "shadow_instance_result.json")
        if not os.path.exists(boxinst_filepath):
            # one seq could have no det result in MOSE.
            # create an empty dir, and skip,
            # since the boxinst result also has an empty dir for empty det seq.
            json_out_seq_dir = os.path.join(json_out_dir, seq_id)
            os.makedirs(json_out_seq_dir, exist_ok=True)
            seq_empty_det.append(seq_id)
            continue 
        boxinst_filter_seq_one = deal_with_seq_one(boxinst_filepath, shadow_filepath, shadow_rate_th)
        json_out_seq_dir = os.path.join(json_out_dir, seq_id)
        os.makedirs(json_out_seq_dir, exist_ok=True)
        json_out_seq_path = os.path.join(json_out_seq_dir, 'shadow_filter_result_rate_0'+str(int(shadow_rate_th*10))+'.json')
        write_json_file(boxinst_filter_seq_one, json_out_seq_path)
    print('empty det seq:', seq_empty_det)
    print('kk')

if __name__=="__main__":
    main()


