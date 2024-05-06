import numpy as np
import json,os
import pycocotools.mask as cocomask
from tqdm import tqdm
from pathlib import Path


def get_json_file(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content
def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)

def combine_shadow_one_class(cat_list):
    cat_list = [cat_dict[x] for x in cat_use_shadow]
    for img_id_one in tqdm(img_id_list):
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
                # comment below for all classes.
                if boxinst_inst_one['category_id'] in cat_list:
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
        boxinst_filter_all.extend(boxinst_img_one)
    json_outpath = '../../my_shadow_det/filter_result/bdd_data/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots/inference/iter_0014999/by_class/shadow_filter_result_rate_0'+\
                    str(int(shadow_rate_th*10))+'_'+'_'.join(cat_use_shadow)+'.json'
    os.makedirs(str(Path(json_outpath).parent),exist_ok=True)
    write_json_file(boxinst_filter_all,json_outpath)

if __name__=='__main__':
    # class: 1-object,2-shadow.
    # category_id: 0-object,1-shadow.
    cat_dict = {'pedestrian':1,'rider':2,'car':3,'truck':4,'bus':5,'motorcycle':7,'bicycle':8}
    boxinst_filepath = "../../my_shadow_det/original_boxinst_result/bdd_data/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots/inference/iter_0014999/coco_instances_results.json"
    shadow_filepath = "../../my_shadow_det/my_shadow_result/my_json_result/bdd_mots_val/shadow_instance_result.json"
    boxinst_content = get_json_file(boxinst_filepath)
    shadow_content = get_json_file(shadow_filepath)
    img_id_list = [x['image_id'] for x in boxinst_content]
    img_id_list = list(dict.fromkeys(img_id_list))
    boxinst_filter_all = []
    shadow_rate_th = 0.2
    # cat_use_shadow = ['car','truck','bus']
    cat_use_shadow = ['car','truck']
    combine_shadow_one_class(cat_use_shadow)