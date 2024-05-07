import numpy as np
import json,os
import pycocotools.mask as cocomask
from tqdm import tqdm
from pathlib import Path

#rate 0.9:46.0
#rate 0.3:46.2
def get_json_file(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content
def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)
# class: 1-object,2-shadow.
# category_id: 0-object,1-shadow.
boxinst_filepath = "my_shadow_det/original_boxinst_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_ckpt_500_steps_4k/inference/iter_0002839/coco_instances_results.json"
shadow_filepath = "my_shadow_det/add_noise_to_shadow_det_kitti_mots_val/shadow_instance_result_erode_kernel_11_iter_1.json"
boxinst_content = get_json_file(boxinst_filepath)
shadow_content = get_json_file(shadow_filepath)
img_id_list = [x['image_id'] for x in boxinst_content]
img_id_list = list(dict.fromkeys(img_id_list))
boxinst_filter_all = []
shadow_rate_th = 0.2
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
    boxinst_filter_all.extend(boxinst_img_one)
json_outpath = 'my_shadow_det/combine_noisy_shadow_with_boxmots_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_ckpt_500_steps_4k/inference/iter_0002839/shadow_filter_result_rate_0'+str(int(shadow_rate_th*10))+'_erode_kernel_11_iter_1.json'
os.makedirs(str(Path(json_outpath).parent),exist_ok=True)
write_json_file(boxinst_filter_all,json_outpath)

