import numpy as np
import json
import pycocotools.mask as cocomask
from tqdm import tqdm

# this file my_code/for_shadow/combine_shadow_use_object.py
# first finds the correspongind object in the shadow result,
# then finds the associated shadow of that object.

def get_json_file(filepath):
    with open(filepath, 'r') as f:
        json_content = json.load(f)
    return json_content
def write_json_file(json_list, filepath):
    with open(filepath, 'w') as f:
        json.dump(json_list, f, indent=2)

# class: 1-object,2-shadow.
# category_id: 0-object,1-shadow.
boxinst_filepath = "../../my_shadow_det/origial_boxinst_result/inference/iter_0003999/coco_instances_results.json"
shadow_filepath = "../../my_shadow_det/my_shadow_result/my_json_result/kitti_mots_val/shadow_instance_result_out_assc.json"
boxinst_content = get_json_file(boxinst_filepath)
shadow_content = get_json_file(shadow_filepath)
img_id_list = [x['image_id'] for x in boxinst_content]
img_id_list = list(dict.fromkeys(img_id_list))
boxinst_filter_all = []
obj_iou_th = 0.7
for img_id_one in tqdm(img_id_list):
    # boxinst choose car class,shadow choose shadow class.
    boxinst_img_one = list(filter(lambda x: x['image_id'] == img_id_one,
                                  boxinst_content))
    obj_in_shadow_img_one = list(filter(lambda x: x['image_id'] == img_id_one and x['category_id'] == 0,
                                 shadow_content))
    shadow_in_shadow_img_one = list(filter(lambda x: x['image_id'] == img_id_one and x['category_id'] == 1,
                                           shadow_content))
    if len(obj_in_shadow_img_one) != len(shadow_in_shadow_img_one):
        print('kk')
    if len(obj_in_shadow_img_one) > 0:
        for ctr, boxinst_inst_one in enumerate(boxinst_img_one):
            # only for car class.
            if boxinst_inst_one['category_id'] == 1:
                boxinst_mask_inst_one = boxinst_inst_one['segmentation']
                inst_iou_list= [cocomask.iou([boxinst_mask_inst_one],[x['segmentation']],pyiscrowd=[0]) for x in obj_in_shadow_img_one]
                if max(inst_iou_list)>obj_iou_th:
                    obj_id = inst_iou_list.index(max(inst_iou_list))
                    pred_assc = obj_in_shadow_img_one[obj_id]['pred_associations']
                    correspond_shadow_in_shadow_img_one = list(filter(lambda x:x['pred_associations']==pred_assc, shadow_in_shadow_img_one))[0]
                    shadow_in_shadow_mask_inst_one = correspond_shadow_in_shadow_img_one['segmentation']
                    intersect_area = cocomask.area(cocomask.merge(
                        [boxinst_mask_inst_one, shadow_in_shadow_mask_inst_one], intersect=True))
                    if intersect_area > 0.0:
                        boxinst_mask_inst_one_decode = cocomask.decode(boxinst_mask_inst_one)
                        # shadow_mask_img_one_decode type is np.uint8.
                        shadow_mask_img_one_decode = cocomask.decode(shadow_in_shadow_mask_inst_one)
                        boxinst_mask_inst_one_decode[np.where(shadow_mask_img_one_decode > 0)] = 0
                        boxinst_mask_inst_one_encode = cocomask.encode(boxinst_mask_inst_one_decode)
                        # decode('UTF-8')-convert type bytes to string.
                        boxinst_mask_inst_one_encode['counts'] = boxinst_mask_inst_one_encode['counts'].decode(
                            'UTF-8')
                        boxinst_inst_one['segmentation'] = boxinst_mask_inst_one_encode
                        boxinst_img_one[ctr] = boxinst_inst_one
    boxinst_filter_all.extend(boxinst_img_one)
json_outpath = '../../my_shadow_det/filter_result_use_object/shadow_filter_result.json'
write_json_file(boxinst_filter_all, json_outpath)
