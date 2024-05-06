import numpy as np
import json
import pycocotools.mask as cocomask
from tqdm import tqdm
import argparse

#rate 0.9:46.0
#rate 0.3:46.2
def get_json_file(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content
def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)

def main(shadow_rate_arg):
    # class: 1-object,2-shadow.
    # category_id: 0-object,1-shadow.
    boxinst_filepath = "../../my_shadow_det/original_boxinst_result/original_result_on_server_4/iter_0003999/coco_instances_results.json"
    shadow_filepath = "../../my_shadow_det/my_shadow_result/my_json_result/kitti_mots_val/shadow_instance_result.json"
    boxinst_content = get_json_file(boxinst_filepath)
    shadow_content = get_json_file(shadow_filepath)
    img_id_list = [x['image_id'] for x in boxinst_content]
    img_id_list = list(dict.fromkeys(img_id_list))
    boxinst_filter_all = []
    for img_id_one in img_id_list:
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
                    if intersect_area>0.0 and shadow_rate<shadow_rate_arg:
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
    json_outname = 'shadow_filter_result_rate_0'+str(int(shadow_rate_arg*10))+'.json'
    json_outpath = '../../my_shadow_det/filter_result/original_result_on_server_4/'+json_outname
    write_json_file(boxinst_filter_all,json_outpath)

if __name__ == "__main__":
    # np.linespace cannot get accurate decimal number.
    # shadow_rate = np.linspace(0.1,1,num=10,endpoint=True)
    shadow_rate_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for rate_one in tqdm(shadow_rate_list):
        main(rate_one)
    