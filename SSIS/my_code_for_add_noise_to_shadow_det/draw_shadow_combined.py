# @File: SSIS/my_code_for_add_noise_to_shadow_det/draw_shadow_combined.py
# @Author: cws 
# @Create Date: 2024/02/28
# @Desc: Draw seg result after shadow filter.

import json
import numpy as np
from PIL import Image
import pycocotools.mask as cocomask
import os

def load_json(filepath):
    with open(filepath,'r') as f:
        json_content = json.load(f)
    return json_content

def apply_mask(image, mask, color, alpha=0.5, category=1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == category,
                            image[:, :, c] * (1 - alpha) + alpha * color[c],
                            image[:, :, c])
    return image

def main():
    # combined sample-kitti val-seq 0006-img 000046.png.
    # shadow_use = shadow_img_one[0]
    seq_name = "0006"
    img_name = "000046"
    img_path = "../my_dataset/KITTI_MOTS/val_in_trainval/"+seq_name+"/img1/"+img_name+".png"
    # shadow_result = "my_shadow_det/combine_noisy_shadow_with_boxmots_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_ckpt_500_steps_4k/inference/iter_0002839/shadow_filter_result_rate_02_erode_kernel_3_iter_1.json"
    shadow_result = "my_shadow_det/combine_noisy_shadow_with_boxmots_result/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_ckpt_500_steps_4k/inference/iter_0002839/shadow_filter_result_rate_02_erode_kernel_7_iter_1.json"
    shadow_out_dir = os.path.join("vis_shadow_result/draw_shadow_combined/", seq_name)
    os.makedirs(shadow_out_dir, exist_ok=True)
    shadow_out_path = os.path.join(shadow_out_dir, img_name+"_shadow_erode_kernel_7_iter_1_combined.png")
    shadow_content = load_json(shadow_result)
    img_id = int(seq_name)*10000+int(img_name)
    # class: 1-object,2-shadow.
    # category_id: 0-object,1-shadow.
    shadow_img_one = list(filter(lambda x: x['image_id']==img_id, shadow_content))
    # there are 2 pairs of object/shadow in the img 0006-000094.png,
    # and we use the bottom-right one for vis.
    shadow_use = shadow_img_one[0]
    img = np.array(Image.open(img_path), dtype="float32") / 255
    label = cocomask.decode(shadow_use['segmentation'])
    # orange color RGB [255,128,0]
    color = [135, 206, 235]
    color = np.array(color)/255.0
    img_w_shadow = apply_mask(img, label, color, alpha=0.5)
    img_w_shadow = (img_w_shadow*255).astype(np.uint8)
    img_w_shadow = Image.fromarray(img_w_shadow)
    img_w_shadow.save(shadow_out_path)
    print('kk')

if __name__=="__main__":
    main()
