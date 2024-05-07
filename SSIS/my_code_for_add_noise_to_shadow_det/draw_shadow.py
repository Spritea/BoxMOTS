# @File: SSIS/my_code_for_add_noise_to_shadow_det/draw_shadow.py 
# @Author: cws 
# @Create Date: 2024/02/28
# @Desc: Draw clean shadow and eroded shadow.

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
    # sample-kitti val-seq 0006-img 000094.png.
    # shadow_use = shadow_img_one[1]
    # sample-kitti val-seq 0006-img 000109.png.
    # shadow_use = shadow_img_one[5]
    # sample-kitti val-seq 0006-img 000046.png.
    # shadow_use = shadow_img_one[3]
    seq_name = "0006"
    img_name = "000046"
    img_path = "../my_dataset/KITTI_MOTS/val_in_trainval/"+seq_name+"/img1/"+img_name+".png"
    # shadow_result = "my_shadow_det/add_noise_to_shadow_det_kitti_mots_val/shadow_instance_result_erode_kernel_9_iter_1.json"
    shadow_result = "my_shadow_det/my_shadow_result/my_json_result/kitti_mots_val/shadow_instance_result.json"
    shadow_out_dir = os.path.join("vis_shadow_result/draw_shadow/", seq_name)
    os.makedirs(shadow_out_dir, exist_ok=True)
    shadow_out_path = os.path.join(shadow_out_dir, img_name+"_clean_shadow.png")
    shadow_content = load_json(shadow_result)
    img_id = int(seq_name)*10000+int(img_name)
    # class: 1-object,2-shadow.
    # category_id: 0-object,1-shadow.
    shadow_img_one = list(filter(lambda x: x['image_id']==img_id, shadow_content))
    # there are 2 pairs of object/shadow in the img 0006-000094.png,
    # and we use the bottom-right one for vis.
    # shadow_use = shadow_img_one[1]
    shadow_use = shadow_img_one[3]
    img = np.array(Image.open(img_path), dtype="float32") / 255
    label = cocomask.decode(shadow_use['segmentation'])
    # orange color RGB [255,128,0]
    color = np.array([255,128,0])/255.0
    img_w_shadow = apply_mask(img, label, color, alpha=1.0)
    img_w_shadow = (img_w_shadow*255).astype(np.uint8)
    img_w_shadow = Image.fromarray(img_w_shadow)
    img_w_shadow.save(shadow_out_path)
    print('kk')

if __name__=="__main__":
    main()
