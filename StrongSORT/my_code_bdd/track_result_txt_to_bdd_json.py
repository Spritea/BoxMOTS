import json,os
from pathlib import Path
from tqdm import tqdm
from bdd_category_dict import bdd_val_video_id_dict,bdd_val_img_id

# below result format needed for bdd.
# - videoName: str, name of current sequence
# - name: str, name of current frame
# - frameIndex: int, index of current frame within sequence
# - labels []:
#     - id: str, unique instance id of prediction in current sequence
#     - category: str, name of the predicted category
#     - rle:
#         - counts: str
#         - size: (height, width)

# note the first column(frame idx) in the deep sort result txt file is
# directly int of the number part in img_filename which starts from 1,
# but the frame_id in the json file is the number part-1 in img_filename,
# which starts from 0, so need to change in this file.
def load_txt(txt_path):
    with open(txt_path,'r') as f:
        # read to a list and remove ending line character.
        txt_content=f.read().splitlines()
    return txt_content
def write_json_file(json_list,filepath):
    with open(filepath,'w') as f:
        json.dump(json_list,f,indent=2)
def get_RLE(line_one):
    split_all=line_one.split(' ')
    RLE={'size':[int(split_all[3]),int(split_all[4])],'counts':split_all[5]}
    return RLE
def deal_one_frame(txt_content,frame_id_zero_base,video_id):
    # note that frame_idx in txt_content is 1 base.
    frame_one_content=list(filter(lambda x: int(x.split(' ')[0])==(frame_id_zero_base+1),txt_content))
    videoName = bdd_val_video_id_dict_reverse[video_id]
    img_name = videoName+'-'+str(frame_id_zero_base).zfill(7)+'.png'
    frameIndex = frame_id_zero_base
    labels = []
    for line_one in frame_one_content:
        instance_id = line_one.split(' ')[1]
        category = cat_reverse[int(line_one.split(' ')[2])]
        RLE = get_RLE(line_one)
        inst_one = {'id': instance_id, 'category':category, 'rle': RLE}
        labels.append(inst_one)
    frame_one_dict = {'videoName':videoName,'name':img_name,'frameIndex':frameIndex,'labels':labels}
    return frame_one_dict
    
val_set_seqmap=list(range(1,33))
cat={'pedestrian':1,'rider':2,'car':3,'truck':4,'bus':5,'motorcycle':7,'bicycle':8}
txt_folder = '../my_data_bdd/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots/to_mots_txt/iter_0014999/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04_bus_conf_06/shadow_filter_result_rate_02_car_truck/all_class_no_overlap/'
json_out_path = '../my_data_bdd/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots/to_mots_txt/iter_0014999/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04_bus_conf_06/shadow_filter_result_rate_02_car_truck/all_class_no_overlap_json_file/seg_track_pred.json'
bdd_val_video_id_dict_reverse = {}
for k,v in bdd_val_video_id_dict.items():
    bdd_val_video_id_dict_reverse[v] = k
cat_reverse = {}
for k,v in cat.items():
    cat_reverse[v] = k
video_frame_id = {}
for video_one in val_set_seqmap:
    video_one_imgs = list(filter(lambda x:x['video_id']==video_one,bdd_val_img_id))
    video_one_frame_ids = [x['frame_id'] for x in video_one_imgs]
    video_frame_id.update({video_one:video_one_frame_ids})
result_all_video = []
for video_id in tqdm(val_set_seqmap):
    txt_path = txt_folder+str(video_id).zfill(4)+'.txt'
    txt_content = load_txt(txt_path)
    # frame_unique_list=list(set([int(x.split(' ')[0]) for x in txt_content]))
    frame_list_full = video_frame_id[video_id]
    for frame_id_zero_base in frame_list_full:
        frame_one_dict = deal_one_frame(txt_content,frame_id_zero_base,video_id)
        result_all_video.append(frame_one_dict)
os.makedirs(Path(json_out_path).parent,exist_ok=True)
write_json_file(result_all_video,json_out_path)
print('kk')