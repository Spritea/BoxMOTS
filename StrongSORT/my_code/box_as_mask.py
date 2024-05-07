import os
from tqdm import tqdm
import numpy as np
from pycocotools import mask as cocomask

# this file box_as_mask.py is to
# use the box as mask directly for fcos's tracking result,
# and boxinst also uses this for comparison with fcos.

def convert_seq_one(track_txt_file,cat_choose,img_size,out_path):
    np_track=np.loadtxt(track_txt_file,dtype=np.float64,delimiter=',')
    if len(np_track)==0:
        # some seqs don't have track results for ped.
        with open(out_path,'w') as f:
            f.writelines([])
        return
    # since negative values could appear in x,y in np_track
    np_track[np_track[:,2]<0,2]=0
    np_track[np_track[:,3]<0,3]=0
    
    np_track_box=np_track[:,2:6]
    # can't get all boxes one time like below, cause the output would be wrong for bounding boxes except the first one.
    # RLE_all=cocomask.frPyObjects(np_track_box,img_size["h"],img_size["w"])
    # for frPyObjects(a,b,c), a must be 2d np array, like np.array([[1,2,3,4]]).
    # frPyObjects() returns a list, so use [0] to get the item.
    RLE_all=[cocomask.frPyObjects(np.expand_dims(x,axis=0),img_size["h"],img_size["w"])[0] for x in np_track_box]

    # def to_str(z):
    #     z['counts']=z['counts'].decode('UTF-8')
    #     return z
    # RLE_all_change=list(map(to_str,RLE_all))
    for item in RLE_all:
        # convert bytes to string.
        item['counts']=item['counts'].decode('UTF-8')

    out_seq=[]
    for ctr,line_one in enumerate(np_track):
        frame_id=line_one[0]
        track_id=line_one[1]
        h_in_rle=RLE_all[ctr]['size'][0]
        w_in_rle=RLE_all[ctr]['size'][1]
        counts_in_rle=RLE_all[ctr]['counts']
        out_one=[int(frame_id),int(track_id),cat[cat_choose],h_in_rle,w_in_rle,counts_in_rle]
        out_one=' '.join(list(map(str,out_one)))+'\n'
        out_seq.append(out_one)
    
    with open(out_path,'w') as f:
        f.writelines(out_seq)
        

if __name__=="__main__":
    cat={'car':1,'pedestrian':2}
    # cat_choose='pedestrian'
    val_in_trainval_seqmap=[2,6,7,8,10,13,14,16,18]
    val_img_size={"0002":{"w":1242,"h":375},"0006":{"w":1242,"h":375},"0007":{"w":1242,"h":375},
              "0008":{"w":1242,"h":375},"0010":{"w":1242,"h":375},"0013":{"w":1242,"h":375},
              "0014":{"w":1224,"h":370},"0016":{"w":1224,"h":370},"0018":{"w":1238,"h":374}}
    track_txt_dir="../my_data/reid_one_class_mask_pool_infer_for_train/COCO_pretrain_strong/search_for_best_iter/long_epoch_one_gpu_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k/BoxInst_MS_R_50_1x_kitti_mots/results_mot/DeepSORT/min_det_conf_06/"
    out_dir="../my_data/reid_one_class_mask_pool_infer_for_train/COCO_pretrain_strong/search_for_best_iter/long_epoch_one_gpu_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k/BoxInst_MS_R_50_1x_kitti_mots/to_mots_txt/mots_seg_track_box_based_mask/DeepSORT/val_in_trainval/min_det_conf_06/"
    for cat_choose in cat.keys():
        os.makedirs(out_dir+cat_choose,exist_ok=True)
        for seq_id in tqdm(val_in_trainval_seqmap):
            track_txt_path=track_txt_dir+cat_choose+'/'+str(seq_id).zfill(4)+'.txt'
            out_path=out_dir+cat_choose+'/'+str(seq_id).zfill(4)+'.txt'
            img_size=val_img_size[str(seq_id).zfill(4)]
            convert_seq_one(track_txt_path,cat_choose,img_size,out_path)
