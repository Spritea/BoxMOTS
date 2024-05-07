import numpy as np
import pickle

file_path="../my_data/reid_one_class_mask_pool_infer_for_train/COCO_pretrain_strong/search_for_best_iter/long_epoch_one_gpu_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_out/iter_0006999/reid_infer_out.pkl"
with open(file_path,'rb') as f:
    pkl_content=pickle.load(f)
print('kk')
for inst_one in pkl_content:
    reid_feat=inst_one['reid_feat']
    if np.isnan(reid_feat).any():
        print('Nan found in reid feat!')
print('kk')
