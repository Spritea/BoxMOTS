import pickle

file_name="../training_dir/test_only/reid_infer/CondInst_MS_R_50_1x_kitti_mots/reid_infer_out_backup/reid_infer_out.pkl"
with open(file_name,'rb') as f:
    a=pickle.load(f)
    print('kk')

file_name="../training_dir/test_only/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0000039/reid_infer_out.pkl"
with open(file_name,'rb') as f:
    b=pickle.load(f)
    print('kk')