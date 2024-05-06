#!/bin/sh

# below is shell variable.
# use $ to call shell variable.

# CONFIG_FILE=configs/CondInst/MS_R_50_1x_bdd_mots_coco_pretrain_strong_lr_0_01_no_class_train.yaml
# OUT_DIR=training_dir/bdd_mots/COCO_pretrain_strong/lr_0_01_no_class_train/BoxInst_MS_R_50_1x_kitti_mots

# below is command
# CUDA_VISIBLE_DEVICES=5 \
# OMP_NUM_THREADS=1 \
# python tools/train_net_my_data_bdd_no_class_train.py \
# --config-file $CONFIG_FILE \
# --num-gpus 1 \
# --dist-url tcp://127.0.0.1:50167 \
# OUTPUT_DIR $OUT_DIR

# below use reid.
# CONFIG_FILE=configs/BDD_DATA/BoxInst_ReID_One_Class_Infer_BDD_Pair_Warp/MS_R_50_1x_bdd_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_2_eval_500_no_color_sim.yaml
# OUT_DIR=training_dir_hdda/bdd_mots/reid_one_class_infer_bdd_pair_warp_right_track/COCO_pretrain_strong/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_2_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots

# CUDA_VISIBLE_DEVICES=6 \
# OMP_NUM_THREADS=1 \
# python tools/bdd_data/train_net_my_data_bdd_reid_one_class_infer_opt_flow_correct_aug_right_track.py \
# --config-file $CONFIG_FILE \
# --num-gpus 1 \
# --dist-url tcp://127.0.0.1:50170 \
# OUTPUT_DIR $OUT_DIR

# below use reid eval in train with class 7.
# CONFIG_FILE=configs/BDD_DATA/BoxInst_ReID_One_Class_Infer_BDD_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_bdd_mots_coco_pretrain_strong_iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_40_no_color_sim_no_class_train.yaml
# OUT_DIR=training_dir/bdd_mots/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_40_no_color_sim_no_class_train/BoxInst_MS_R_50_1x_kitti_mots

# CUDA_VISIBLE_DEVICES=0 \
# OMP_NUM_THREADS=1 \
# python tools/bdd_data/train_net_my_data_bdd_reid_one_class_infer_opt_flow_correct_aug_right_track_reid_eval_in_train_no_class_train.py \
# --config-file $CONFIG_FILE \
# --num-gpus 1 \
# --dist-url tcp://127.0.0.1:50171 \
# OUTPUT_DIR $OUT_DIR

# below use reid eval in train with pretrain weights.
CONFIG_FILE=configs/BDD_DATA/BoxInst_ReID_One_Class_Infer_BDD_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_bdd_mots_coco_pretrain_strong_iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim_pretrain_weights.yaml
OUT_DIR=training_dir/bdd_mots/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim_pretrain_weights_time_4/BoxInst_MS_R_50_1x_kitti_mots

CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 \
python tools/bdd_data/train_net_my_data_bdd_reid_one_class_infer_opt_flow_correct_aug_right_track_reid_eval_in_train.py \
--config-file $CONFIG_FILE \
--num-gpus 1 \
--dist-url tcp://127.0.0.1:50173 \
OUTPUT_DIR $OUT_DIR
