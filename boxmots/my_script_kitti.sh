#!/bin/sh

# below is shell variable.
# use $ to call shell variable.

# below for shadow.
# CONFIG_FILE=configs/BoxInst_ReID_One_Class_Infer_Car_Shadow/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_no_color_sim_pretrain_weights.yaml
# OUT_DIR=training_dir/reid_one_class_infer_car_shadow/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_no_color_sim_pretrain_weights/BoxInst_MS_R_50_1x_kitti_mots

# # below is command
# CUDA_VISIBLE_DEVICES=3 \
# OMP_NUM_THREADS=1 \
# python tools/train_net_my_data_reid_one_class_infer_car_shadow.py \
# --config-file $CONFIG_FILE \
# --num-gpus 1 \
# --dist-url tcp://127.0.0.1:50161 \
# OUTPUT_DIR $OUT_DIR

CONFIG_FILE=configs/BoxInst_ReID_One_Class_Infer_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_kitti_mots_imagenet_pretrain_long_epoch_seq_shuffle_fl_2_lr_0_01_bs_4_eval_500_no_color_sim_pretrain_weights_v2.yaml
OUT_DIR=training_dir/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train/imagenet_pretrain/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_01_bs_4_eval_500_no_color_sim_pretrain_weights_v2_time_2/BoxInst_MS_R_50_1x_kitti_mots

# below is command
CUDA_VISIBLE_DEVICES=6 \
OMP_NUM_THREADS=1 \
python tools/train_net_my_data_reid_one_class_infer_opt_flow_correct_aug_right_track_reid_eval_in_train.py \
--config-file $CONFIG_FILE \
--num-gpus 1 \
--dist-url tcp://127.0.0.1:50172 \
OUTPUT_DIR $OUT_DIR
