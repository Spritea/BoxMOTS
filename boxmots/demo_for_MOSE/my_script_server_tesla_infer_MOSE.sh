#!/bin/sh

# run this script under the AdelaiDet main folder
# below for MOSE inference
# CONFIG_FILE=configs/BoxInst_ReID_One_Class_Infer_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_steps_4k_no_color_sim_pretrain_weights_v3.yaml
CONFIG_FILE=configs/BoxInst_ReID_One_Class_Infer_Right_Track/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k.yaml
MODEL_CKPT=training_dir/reid_one_class_infer_right_track/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k/BoxInst_MS_R_50_1x_kitti_mots/model_0002499.pth

python demo_for_MOSE/my_demo_simplified.py \
    --config-file $CONFIG_FILE \
    --input demo_for_MOSE/samples/0a7a3629/00002.jpg \
    --output demo_for_MOSE/test \
    --opts MODEL.WEIGHTS $MODEL_CKPT
    