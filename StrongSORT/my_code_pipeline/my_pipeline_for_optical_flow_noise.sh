#!/bin/bash
set -e
#set -e is to stop after the first error.

# no space around assignment in shell, like a = 123,
# this is wrong, it would be command 'a' with 2 args '=' and '123'.
# use ; to seperate commands that run in order.

# need to change opt.dir_dataset in opts.py for different datasets.
BOX_INST_DIR=my_data_for_kitti_add_optical_flow_noise/training_dir_flow_noise_hddc/flow_noise_kitti_mots/reid_one_class_infer_pair_warp_right_track_reid_eval_in_train_with_noise_optical_flow/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_steps_4k_no_color_sim_pretrain_weights_v3_flow_noise_std_30/BoxInst_MS_R_50_1x_kitti_mots
ITER_NUM=iter_0002839
# 1.below for coco_to_mot_det_w_seg_per_class_arg.py
COCO_RESULT_PATH=../$BOX_INST_DIR/inference/$ITER_NUM/coco_instances_results.json
OUTDIR_S1=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_det_seg/val_in_trainval/


# 2.below for reid_feat_to_strong_sort_arg.py
FILE_PATH=../$BOX_INST_DIR/reid_infer_and_eval_out/$ITER_NUM/reid_infer_out.pkl
OUT_DIR_S2=../$BOX_INST_DIR/reid_np/$ITER_NUM/

# 3.below for strong_sort.py
DIR_DETS_PED=$BOX_INST_DIR/reid_np/$ITER_NUM/pedestrian
DIR_SAVE_PED=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/pedestrian
DIR_DETS_CAR=$BOX_INST_DIR/reid_np/$ITER_NUM/car
DIR_SAVE_CAR=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/car

# 4.below for match_mask_to_box_arg.py
TRACK_TXT_DIR=../$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/
MASK_TXT_DIR=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_det_seg/val_in_trainval/
OUT_DIR_S4=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/

# 5.below for combine_txt_no_overlap_arg.py
FILE_DIR=$OUT_DIR_S4


echo step 1;
python coco_to_mot_det_w_seg_per_class_arg.py \
--coco_result_path $COCO_RESULT_PATH \
--outdir $OUTDIR_S1;

echo step 2;
python reid_feat_to_strong_sort_arg.py \
--file_path $FILE_PATH \
--out_dir $OUT_DIR_S2;
cd ..;

echo step 3 car;
python strong_sort.py \
KITTI_MOTS val_in_trainval \
--dir_dets $DIR_DETS_CAR \
--dir_save $DIR_SAVE_CAR;
echo step 3 ped;
python strong_sort.py \
KITTI_MOTS val_in_trainval \
--dir_dets $DIR_DETS_PED \
--dir_save $DIR_SAVE_PED;
cd my_code_pipeline;

echo step 4;
python match_mask_to_box_arg.py \
--track_txt_dir $TRACK_TXT_DIR \
--mask_txt_dir $MASK_TXT_DIR \
--out_dir $OUT_DIR_S4;

echo step 5;
python combine_txt_no_overlap_arg.py \
--file_dir $FILE_DIR
