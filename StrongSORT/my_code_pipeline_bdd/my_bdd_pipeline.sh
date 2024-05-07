#!/bin/bash
set -e
#set -e is to stop after the first error.

# no space around assignment in shell, like a = 123,
# this is wrong, it would be command 'a' with 2 args '=' and '123'.
# use ; to seperate commands that run in order.
# when use shadow, only need to change step 1,4,5,6.

# need to change opt.dir_dataset in opts.py for different datasets.
BOX_INST_DIR=my_data_bdd/reid_one_class_infer_bdd_pair_warp_right_track_reid_eval_in_train/COCO_pretrain_strong/iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim/BoxInst_MS_R_50_1x_kitti_mots
ITER_NUM=iter_0014999
# 1.below for bdd_coco_to_mot_det_w_seg_per_class_arg.py
# below for shadow.
COCO_RESULT_PATH=../$BOX_INST_DIR/inference/$ITER_NUM/shadow_filter_result/shadow_filter_result_rate_02_car_truck.json
OUT_DIR_S1=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_det_seg/val_in_trainval/shadow_filter_result_rate_02_car_truck/
# COCO_RESULT_PATH=../$BOX_INST_DIR/inference/$ITER_NUM/coco_instances_results.json
# OUT_DIR_S1=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_det_seg/val_in_trainval/

# 2.below for bdd_reid_feat_to_strong_sort_arg.py
FILE_PATH=../$BOX_INST_DIR/reid_infer_and_eval_out/$ITER_NUM/reid_infer_out.pkl
OUT_DIR_S2=../$BOX_INST_DIR/reid_np/$ITER_NUM/

# 3.below for strong_sort.py
DIR_DETS_BICYCLE=$BOX_INST_DIR/reid_np/$ITER_NUM/bicycle
DIR_SAVE_BICYCLE=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/bicycle
DIR_DETS_BUS=$BOX_INST_DIR/reid_np/$ITER_NUM/bus
DIR_SAVE_BUS=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/bus
DIR_DETS_CAR=$BOX_INST_DIR/reid_np/$ITER_NUM/car
DIR_SAVE_CAR=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/car
DIR_DETS_MOTORCYCLE=$BOX_INST_DIR/reid_np/$ITER_NUM/motorcycle
DIR_SAVE_MOTORCYCLE=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/motorcycle
DIR_DETS_PEDESTRIAN=$BOX_INST_DIR/reid_np/$ITER_NUM/pedestrian
DIR_SAVE_PEDESTRIAN=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/pedestrian
DIR_DETS_RIDER=$BOX_INST_DIR/reid_np/$ITER_NUM/rider
DIR_SAVE_RIDER=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/rider
DIR_DETS_TRUCK=$BOX_INST_DIR/reid_np/$ITER_NUM/truck
DIR_SAVE_TRUCK=$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/truck

# 4.below for bdd_match_mask_to_box_arg.py
TRACK_TXT_DIR=../$BOX_INST_DIR/results_mot/$ITER_NUM/DeepSORT/min_det_conf_04/
MASK_TXT_DIR=$OUT_DIR_S1
# below for shadow.
OUT_DIR_S4=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/shadow_filter_result_rate_02_car_truck/
# OUT_DIR_S4=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/

# 5.below for bdd_combine_txt_no_overlap_arg.py
FILE_DIR=$OUT_DIR_S4

# 6.below for track_result_txt_to_bdd_json_arg.py
# below for shadow.
TXT_FOLDER=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/shadow_filter_result_rate_02_car_truck/all_class_no_overlap/
OUT_DIR_S5=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/shadow_filter_result_rate_02_car_truck/all_class_no_overlap_json_file/seg_track_pred.json
# TXT_FOLDER=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/all_class_no_overlap/
# OUT_DIR_S5=../$BOX_INST_DIR/to_mots_txt/$ITER_NUM/mots_seg_track/DeepSORT/val_in_trainval/mask_min_det_conf_04/all_class_no_overlap_json_file/seg_track_pred.json

echo step 1;
python bdd_coco_to_mot_det_w_seg_per_class_arg.py \
--coco_result_path $COCO_RESULT_PATH \
--outdir $OUT_DIR_S1;

# echo ;
# echo step 2;
# python bdd_reid_feat_to_strong_sort_arg.py \
# --file_path $FILE_PATH \
# --out_dir $OUT_DIR_S2;

# cd ..;
# echo ;
# echo step 3 bicycle;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_BICYCLE \
# --dir_save $DIR_SAVE_BICYCLE;
# echo step 3 bus;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_BUS \
# --dir_save $DIR_SAVE_BUS;
# echo step 3 car;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_CAR \
# --dir_save $DIR_SAVE_CAR;
# echo step 3 motorcycle;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_MOTORCYCLE \
# --dir_save $DIR_SAVE_MOTORCYCLE;
# echo step 3 pedestrian;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_PEDESTRIAN \
# --dir_save $DIR_SAVE_PEDESTRIAN;
# echo step 3 rider;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_RIDER \
# --dir_save $DIR_SAVE_RIDER;
# echo step 3 truck;
# python strong_sort.py \
# BDD_MOTS val_set \
# --dir_dets $DIR_DETS_TRUCK \
# --dir_save $DIR_SAVE_TRUCK;

# cd my_code_pipeline_bdd;
echo ;
echo step 4;
python bdd_match_mask_to_box_arg.py \
--track_txt_dir $TRACK_TXT_DIR \
--mask_txt_dir $MASK_TXT_DIR \
--out_dir $OUT_DIR_S4;

echo ;
echo step 5;
python bdd_combine_txt_no_overlap_arg.py \
--file_dir $FILE_DIR

echo ;
echo step 6;
python track_result_txt_to_bdd_json_arg.py \
--txt_folder $TXT_FOLDER \
--json_out_path $OUT_DIR_S5
