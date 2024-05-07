For detection/instance segmentation and ReID results
generated in the training time, take following steps to get mots_seg_track result.
1. use coco_to_mot_det_w_seg_per_class.py to convert coco format detection/instance segmentation
results to txt file, output folder: to_mots_txt/mots_det_seg/
2. use reid_feat_to_strong_sort.py to convert ReID results to box+features needed by
StrongSORT, output folder: reid_np/
3. use strong_sort.py to do DeepSORT, its input file is reid_np/,
output folder: results_mot/DeepSORT/min_det_conf_06
4. use match_mask_to_box.py to match mask to bboxes in the track,
output folder: to_mots_txt/mots_seg_track/DeepSORT/val_in_trainval/min_det_conf_06/car (or pedestrian)
5. use combine_txt_no_overlap.py to combine car and pedestrian results together, and remove overlapping pixels,
output folder: to_mots_txt/mots_seg_track/DeepSORT/val_in_trainval/min_det_conf_06/both_car_pedestrian_no_overlap
