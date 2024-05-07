1. since deep_sort gets frame id directly from int the number part
of the image file, the frame id should start with 1 for 
bdd_coco_to_mot_det_w_seg_per_class.py, bdd_combine_txt_no_overlap.py,
bdd_match_mask_to_box.py, bdd_reid_feat_to_strong_sort.py.
2. for track_result_txt_to_bdd_json.py, since the frame_id in 
BDD label file goes from 0, so we need to convert the frame start id
from 1 to 0, then we can do evaluation.
