1.Note that the linear_assignment function in utils.py should not use package lap,
because it cannot handle empty cost_matrix.
For kitti and bdd100k dataset, in fact lap is not used,
because the strongsort conda env does not install package lap.
2.When run strongsort on MOSE, we need to change following params,
because there is very long-term occlusion in MOSE,
and we need to lower the reid feat match threshold,
and remove the kalman gate which limits that the motion between the old track and the det cannot be large.
The max_age is also increased not intensionally, but this may not be necessary for seq 0003,0005 (not tested yet),
Specifically, we need to change below settings:
2.1 set opt.max_cosine_distance = 0.6 in opts.py.
2.2 comment the sentence below in function gated_metric() in deep_sort/tracker.py.
#cost_matrix = linear_assignment.gate_cost_matrix(
#    cost_matrix, tracks, dets, track_indices,
#    detection_indices)
2.3 set max_age=70 in init() of class Tracker in deep_sort/tracker.py.
2.3 these params are tuned based on the 2 seqs: 
0fc00006,1ca2531a (after rename: 0003,0005) in the train set.
2.4 when run DeepSORT for MOSE, replace the deep_sort/tracker.py content
with the deep_sort/tracker_for_MOSE.py content.
3.the final version for test on MOSE use default max_age=30, opt.max_cosine_distance=0.6, and no kalman gate.
