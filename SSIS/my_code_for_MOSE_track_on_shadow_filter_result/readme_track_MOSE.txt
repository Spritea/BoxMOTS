1.Files in the folder my_code_for_MOSE_track_on_shadow_filter_result is to
get mask trajectories with shadow detection results for MOSE.
2.In this folder, for mask trajectories, we only need to perform step 1,4,5 of StrongSORT folder on server tesla,
since the shadow filter result does not influence box detections
and its DeepSORT data association results.
3.We copy the step 1,4,5 code to this place, to keep the StrongSORT folder on server tesla clean.