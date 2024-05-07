1.the implementation for ytvis 2019 is mainly based on
the implementation for bdd dataset and MOSE dataset.
2.for directly test the kitti pretrain model on ytvis val set,
the deepsort setting is the same with MOSE:
results_mots/DeepSORT/min_det_conf_04_max_cos_0.6_no_kalman_gate.
3.for train our method on ytvis train set and then test it on ytvis val set,
we first try the deepsort setting that is the same with KITTI MOTS:
results_mots/DeepSORT/min_det_conf_06,
then we try the MOSE setting:
results_mots/DeepSORT/min_det_conf_04_max_cos_0.6_no_kalman_gate.
We decided to use the MOSE setting, because it has better AP on ytvis val set.
