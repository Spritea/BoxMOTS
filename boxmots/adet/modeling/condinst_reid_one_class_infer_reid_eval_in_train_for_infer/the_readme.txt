1.The model of condinst_reid_one_class_infer_reid_eval_in_train_for_infer folder 
is totally same with condinst_reid_one_class_infer_reid_eval_in_train folder,
and the only difference is in the inference, reid eval is not used now.
Because we may test the model on any video, like the YouTube VIS 2019 Validation dataset, which does not have any label,
and hence we cannot do reid eval in the inference, and we should just do reid infer.
to get the object's reid feat in the inference process.
2.The model of condinst_reid_one_class_infer_reid_eval_in_train_for_infer folder
can directly use the trained model ckpt of condinst_reid_one_class_infer_reid_eval_in_train folder,
since they have the same model arch.
