import torch

# file_name="../training_dir/COCO_pretrain/CondInst_MS_R_50_1x_kitti_mots/inference/instances_predictions.pth"
file_name="../training_dir/test_only/reid_eval/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pth"
a=torch.load(file_name)
print('kk')

