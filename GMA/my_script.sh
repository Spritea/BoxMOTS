#!/bin/bash
# below \ is needed after the env variable,
# to make the shell command one line,
# otherwise the env variable would only be taken as normal shell variable.
# CUDA_VISIBLE_DEVICES=3 \
# python my_eval_and_save_bidirect.py --model checkpoints/gma-kitti.pth --path my_data/KITTI_MOTS/imgs/train_in_trainval

# below for bdd.
CUDA_VISIBLE_DEVICES=7 \
python my_eval_and_save_bidirect_bdd.py --model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train
