#!/bin/sh

cd demo
CUDA_VISIBLE_DEVICES=0 \
python demo_my_data_out_assc.py --input ../../my_dataset/KITTI_MOTS/imgs/train_in_trainval/
