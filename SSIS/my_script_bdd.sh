#!/bin/bash

cd demo
CUDA_VISIBLE_DEVICES=7 \
python demo_my_data_bdd.py --input ../../my_dataset/bdd100k/images/seg_track_20/val/
