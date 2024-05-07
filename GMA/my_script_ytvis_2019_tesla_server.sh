#!/bin/bash
# below \ is needed after the env variable,
# to make the shell command one line,
# otherwise the env variable would only be taken as normal shell variable.

CUDA_VISIBLE_DEVICES=1 \
python my_eval_and_save_bidirect_ytvis_2019.py --model checkpoints/gma-sintel.pth --path my_data/youtube_vis_2019/train/JPEGImages

