#!/bin/bash
set -e
#set -e is to stop after the first error.

# below for bdd.
CUDA_VISIBLE_DEVICES=0 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 1 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=2 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 2 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=3 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 3 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=4 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 4 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=5 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 5 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=6 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 6 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=0 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 7 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=2 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 8 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=3 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 9 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=4 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 10 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=5 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 11 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train &
CUDA_VISIBLE_DEVICES=6 \
python my_eval_and_save_bidirect_bdd_parallel.py --part 12 \
--model checkpoints/gma-sintel.pth --path my_data/bdd100k/images/seg_track_20/train

