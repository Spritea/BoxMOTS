import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from network import RAFTGMA
from utils import flow_viz
from utils.utils import InputPadder
import os
from tqdm import tqdm
from pathlib import Path

DEVICE = 'cuda'

# this gets opt flow from img1 to img2,
# and opt flow from img2 to img1.

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def my_viz(img, flo, flow_dir,img_name):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, img_name), flo)
    # print(f"Saving optical flow visualisation at {os.path.join(flow_dir, img_name)}")


def normalize(x):
    return x / (x.max() - x.min())


def demo(args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    # flow_dir = os.path.join(args.path, args.model_name)
    # if not os.path.exists(flow_dir):
    #     os.makedirs(flow_dir)

    # below added by cws.
    # deal with many seqs.
    with torch.no_grad():
        seq_dir=[]
        out_dir="my_results_hdda/BDD_MOTS/parallel/train"
        for item in os.scandir(args.path):
            seq_dir.append(item.path)
        
        seq_dir=sorted(seq_dir)
        gpu_num=12
        length=int(len(seq_dir)/gpu_num)+1
        # args.part is [1,...,gpu_num].
        start=(args.part-1)*length
        end=min(start+length,len(seq_dir))
        seq_dir_part=seq_dir[start:end]
        for seq_one_path in tqdm(seq_dir_part):
            images = glob.glob(os.path.join(seq_one_path, '*.png')) + \
            glob.glob(os.path.join(seq_one_path, '*.jpg'))
            images = sorted(images)
            
            seq_name=Path(seq_one_path).stem
            flow_dir = os.path.join(out_dir,seq_name,"ckpt_sintel_flow_img_bidirect",)
            os.makedirs(flow_dir,exist_ok=True)
            np_path=os.path.join(out_dir, seq_name,"ckpt_sintel_np_offset_mat_bidirect")
            os.makedirs(np_path,exist_ok=True)

            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                # print(f"Reading in images at {imfile1} and {imfile2}")

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
                flow_up_np=flow_up.detach().clone().cpu().numpy()
                pair_name=seq_name+'_'+Path(imfile1).stem+'_'+Path(imfile2).stem
                img_name=pair_name+'.png'
                np_name=os.path.join(np_path,pair_name+'.npy')
                with open(np_name,'wb') as f:
                    np.save(f,flow_up_np)
                # print(f"Estimating optical flow...")

                my_viz(image1, flow_up, flow_dir,img_name)
            # below gets opt flow from img2 to img1 too.
            for imfile1, imfile2 in zip(images[1:], images[:-1]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                # print(f"Reading in images at {imfile1} and {imfile2}")

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
                flow_up_np=flow_up.detach().clone().cpu().numpy()
                pair_name=seq_name+'_'+Path(imfile1).stem+'_'+Path(imfile2).stem
                img_name=pair_name+'.png'
                np_name=os.path.join(np_path,pair_name+'.npy')
                with open(np_name,'wb') as f:
                    np.save(f,flow_up_np)
                # print(f"Estimating optical flow...")

                my_viz(image1, flow_up, flow_dir,img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--part',help='divide the whole seqdir to different parts and run',
                        type=int,required=True)
    args = parser.parse_args()

    demo(args)
