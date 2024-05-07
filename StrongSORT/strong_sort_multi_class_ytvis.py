# @File: StrongSORT/strong_sort_multi_class_ytvis.py 
# @Author: cws 
# @Create Date: 2024/04/22
# @Desc: Process multiple categroies for ytvis 2019 valid dataset.
# No need to manually change the opt.dir_dets, opt.dir_save in opts.py for each category.

import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run
from AFLink.AppFreeLink import *
from GSI import GSInterpolation

from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    # add by cws.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # ytvis 2019 category list.
    category_list = ["person", "giant_panda", "lizard", "parrot", "skateboard",
                     "sedan", "ape", "dog", "snake", "monkey", 
                     "hand", "rabbit", "duck", "cat", "cow",
                     "fish", "train", "horse", "turtle", "bear", 
                     "motorbike", "giraffe", "leopard", "fox", "deer",
                     "owl", "surfboard", "airplane", "truck", "zebra",
                     "tiger", "elephant", "snowboard", "boat", "shark",
                     "mouse", "frog", "eagle", "earless_seal", "tennis_racket"]
    
    for cls_ctr, category_one in enumerate(category_list):
        opt.category = category_one

        dir_dets = opt.dir_dets
        dir_save = opt.dir_save
        
        dir_dets_by_class = join(str(Path(dir_dets).parent), category_one)
        dir_save_by_class = join(str(Path(dir_save).parent), category_one)
        
        opt.dir_dets = dir_dets_by_class
        opt.dir_save = dir_save_by_class
        os.makedirs(opt.dir_save, exist_ok=True)
        
        print("Processing the {}th category: {}".format(cls_ctr, category_one))
        
        if opt.AFLink:
            model = PostLinker()
            model.load_state_dict(torch.load(opt.path_AFLink))
            dataset = LinkData('', '')
        for i, seq in tqdm(enumerate(opt.sequences, start=1)):
            # print('processing the {}th video {}...'.format(i, seq))
            path_save = join(opt.dir_save, seq + '.txt')
            run(
                sequence_dir=join(opt.dir_dataset, seq),
                detection_file=join(opt.dir_dets, seq + '.npy'),
                output_file=path_save,
                min_confidence=opt.min_confidence,
                nms_max_overlap=opt.nms_max_overlap,
                min_detection_height=opt.min_detection_height,
                max_cosine_distance=opt.max_cosine_distance,
                nn_budget=opt.nn_budget,
                display=False
            )
            if opt.AFLink:
                linker = AFLink(
                    path_in=path_save,
                    path_out=path_save,
                    model=model,
                    dataset=dataset,
                    thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
                    thrS=75,
                    thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
                )
                linker.link()
            if opt.GSI:
                GSInterpolation(
                    path_in=path_save,
                    path_out=path_save,
                    interval=20,
                    tau=10
                )




