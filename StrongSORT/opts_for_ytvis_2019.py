"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join
import os

data = {
    # add KITTI MOTS
    'KITTI_MOTS':{
      'val_in_trainval':[
          '0002',
          '0006',
          '0007',
          '0008',
          '0010',
          '0013',
          '0014',
          '0016',
          '0018',
      ]  
    },
    # add BDD MOTS.
    'BDD_MOTS':{
      'val_set':[
          '0001',
          '0002',
          '0003',
          '0004',
          '0005',
          '0006',
          '0007',
          '0008',
          '0009',
          '0010',
          '0011',
          '0012',
          '0013',
          '0014',
          '0015',
          '0016',
          '0017',
          '0018',
          '0019',
          '0020',
          '0021',
          '0022',
          '0023',
          '0024',
          '0025',
          '0026',
          '0027',
          '0028',
          '0029',
          '0030',
          '0031',
          '0032',
      ]  
    },
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            help='MOT17 or MOT20',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLink',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            # default='dataset/MOTChallenge'
            default='dataset'

        )
        self.parser.add_argument(
            '--path_AFLink',
            default='results/StrongSORT_Git/AFLink_epoch20.pth'
        )
        # below add by cws.
        # to get dir_dets from command line.
        self.parser.add_argument(
            '--dir_dets',
            # default='results/StrongSORT_Git/tmp'
            default='my_data_for_ytvis_2019_track_result/youtube_vis_2019_out_no_pair_warp/valid_for_VIS/combined_result_all_videos/reid_np/airplane'
        )
        self.parser.add_argument(
            '--dir_save',
            # default='results/StrongSORT_Git/tmp'
            default='my_data_for_ytvis_2019_track_result/youtube_vis_2019_out_no_pair_warp/valid_for_VIS/combined_result_all_videos/results_mot/DeepSORT/min_det_conf_06/airplane'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        # opt.min_confidence = 0.4
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.BoT:
            opt.max_cosine_distance = 0.4
            opt.dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+BoT'.format(opt.dataset, opt.mode)
        else:
            opt.max_cosine_distance = 0.3
            # opt.dir_dets = 'dataspace/{}_{}_YOLOX+simpleCNN'.format(opt.dataset, opt.mode)
            # opt.dir_dets = 'my_data/reid_one_class_infer_pair_warp_right_track/COCO_pretrain_strong/search_for_loss_combination/long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_40_no_color_sim_pretrain_weights_v3_time_7/BoxInst_MS_R_50_1x_kitti_mots/reid_np/iter_0007919/pedestrian'

        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        if opt.ECC:
            path_ECC = '/data/dyh/results/StrongSORT_Git/{}_ECC_{}.json'.format(opt.dataset, opt.mode)
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = data[opt.dataset][opt.mode]
        # opt.dir_dataset = join(
        #     opt.root_dataset,
        #     opt.dataset,
        #     'train' if opt.mode == 'val' else 'test'
        # )
        opt.dir_dataset="my_data_for_ytvis_2019/youtube_vis_2019_valid_for_VIS_rename_for_deepsort"
        # opt.dir_dataset="dataset/BDD_MOTS/val_set"
        # below add by cws.
        os.makedirs(opt.dir_save,exist_ok=True)
        return opt

# below add ytvis 2019 valid seqs for VIS to the data dict.
def load_txt(file_name):
    with open(file_name,'r') as f:
        txt_content=f.read().splitlines()
    return txt_content
valid_seqs_txt = "my_code_for_ytvis_2019/valid_set_video_names_for_VIS.txt"
data['YouTube_VIS_2019'] = {}
data['YouTube_VIS_2019']['valid'] = load_txt(valid_seqs_txt)

opt = opts().parse()