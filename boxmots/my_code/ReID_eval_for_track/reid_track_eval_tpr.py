import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from scipy import interpolate


def get_pkl_seq_one_by_img(file_name, seq):
    with open(file_name, 'rb') as f:
        out_all = pickle.load(f)
    out_all_seq_one = list(filter(lambda x: int(
        (x['img_info']['file_name']).split('/')[-2]) == seq, out_all))
    min_img_id = out_all_seq_one[0]['img_info']['image_id']
    max_img_id = out_all_seq_one[-1]['img_info']['image_id']
    out_all_by_img = {}
    for img_id in range(min_img_id, max_img_id+1):
        out_all_img_one = list(
            filter(lambda x: x['img_info']['image_id'] == img_id, out_all_seq_one))
        out_all_by_img[img_id] = out_all_img_one
    return out_all_by_img


def get_feat_del_nan_per_img(img_content):
    track_id_np = np.array([x["track_id"] for x in img_content])
    # below get reid feat only.
    reid_feat_np = np.array([x["reid_feat"] for x in img_content])
    # below del nan.
    no_nan_id = np.unique(np.where(~np.isnan(reid_feat_np))[0])
    reid_feat_np_no_nan = reid_feat_np[no_nan_id, :]
    track_id_np_no_nan = track_id_np[no_nan_id]
    return track_id_np_no_nan, reid_feat_np_no_nan


def reid_track_eval_two_frame(cur_img_content, next_img_content):
    cur_track_id, cur_reid_feat = get_feat_del_nan_per_img(cur_img_content)
    next_track_id, next_reid_feat = get_feat_del_nan_per_img(next_img_content)
    dist_mat = cdist(cur_reid_feat, next_reid_feat, metric='cosine')
    label_mat = cdist(np.expand_dims(cur_track_id, axis=1), np.expand_dims(
        next_track_id, axis=1), metric='cityblock')
    # change cos distance to cos similarity.
    sim_flatten = 1-dist_mat.flatten()
    # below change [-1,1] of cos dist to [0,1].
    prob_flatten = (sim_flatten+1)/2
    assert np.all(prob_flatten>=0) and np.all(prob_flatten<=1),"prob must be in [0,1]."
    label_flatten = [1 if x == 0 else 0 for x in label_mat.flatten()]
    return prob_flatten, label_flatten


def reid_eval_by_seq(content_by_img):
    prob_seq_one = []
    label_seq_one = []
    for cur_img_id, cur_img_content in content_by_img.items():
        next_img_id = cur_img_id+1
        if next_img_id not in content_by_img.keys()  \
                or len(cur_img_content) == 0 or len(content_by_img[next_img_id]) == 0:
            # cur_img could be the last img, or some imgs don't have gt_box.
            continue
        next_img_content = content_by_img[next_img_id]
        prob_pair_one, label_pair_one = reid_track_eval_two_frame(
            cur_img_content, next_img_content)
        prob_seq_one.extend(prob_pair_one)
        label_seq_one.extend(label_pair_one)
    return prob_seq_one,label_seq_one

if __name__ == '__main__':
    # fl_8: area: 0.9958639779203857
    # fl_2: area: 0.9966677055198824
    # fl_2_pair_triplet: area: 0.9965018464573892
    # fl_8.
    # TPR@FAR=0.0000010: 0.518565135
    # TPR@FAR=0.0000100: 0.518565135
    # TPR@FAR=0.0001000: 0.596781444
    # TPR@FAR=0.0010000: 0.787557314
    # TPR@FAR=0.0100000: 0.934190416
    # TPR@FAR=0.1000000: 0.992268273
    # fl_2.
    # TPR@FAR=0.0000010: 0.438460847
    # TPR@FAR=0.0000100: 0.438460847
    # TPR@FAR=0.0001000: 0.650633822
    # TPR@FAR=0.0010000: 0.817225569
    # TPR@FAR=0.0100000: 0.952890407
    # TPR@FAR=0.1000000: 0.992987503
    # fl_2_pair_triplet.
    # TPR@FAR=0.0000010: 0.263238335
    # TPR@FAR=0.0000100: 0.263238335
    # TPR@FAR=0.0001000: 0.622853547
    # TPR@FAR=0.0010000: 0.817045761
    # TPR@FAR=0.0100000: 0.939404837
    # TPR@FAR=0.1000000: 0.993347119
    val_list = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    framelink_8 = "../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_8_lr_0_005_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl"
    framelink_2 = '../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_2_lr_0_005_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl'
    framelink_8_pair_triplet='../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train_pair_triplet/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_8_lr_0_005_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl'
    framelink_8_lr_0_0001="../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_8_lr_0_0001_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl"
    framelink_2_lr_0_0001="../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_2_lr_0_0001_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl"
    framelink_8_pair_triplet_lr_0_0001="../../training_dir_hddb/reid_framelink/reid_one_class_infer_right_track_reid_eval_in_train_pair_triplet/COCO_pretrain_strong/search_for_loss_combination/seq_shuffle_fl_8_lr_0_0001_bs_8_eval_500/BoxInst_MS_R_50_1x_kitti_mots/reid_infer_and_eval_out/iter_0006999/reid_eval_out.pkl"
    file_name=framelink_8_pair_triplet_lr_0_0001
    prob_seq_all = []
    label_seq_all = []
    for seq_one in tqdm(val_list):
        out_all_by_img = get_pkl_seq_one_by_img(file_name, seq=seq_one)
        prob_seq_one,label_seq_one = reid_eval_by_seq(out_all_by_img)
        prob_seq_all.extend(prob_seq_one)
        label_seq_all.extend(label_seq_one)

    fpr, tpr, thersholds = roc_curve(label_seq_all, prob_seq_all)
    # for i, value in enumerate(thersholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    print(f'area: {roc_auc}')
    
    far_levels = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    interp = interpolate.interp1d(fpr, tpr)
    tar_at_far = [interp(x) for x in far_levels]
    for f,fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.9f}'.format(fa, tar_at_far[f]))