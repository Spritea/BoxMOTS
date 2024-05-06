import numpy as np
from scipy.spatial.distance import cdist
import pickle,time


cat_dict={'pedestrian':0,'rider':1,'car':2,'truck':3,
            'bus':4,'motorcycle':6,'bicycle':7}
def get_pkl_and_del_nan_by_category(file_name,category='all'):
    with open(file_name,'rb') as f:
        out_all=pickle.load(f)
    if category!='all':
        out_all=list(filter(lambda x:x['gt_class']==cat_dict[category],out_all))
    track_id_np=np.array([x["track_id"] for x in out_all])
    # below get reid feat only.
    reid_feat_np=np.array([x["reid_feat"] for x in out_all])
    # below del nan.
    no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
    reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
    track_id_np_no_nan=track_id_np[no_nan_id]
    return track_id_np_no_nan,reid_feat_np_no_nan

def run_reid_eval(track_id_np,reid_feat_np):
    t1=time.time()
    USE_NORM=False
    if not USE_NORM:
        dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
        dist_id=np.argsort(dist_mat,axis=1)
    else:
        # [:,None] is to expand norm dimension for division.
        reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
        dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
        dist_id=np.argsort(dist_mat_norm,axis=1)
    # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
    track_id_sort_all=[]
    for inst_one_id in dist_id:
        track_id_sort_one=track_id_np[inst_one_id]
        track_id_sort_all.append(track_id_sort_one)
    
    # below get top-n acc.
    top_n=1
    track_id_np=np.array(track_id_sort_all)
    target=track_id_np[:,0]
    # don't use instance that only appears 1 time as query image.
    track_id_np=track_id_np[np.count_nonzero(track_id_np==(target[:,None]),axis=1)>1]
    target=track_id_np[:,0]
    obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
    # below check whether any object only appears one time.
    # if so, they line should be removed from track_id_np.
    if obj_min_appear>1:
        print("Objects at least appear {:} time".format(obj_min_appear))
    if top_n==1:
        track_id_top_n=track_id_np[:,1]
        correct_count=np.count_nonzero((track_id_top_n==target))
    else:
        track_id_top_n=track_id_np[:,1:top_n+1]
        target=np.reshape(target,(len(target),-1))
        correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))

    # pedestrian reid eval:
    # Objects at least appear 2 time
    # Time: 23.0670
    # Top 1 Acc: 57.0602
    # rider reid eval:
    # Objects at least appear 4 time
    # Time: 0.0117
    # Top 1 Acc: 87.9271
    # car reid eval:
    # Objects at least appear 2 time
    # Time: 913.2862
    # Top 1 Acc: 80.2584
    # truck reid eval:
    # Objects at least appear 2 time
    # Time: 1.8277
    # Top 1 Acc: 94.7223
    # bus reid eval:
    # Objects at least appear 2 time
    # Time: 0.0516
    # Top 1 Acc: 97.2881
    # motorcycle reid eval:
    # Objects at least appear 118 time
    # Time: 0.0009
    # Top 1 Acc: 100.0000
    # bicycle reid eval:
    # Objects at least appear 2 time
    # Time: 0.0140
    # Top 1 Acc: 85.4123
    # Mean Acc by class: 86.0955
    top_n_acc=correct_count/len(track_id_np)
    acc_out=format(top_n_acc*100,".4f")
    t2=time.time()-t1
    print("Time:",format(t2,".4f"))
    print(f"Top {top_n} Acc: {acc_out}")
    return top_n_acc

if __name__=='__main__':
    # file_name="../../training_dir/test_only/reid_gt_mask_pool_eval/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pkl"
    file_name="../../training_dir/test_only/bdd_data/reid_eval/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pkl"
    # can't compute based on all samples,
    # the system shows the process is killed.
    # print('all reid eval:')
    # track_id_np,reid_feat_np=get_pkl_and_del_nan_by_category(file_name)
    # run_reid_eval(track_id_np,reid_feat_np)
    acc_list=[]
    for cat in cat_dict.keys():
        print(f'{cat} reid eval:')
        track_id_np_cat,reid_feat_np_cat=get_pkl_and_del_nan_by_category(file_name,category=cat)
        acc=run_reid_eval(track_id_np_cat,reid_feat_np_cat)
        acc_list.append(acc)
    acc_mean=np.mean(acc_list)
    print(f'Mean Acc by class: {format(acc_mean*100,".4f")}')