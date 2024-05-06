import numpy as np
from scipy.spatial.distance import cdist
import pickle,time

def get_pkl_and_del_nan(file_name):
    with open(file_name,'rb') as f:
        out_all=pickle.load(f)
    track_id_np=np.array([x["track_id"] for x in out_all])
    # below get reid feat only.
    reid_feat_np=np.array([x["reid_feat"] for x in out_all])
    # below del nan.
    no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
    reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
    track_id_np_no_nan=track_id_np[no_nan_id]
    return track_id_np_no_nan,reid_feat_np_no_nan

if __name__=='__main__':
    t1=time.time()
    # file_name="../../training_dir/test_only/reid_gt_mask_pool_eval/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pkl"
    file_name="../../training_dir/test_only/reid_eval_v2/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pkl"
    track_id_np,reid_feat_np=get_pkl_and_del_nan(file_name)

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
    top_n=5
    track_id_np=np.array(track_id_sort_all)
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

    # box reid-condinst
    # top 1:0.9526195499296765
    # top 5:98.4089
    # box reid v2-condinst tune param
    # top 1:92.6072
    # top 5:97.4684
    # mask reid
    # top 1:91.8601
    # top 5:96.8003
    # mask reid no nan
    # top 1:92.0543
    # top 5:97.0049
    # gt mask reid no nan
    # iter 5999.
    # top 1:93.1818
    # top 5:97.4278
    # iter 5899.
    # top 1:93.1730
    # top 5:97.4366
    top_n_acc=correct_count/len(track_id_np)
    acc_out=format(top_n_acc*100,".4f")
    t2=time.time()-t1
    print("Time:",format(t2,".4f"))
    print(f"Top {top_n} Acc: {acc_out}")

