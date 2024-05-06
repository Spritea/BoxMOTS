import numpy as np
import torch
from sklearn import metrics
from scipy import interpolate
from reid_rank_n import get_pkl_and_del_nan
import time

# this is mainly from JDE code.
t1=time.time()
file_name="../../training_dir/test_only/reid_eval_v2/CondInst_MS_R_50_1x_kitti_mots/reid_eval_out/reid_eval_out.pkl"
track_id_np,reid_feat_np=get_pkl_and_del_nan(file_name)

# embedding = torch.stack(embedding, dim=0).cuda()
# id_labels = torch.LongTensor(id_labels)
# n = len(id_labels)
# print(n, len(embedding))
# assert len(embedding) == n

reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
pdist=np.matmul(reid_feat_norm,reid_feat_norm.T)

id_labels=torch.tensor([1,2,3,4,1,7,3,9])
n=len(track_id_np)
# gt = id_labels.expand(n,n).eq(id_labels.expand(n,n).t()).numpy()
gt=np.equal(np.tile(track_id_np,(n,1)),np.tile(track_id_np,(n,1)).T)
# embedding = F.normalize(embedding, dim=1)
# pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
# gt = id_labels.expand(n,n).eq(id_labels.expand(n,n).t()).numpy()

up_triangle = np.where(np.triu(pdist)- np.eye(n)*pdist !=0)
pdist = pdist[up_triangle]
gt = gt[up_triangle]

far_levels = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
far,tar,threshold = metrics.roc_curve(gt, pdist)
area=metrics.auc(far,tar)
interp = interpolate.interp1d(far, tar)
tar_at_far = [interp(x) for x in far_levels]
for f,fa in enumerate(far_levels):
    print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
# return tar_at_far
# box reid-condinst
# far-0.1:0.4041.
# area:0.7679.
# box reid-condinst tune param
# far-0.1:0.3609
# area:0.7263
# mask reid no nan
# far-0.1:0.3666.
# area:0.7403.
print("area:",area)
print("time:",time.time()-t1)