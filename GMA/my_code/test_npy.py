import numpy as np

np_file="../my_results/KITTI_MOTS/train_in_trainval/0000/np_offset_mat/0000_000002_000003.npy"
with open(np_file,'rb') as f:
    a=np.load(f)
print('kk')