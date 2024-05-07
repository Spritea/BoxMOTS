import numpy as np
import pickle

file_path = "../my_data_for_MOSE/MOSE_car_seq_out_no_pair_warp/train_car_seq/18840617/reid_out/reid_infer_out.pkl"
with open(file_path,'rb') as f:
    pkl_content=pickle.load(f)
print('kk')