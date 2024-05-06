import pickle

file_name="test_data/reid_infer_out.pkl"
with open(file_name,'rb') as f:
    a=pickle.load(f)
    print('kk')

file_name="../data_out/reid_out/train/0a7a3629/reid_infer_out.pkl"
with open(file_name,'rb') as f:
    b=pickle.load(f)
    print('kk')