import itertools
import random

world_size=4
rank=0
img_all_size=20
frame_link=4

id_list=list(range(img_all_size))
random.seed(123)
start_id=random.sample(range(frame_link),1)[0]

out_all=[]
for k in range(len(id_list)//frame_link):
    end_id=start_id+frame_link*(k+1)
    if end_id<len(id_list):
        out_one=id_list[start_id+frame_link*k:end_id]
        out_all.append(out_one)

random.seed(123)
list_random=random.sample(out_all,len(out_all))
list_out=list(itertools.chain(*list_random))
# above is for one gpu, below is for multi-gpu.

size_per_gpu=len(list_out)//world_size
list_per_gpu=list_out[rank*size_per_gpu:(rank+1)*size_per_gpu]
print('kk')