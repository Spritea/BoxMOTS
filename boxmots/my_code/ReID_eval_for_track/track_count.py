import numpy as np
import json

# this file my_code/ReID_eval_for_track/track_count.py is to
# count the length of tracks.
def load_json(file_name):
    with open(file_name,'r') as f:
        json_content=json.load(f)
    return json_content

def get_track_id(json_content):
    track_id_with_seq_list=[]
    ann=json_content['annotations']
    # only keep annotations that are not ignored as crowd.
    ann=list(filter(lambda x:x['iscrowd']==0, ann))
    for line_one in ann:
        # note here line_one['category_id'] needn't +1,
        # since here cat:1,ped:2, but in detectron2 training, it's cat:0,ped:1.
        track_id=line_one['id']-line_one['image_id']*1000-(line_one['category_id']*100)
        # below make track_id unique across seqs by adding seq_id to the object_id.
        seq_id=line_one['image_id']//10000
        track_id_with_seq=track_id+seq_id*10000
        track_id_with_seq_list.append(track_id_with_seq)
    assert len(ann)==len(track_id_with_seq_list)
    return track_id_with_seq_list

if __name__=="__main__":
    # distinct
    # train total 530, >=8: 501, >8: 494.
    # val total 219, >=8: 211, >8: 208.
    train_file="../../my_dataset/KITTI_MOTS/annotations/train_in_trainval_gt_as_coco_instances.json"
    val_file="../../my_dataset/KITTI_MOTS/annotations/val_in_trainval_gt_as_coco_instances.json"
    filename=train_file
    json_content=load_json(filename)
    track_id_with_seq_list=get_track_id(json_content)
    items, counts = np.unique(track_id_with_seq_list, return_counts=True)
    long_count=np.count_nonzero(counts>=8)
    print(f'File path: {filename}')
    print(f'Total track number: {len(counts)}')
    print(f'Tracks with length >8 number: {long_count}')
    print('kk')