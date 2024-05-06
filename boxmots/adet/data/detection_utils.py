import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

import math
import json

# below by cws.
from detectron2.data.detection_utils import \
    annotations_to_instances_reid as d2_anno_to_inst_reid
from pathlib import Path

bdd_train_videos_id=[
    {
      "id": 1,
      "name": "0000f77c-6257be58"
    },
    {
      "id": 2,
      "name": "0000f77c-62c2a288"
    },
    {
      "id": 3,
      "name": "0000f77c-cb820c98"
    },
    {
      "id": 4,
      "name": "0001542f-5ce3cf52"
    },
    {
      "id": 5,
      "name": "0001542f-7c670be8"
    },
    {
      "id": 6,
      "name": "0001542f-ec815219"
    },
    {
      "id": 7,
      "name": "00054602-3bf57337"
    },
    {
      "id": 8,
      "name": "00067cfb-5443fe39"
    },
    {
      "id": 9,
      "name": "00067cfb-5adfaaa7"
    },
    {
      "id": 10,
      "name": "00067cfb-caba8a02"
    },
    {
      "id": 11,
      "name": "00067cfb-e535423e"
    },
    {
      "id": 12,
      "name": "00067cfb-f1b91e3c"
    },
    {
      "id": 13,
      "name": "0008a165-c48f4b3e"
    },
    {
      "id": 14,
      "name": "00091078-59817bb0"
    },
    {
      "id": 15,
      "name": "00091078-7cff8ea6"
    },
    {
      "id": 16,
      "name": "00091078-84635cf2"
    },
    {
      "id": 17,
      "name": "00091078-875c1f73"
    },
    {
      "id": 18,
      "name": "00091078-c1d32eea"
    },
    {
      "id": 19,
      "name": "00091078-cedbfea7"
    },
    {
      "id": 20,
      "name": "00091078-f32de4d2"
    },
    {
      "id": 21,
      "name": "000d35d3-41990aa4"
    },
    {
      "id": 22,
      "name": "000d4f89-3bcbe37a"
    },
    {
      "id": 23,
      "name": "000e0252-8523a4a9"
    },
    {
      "id": 24,
      "name": "000f157f-30b30f5e"
    },
    {
      "id": 25,
      "name": "000f157f-37797ff9"
    },
    {
      "id": 26,
      "name": "000f157f-dab3a407"
    },
    {
      "id": 27,
      "name": "000f8d37-d4c09a0f"
    },
    {
      "id": 28,
      "name": "0010bf16-9ee17cd9"
    },
    {
      "id": 29,
      "name": "0010bf16-a457685b"
    },
    {
      "id": 30,
      "name": "00131ea7-624f538d"
    },
    {
      "id": 31,
      "name": "00134776-9123d227"
    },
    {
      "id": 32,
      "name": "001bad4e-2fa8f3b6"
    },
    {
      "id": 33,
      "name": "001c5339-08faca55"
    },
    {
      "id": 34,
      "name": "001c5339-13a07470"
    },
    {
      "id": 35,
      "name": "001c5339-9a6cdd3e"
    },
    {
      "id": 36,
      "name": "00202076-7a95b4e3"
    },
    {
      "id": 37,
      "name": "00202076-9eaa8e42"
    },
    {
      "id": 38,
      "name": "00207869-046fa443"
    },
    {
      "id": 39,
      "name": "00207869-902288d1"
    },
    {
      "id": 40,
      "name": "00211b17-ad3e206b"
    },
    {
      "id": 41,
      "name": "00225f53-4200bde2"
    },
    {
      "id": 42,
      "name": "00225f53-67614580"
    },
    {
      "id": 43,
      "name": "00232de3-19eca24a"
    },
    {
      "id": 44,
      "name": "0024b742-83709bd4"
    },
    {
      "id": 45,
      "name": "0024b742-acbed4fb"
    },
    {
      "id": 46,
      "name": "002685b6-856c17f7"
    },
    {
      "id": 47,
      "name": "00268999-0b20ef00"
    },
    {
      "id": 48,
      "name": "00268999-9f6d5823"
    },
    {
      "id": 49,
      "name": "00268999-a4b8e39d"
    },
    {
      "id": 50,
      "name": "00268999-cb063914"
    },
    {
      "id": 51,
      "name": "002827ad-8748e2fb"
    },
    {
      "id": 52,
      "name": "0028cbbf-92f30408"
    },
    {
      "id": 53,
      "name": "002ab96a-a96704f9"
    },
    {
      "id": 54,
      "name": "002ab96a-ea678692"
    },
    {
      "id": 55,
      "name": "002b485a-3f6603f2"
    },
    {
      "id": 56,
      "name": "002b485a-d1301c7c"
    },
    {
      "id": 57,
      "name": "002b562f-e0ac84fe"
    },
    {
      "id": 58,
      "name": "002cd38e-c7defded"
    },
    {
      "id": 59,
      "name": "002cd38e-ebe888e1"
    },
    {
      "id": 60,
      "name": "002d290d-01969e7d"
    },
    {
      "id": 61,
      "name": "002d290d-89d1aea8"
    },
    {
      "id": 62,
      "name": "002d290d-89f4e5c0"
    },
    {
      "id": 63,
      "name": "002d290d-90f2bab2"
    },
    {
      "id": 64,
      "name": "002d290d-f9108a13"
    },
    {
      "id": 65,
      "name": "002d290d-fa2a8964"
    },
    {
      "id": 66,
      "name": "002e6895-442e6bc1"
    },
    {
      "id": 67,
      "name": "0030f434-3eb4a3a9"
    },
    {
      "id": 68,
      "name": "00313a01-62156032"
    },
    {
      "id": 69,
      "name": "00313a01-725ddf3a"
    },
    {
      "id": 70,
      "name": "00313a01-97de1f42"
    },
    {
      "id": 71,
      "name": "00313a01-b53a2998"
    },
    {
      "id": 72,
      "name": "00322f82-286ab9b3"
    },
    {
      "id": 73,
      "name": "0032419c-7d20b300"
    },
    {
      "id": 74,
      "name": "0033b19f-65613f7e"
    },
    {
      "id": 75,
      "name": "0033b19f-7f07efac"
    },
    {
      "id": 76,
      "name": "0034a363-24f318c4"
    },
    {
      "id": 77,
      "name": "0035afff-3295dbd6"
    },
    {
      "id": 78,
      "name": "0035afff-434368a1"
    },
    {
      "id": 79,
      "name": "0035afff-47378fa3"
    },
    {
      "id": 80,
      "name": "0035afff-572b2d4e"
    },
    {
      "id": 81,
      "name": "0035afff-bd191d6a"
    },
    {
      "id": 82,
      "name": "00378858-c5f802ac"
    },
    {
      "id": 83,
      "name": "00390995-49d9a01e"
    },
    {
      "id": 84,
      "name": "00391a82-8be5b76d"
    },
    {
      "id": 85,
      "name": "00391a82-d1428e56"
    },
    {
      "id": 86,
      "name": "003baca5-70c87fc6"
    },
    {
      "id": 87,
      "name": "003baca5-aab2e274"
    },
    {
      "id": 88,
      "name": "003baca5-ad660439"
    },
    {
      "id": 89,
      "name": "003baca5-d6cd84e5"
    },
    {
      "id": 90,
      "name": "003c4a61-52588960"
    },
    {
      "id": 91,
      "name": "003e23ee-07d32feb"
    },
    {
      "id": 92,
      "name": "003e23ee-67d25f19"
    },
    {
      "id": 93,
      "name": "004071a4-049b7b85"
    },
    {
      "id": 94,
      "name": "004071a4-4e8a363a"
    },
    {
      "id": 95,
      "name": "004071a4-a45d905f"
    },
    {
      "id": 96,
      "name": "004071a4-ef4bf541"
    },
    {
      "id": 97,
      "name": "00417c23-220bbc98"
    },
    {
      "id": 98,
      "name": "00423717-0ef3c8dc"
    },
    {
      "id": 99,
      "name": "00423717-eba4da41"
    },
    {
      "id": 100,
      "name": "00423ac5-1fa7ff43"
    },
    {
      "id": 101,
      "name": "0045e757-02530231"
    },
    {
      "id": 102,
      "name": "0045e757-088840fc"
    },
    {
      "id": 103,
      "name": "0045e757-66a334b5"
    },
    {
      "id": 104,
      "name": "004855fc-ff3946ad"
    },
    {
      "id": 105,
      "name": "00488a66-6c729bde"
    },
    {
      "id": 106,
      "name": "00488b40-0a8cf1a0"
    },
    {
      "id": 107,
      "name": "0048f391-2c5344eb"
    },
    {
      "id": 108,
      "name": "0048f391-8eb40ca6"
    },
    {
      "id": 109,
      "name": "0048f391-e9bfaf62"
    },
    {
      "id": 110,
      "name": "0048f391-eae6a189"
    },
    {
      "id": 111,
      "name": "0049e5b8-725e21a0"
    },
    {
      "id": 112,
      "name": "0049e5b8-afda7206"
    },
    {
      "id": 113,
      "name": "004ea016-0b1932a7"
    },
    {
      "id": 114,
      "name": "00516b75-7ac91661"
    },
    {
      "id": 115,
      "name": "0051e391-d32a618e"
    },
    {
      "id": 116,
      "name": "0052b279-87c692aa"
    },
    {
      "id": 117,
      "name": "005645a1-dbe34c9d"
    },
    {
      "id": 118,
      "name": "00589f25-4cf2b9e0"
    },
    {
      "id": 119,
      "name": "0059f17f-f0882eef"
    },
    {
      "id": 120,
      "name": "005c4fd3-cb4d6287"
    },
    {
      "id": 121,
      "name": "005cdef0-180a776c"
    },
    {
      "id": 122,
      "name": "005ee183-0100ac18"
    },
    {
      "id": 123,
      "name": "005ee183-bcfab89f"
    },
    {
      "id": 124,
      "name": "005fd7be-d8e0dc65"
    },
    {
      "id": 125,
      "name": "005ff190-5be4cdfa"
    },
    {
      "id": 126,
      "name": "00618cec-3bc5573e"
    },
    {
      "id": 127,
      "name": "00618cec-a7f7e470"
    },
    {
      "id": 128,
      "name": "00618cec-ef92ac05"
    },
    {
      "id": 129,
      "name": "0062298d-2d787502"
    },
    {
      "id": 130,
      "name": "0062298d-cbbec2cd"
    },
    {
      "id": 131,
      "name": "0062298d-e6abad2f"
    },
    {
      "id": 132,
      "name": "0062298d-fd69d0ec"
    },
    {
      "id": 133,
      "name": "0062ab5a-54bb129b"
    },
    {
      "id": 134,
      "name": "0062e803-38c0a33a"
    },
    {
      "id": 135,
      "name": "0062f18d-f8cd3a65"
    },
    {
      "id": 136,
      "name": "006382a3-4a442001"
    },
    {
      "id": 137,
      "name": "0065fe83-c80cf6c3"
    },
    {
      "id": 138,
      "name": "006676e5-0a512e75"
    },
    {
      "id": 139,
      "name": "006676e5-624162a7"
    },
    {
      "id": 140,
      "name": "0066b72f-974f6883"
    },
    {
      "id": 141,
      "name": "00690c26-e4bbbd72"
    },
    {
      "id": 142,
      "name": "00699de6-58847872"
    },
    {
      "id": 143,
      "name": "006a4209-286a5664"
    },
    {
      "id": 144,
      "name": "006a4209-4f3bf6cf"
    },
    {
      "id": 145,
      "name": "006a7635-c42f9f97"
    },
    {
      "id": 146,
      "name": "006a7635-ec8fe02c"
    },
    {
      "id": 147,
      "name": "006bb043-fe4bbaf6"
    },
    {
      "id": 148,
      "name": "006bff20-752bd845"
    },
    {
      "id": 149,
      "name": "006c0799-10697723"
    },
    {
      "id": 150,
      "name": "006c0799-964a2695"
    },
    {
      "id": 151,
      "name": "006d6f8e-4759e8c1"
    },
    {
      "id": 152,
      "name": "006dd004-34630e8d"
    },
    {
      "id": 153,
      "name": "006fdb67-526bb14e"
    },
    {
      "id": 154,
      "name": "006fdb67-f4820206"
    }
  ]

bdd_val_videos_id=[
    {
      "id": 1,
      "name": "b1c66a42-6f7d68ca"
    },
    {
      "id": 2,
      "name": "b1c81faa-3df17267"
    },
    {
      "id": 3,
      "name": "b1c81faa-c80764c5"
    },
    {
      "id": 4,
      "name": "b1c9c847-3bda4659"
    },
    {
      "id": 5,
      "name": "b1ca2e5d-84cf9134"
    },
    {
      "id": 6,
      "name": "b1cd1e94-549d0bfe"
    },
    {
      "id": 7,
      "name": "b1ceb32e-3f481b43"
    },
    {
      "id": 8,
      "name": "b1ceb32e-51852abe"
    },
    {
      "id": 9,
      "name": "b1cebfb7-284f5117"
    },
    {
      "id": 10,
      "name": "b1d0091f-75824d0d"
    },
    {
      "id": 11,
      "name": "b1d0091f-f2c2d2ae"
    },
    {
      "id": 12,
      "name": "b1d0a191-03dcecc2"
    },
    {
      "id": 13,
      "name": "b1d0a191-06deb55d"
    },
    {
      "id": 14,
      "name": "b1d0a191-28f0e779"
    },
    {
      "id": 15,
      "name": "b1d0a191-2ed2269e"
    },
    {
      "id": 16,
      "name": "b1d0a191-5490450b"
    },
    {
      "id": 17,
      "name": "b1d0a191-65deaeef"
    },
    {
      "id": 18,
      "name": "b1d0a191-de8948f6"
    },
    {
      "id": 19,
      "name": "b1d10d08-5b108225"
    },
    {
      "id": 20,
      "name": "b1d10d08-743fd86c"
    },
    {
      "id": 21,
      "name": "b1d10d08-c35503b8"
    },
    {
      "id": 22,
      "name": "b1d10d08-da110fcb"
    },
    {
      "id": 23,
      "name": "b1d10d08-ec660956"
    },
    {
      "id": 24,
      "name": "b1d22449-117aa773"
    },
    {
      "id": 25,
      "name": "b1d22449-15fb948f"
    },
    {
      "id": 26,
      "name": "b1d22ed6-f1cac061"
    },
    {
      "id": 27,
      "name": "b1d3907b-2278601b"
    },
    {
      "id": 28,
      "name": "b1d4b62c-60aab822"
    },
    {
      "id": 29,
      "name": "b1d59b1f-a38aec79"
    },
    {
      "id": 30,
      "name": "b1d7b3ac-0bdb47dc"
    },
    {
      "id": 31,
      "name": "b1d7b3ac-36f2d3b7"
    },
    {
      "id": 32,
      "name": "b1d7b3ac-5744370e"
    }
  ]
bdd_video_id_dict={z['name']:z['id'] for z in bdd_train_videos_id}
bdd_val_video_id_dict={z['name']:z['id'] for z in bdd_val_videos_id}

# ytvis 2019 dict.
json_path = "my_dataset/youtube_vis_2019/label_for_training_model_with_optical_flow/train_img_filename_to_full_info_dict.json"
ytvis_2019_img_filename_to_full_info_dict=json.load(open(json_path,'r'))

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_beziers_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    # changed by cws.
    # add id of each object to instance.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    # instance = d2_anno_to_inst_reid(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def annotations_to_instances_reid(image_id,annos, image_size, mask_format="polygon"):
    # image_id is needed to get track_id of one object.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    # change object_id to track_id.
    # track_id=annos[0]['id']-image_id*1000-((annos[0]['category_id']+1)*100)
    track_id_list=[x['id']-image_id*1000-((x['category_id']+1)*100) for x in annos]
    track_id_tensor=torch.tensor(track_id_list,dtype=torch.int64)
    instance.track_ids=track_id_tensor


    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def annotations_to_instances_reid_right_track(image_id,annos, image_size, mask_format="polygon"):
    # image_id is needed to get track_id of one object.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    # below track_id_list is only unique in one seq, not unique across different seqs,
    # since in the original label txt, object_id is only unique in one seq,
    # and different seqs could have same object_id.
    track_id_list=[x['id']-image_id*1000-((x['category_id']+1)*100) for x in annos]
    # below make track_id unique across seqs by adding seq_id to the object_id.
    seq_id=image_id//10000
    track_id_with_seq_list=[x+seq_id*10000 for x in track_id_list]
    track_id_tensor=torch.tensor(track_id_with_seq_list,dtype=torch.int64)
    instance.track_ids=track_id_tensor

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def bdd_annotations_to_instances_reid_right_track(file_name,annos, image_size, mask_format="polygon"):
    # image_id is needed to get track_id of one object.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    folder=Path(file_name).stem.rsplit('-',1)[0]
    seq_id=bdd_video_id_dict[folder]
    # below make track_id unique across seqs by adding seq_id to the object_id.    
    track_id_with_seq_list=[x['instance_id']+seq_id*10000 for x in annos]
    track_id_tensor=torch.tensor(track_id_with_seq_list,dtype=torch.int64)
    instance.track_ids=track_id_tensor

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def bdd_annotations_to_instances_reid_right_track_eval(file_name,annos, image_size, mask_format="polygon"):
    # image_id is needed to get track_id of one object.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    folder=Path(file_name).stem.rsplit('-',1)[0]
    seq_id=bdd_val_video_id_dict[folder]
    # below make track_id unique across seqs by adding seq_id to the object_id.    
    track_id_with_seq_list=[x['instance_id']+seq_id*10000 for x in annos]
    track_id_tensor=torch.tensor(track_id_with_seq_list,dtype=torch.int64)
    instance.track_ids=track_id_tensor

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def ytvis_2019_annotations_to_instances_reid_right_track(file_name,annos, image_size, mask_format="polygon"):
    # image_id is needed to get track_id of one object.
    instance = d2_anno_to_inst(annos, image_size, mask_format)
    
    # track_id in youtube-vis-2019 is already unique across seqs.  
    track_id_with_seq_list=[x['instance_id'] for x in annos]
    track_id_tensor=torch.tensor(track_id_with_seq_list,dtype=torch.int64)
    instance.track_ids=track_id_tensor

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance

def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""



class HeatmapGenerator():
    def __init__(self, num_joints, sigma, head_sigma):
        self.num_joints = num_joints
        self.sigma = sigma
        self.head_sigma = head_sigma

        self.p3_sigma = sigma / 2

        size = 2*np.round(3 * sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        size = 2*np.round(3 * self.p3_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.p3_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.p3_sigma ** 2))

        size = 2*np.round(3 * head_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.head_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * head_sigma ** 2))

    def __call__(self, gt_instance, gt_heatmap_stride):
        heatmap_size = gt_instance.image_size
        heatmap_size = [math.ceil(heatmap_size[0]/ 32)*(32/gt_heatmap_stride),
                    math.ceil(heatmap_size[1]/ 32)*(32/gt_heatmap_stride)]

        h,w = heatmap_size
        h,w = int(h),int(w) 
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / gt_heatmap_stride
        sigma = self.sigma
        head_sigma = self.head_sigma
        p3_sigma = self.p3_sigma

        output_list = []
        head_output_list = []
        for p in joints:
            hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            head_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

                    ul = int(np.round(x - 3 * head_sigma - 1)), int(np.round(y - 3 * head_sigma - 1))
                    br = int(np.round(x + 3 * head_sigma + 2)), int(np.round(y + 3 * head_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    head_hms[idx, aa:bb, cc:dd] = np.maximum(
                        head_hms[idx, aa:bb, cc:dd], self.head_g[a:b, c:d])
                    
            hms = torch.from_numpy(hms)
            head_hms = torch.from_numpy(head_hms)
            output_list.append(hms)
            head_output_list.append(head_hms)

        h,w = h//4, w//4
        p3_output_list = []
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / 8
        for p in joints:
            p3_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * p3_sigma - 1)), int(np.round(y - 3 * p3_sigma - 1))
                    br = int(np.round(x + 3 * p3_sigma + 2)), int(np.round(y + 3 * p3_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    p3_hms[idx, aa:bb, cc:dd] = np.maximum(
                        p3_hms[idx, aa:bb, cc:dd], self.p3_g[a:b, c:d])
                    
            p3_hms = torch.from_numpy(p3_hms)
            p3_output_list.append(p3_hms)
        output_list = torch.stack(output_list,dim=0)
        p3_output_list = torch.stack(p3_output_list,dim=0)
        head_output_list = torch.stack(head_output_list,dim=0)
        gt_instance.keypoint_heatmap = output_list
        gt_instance.head_heatmap = head_output_list
        gt_instance.p3_output_list = p3_output_list
        return gt_instance