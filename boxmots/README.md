# BoxMOTS Usage
## Dataset Preparation
### KITTI MOTS
- Download: Go to the official [homepage](https://www.vision.rwth-aachen.de/page/mots) to download the images and labels, and check the [label format](https://www.vision.rwth-aachen.de/page/mots#:~:text=code%20on%20github-,Annotation%20Format,-We%20provide%20two).
- Convert format: Use the code mentioned in the [issue](https://github.com/VisualComputingInstitute/TrackR-CNN/issues/60) to convert the label to the COCO format.
- Evaluation: Use the [MOTS Tools](https://github.com/VisualComputingInstitute/mots_tools) to evaluate the MOTS performance.

<details>
<summary>KITTI MOTS Dataset structure sample.</summary>
  
```
├── KITTI_MOTS
│   ├── annotations
│       ├── train_in_trainval_gt_as_coco_instances.json
|       ├── val_in_trainval_gt_as_coco_instances.json
│   ├── imgs
│       ├── train_in_trainval
│           ├── 0000
|               ├── 000000.png
|               ├── 000001.png
|               ├── ...
|               ├── 000153.png
│           ├── 0001
│           ├── 0003
│           ├── ...
│           ├── 0020
│       ├── val_in_trainval
│           ├── 0002
│           ├── 0006
│           ├── 0007
│           ├── ...
│           ├── 0018
```

</details>

### BDD100K MOTS
- Download: Go to the BDD100K official [homepage](https://doc.bdd100k.com/download.html) to download MOTS [images](https://doc.bdd100k.com/download.html#mots-2020-images) and [labels](https://doc.bdd100k.com/download.html#mots-2020-labels), and check the [label format](https://doc.bdd100k.com/download.html#mots-2020-labels:~:text=2020%20The%20bitmask%20format%20is%20explained%20at%3A-,Instance%20Segmentation%20Format,-.).
- Convert format: It is already in COCO label format.
- Evaluation: Check this [page](https://doc.bdd100k.com/evaluate.html#multi-object-tracking-and-segmentation-segmentation-tracking) to see how to evaluate the MOTS performance on this dataset.

<details>
<summary>BDD100K MOTS Dataset structure sample.</summary>
  
```
├── bdd100k
│   ├── images
│       ├── seg_track_20
│           ├── train
│               ├── 000d4f89-3bcbe37a
│                   ├── 000d4f89-3bcbe37a-0000001.jpg
│                   ├── ...
│               ├── 000d35d3-41990aa4
│               ├── ...
│           ├── val
│   ├── labels
│       ├── seg_track_20
│           ├── bitmasks
|           ├── colormaps
│           ├── from_rles
│               ├── train_seg_track.json
|               ├── val_seg_track.json
|           ├── polygons
│           ├── rles

```

</details>

## Pipeline
1. Run [GMA](https://github.com/Spritea/BoxMOTS/tree/main/GMA) to extract the optical flow information of KITTI/BDD.
2. Train the BoxMOTS model with the [boxmots](.) folder.
3. Run [SSIS](https://github.com/Spritea/BoxMOTS/tree/main/SSIS) to get the shadow detectin results of KITTI/BDD.
4. Combine the shadow detection result to refine the model's segmentation result. [This code](https://github.com/Spritea/BoxMOTS/blob/main/boxmots/my_code/for_shadow/combine_shadow.py) for KITTI, and [this code](https://github.com/Spritea/BoxMOTS/blob/main/boxmots/my_code/for_shadow/combine_shadow_bdd_by_class.py) for BDD.
5. Run [StrongSORT](https://github.com/Spritea/BoxMOTS/tree/main/StrongSORT) for the data association. This step generates mask-based trajectories, which is the final output of the MOTS task.

## Installation
Please check the [my_install](my_install.md) doc for installation details of the BoxMOTS part.

## Training
Please use the [my_script_kitti](my_script_kitti.sh) script to train the model on KITTI MOTS. Use the [my_script_bdd](my_script_bdd.sh) script to train the model on BDD100K MOTS. Make sure you have obtained the optical flow information with [GMA](https://github.com/Spritea/BoxMOTS/tree/main/GMA) before training.

## Inference
Note that we perform evaluation on the validation set in the training process, and we save both segmentation and embedding results for each evaluation. Hence you can directly use these results as the inference outputs for the model, which is trained by a specific number of iterations, without the need to perform the inference process explicitly.

## Notes
- Recommended project organization: 4 conda environments for the 4 components (BoxMOTS, GMA, SSIS, StrongSORT) to avoid package conflicts.
