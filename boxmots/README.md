# BoxMOTS Usage
## Dataset Preparation
### KITTI MOTS
- Download: Go to the official [homepage](https://www.vision.rwth-aachen.de/page/mots) to download the images and labels, and check the [label format](https://www.vision.rwth-aachen.de/page/mots#:~:text=code%20on%20github-,Annotation%20Format,-We%20provide%20two).
- Convert format: Use the code mentioned in the [issue](https://github.com/VisualComputingInstitute/TrackR-CNN/issues/60) to convert the label to the COCO format. The label after convertion is provided [here](https://github.com/Spritea/BoxMOTS/releases/download/v0.1/annotations.zip).
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
- Dataset toolkit: Download the official dataset [toolkit](https://github.com/bdd100k/bdd100k) with commit `b7e1781` for data format convertion and result evaluation. Install `scalabel=0.3.0` required by the toolkit.
- Convert format: It is already in COCO label format. The label after convertion is provided [here](https://github.com/Spritea/BoxMOTS/releases/download/v0.1/from_rles.zip).
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
2. Train the BoxMOTS model with the [boxmots](.) folder. Check [Training](#training) for details.
3. Run [SSIS](https://github.com/Spritea/BoxMOTS/tree/main/SSIS) to get the shadow detectin results of KITTI/BDD.
4. Combine the shadow detection result to refine the model's segmentation result. [This code](https://github.com/Spritea/BoxMOTS/blob/main/boxmots/my_code/for_shadow/combine_shadow.py) for KITTI, and [this code](https://github.com/Spritea/BoxMOTS/blob/main/boxmots/my_code/for_shadow/combine_shadow_bdd_by_class.py) for BDD.
5. Run [StrongSORT](https://github.com/Spritea/BoxMOTS/tree/main/StrongSORT) for the data association. This step generates mask-based trajectories, which is the final output of the MOTS task.

## Installation
Please check the [my_install](my_install.md) doc for installation details of the boxmots folder.

## Training
1. **Optical flow data**. Make sure you have obtained the optical flow information of the dataset with [GMA](https://github.com/Spritea/BoxMOTS/tree/main/GMA) before training.
   
2. **BoxInst weights**. BoxMOTS uses BoxInst for instance segmentation. Check its official [page](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BoxInst/README.md) to download the `BoxInst_MS_R_50_3x.pth` model weights. It will be loaded to initialize BoxMOTS when the training starts.

3. **KITTI MOTS**. Please use the [my_script_kitti](my_script_kitti.sh) script, and change the config file in the script to this [config](configs/BoxInst_ReID_One_Class_Infer_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k.yaml) to train the full model (using optical flow information) on KITTI MOTS.
You can also train a base model without using optical flow with the [base](configs/BoxInst_ReID_One_Class_Infer_Right_Track/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k.yaml) config first, and then train the full model with the base model weights as initialization by modifying the `WEIGHTS` part in the [config](configs/BoxInst_ReID_One_Class_Infer_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_kitti_mots_coco_pretrain_strong_long_epoch_seq_shuffle_fl_2_lr_0_0001_bs_4_eval_500_steps_4k.yaml). You may also need to adjust other hyper-parameters, like loss weights, learning rate decay steps, and so on. This may give you more stable training process and (probably) a little better performance.

4. **BDD100K MOTS**. Use the [my_script_bdd](my_script_bdd.sh) script, and change the config file in the script to this [config](configs/BDD_DATA/BoxInst_ReID_One_Class_Infer_BDD_Pair_Warp_ReID_Eval_In_Train/MS_R_50_1x_bdd_mots_coco_pretrain_strong_iter_21k_seq_shuffle_fl_2_lr_0_001_bs_4_eval_500_no_color_sim.yaml) to train the full model (using optical flow information) on BDD100K MOTS. 

5. **BoxMOTS weights**. The trained BoxMOTS models are provided [here](https://github.com/Spritea/BoxMOTS/releases/tag/v0.1): the [model](https://github.com/Spritea/BoxMOTS/releases/download/v0.1/model_kitti_use_optical_flow.pth) trained on KITTI, and the [model](https://github.com/Spritea/BoxMOTS/releases/download/v0.1/model_bdd_use_optical_flow.pth) trained on BDD.

## Inference
Note that we perform evaluation on the validation set in the training process, and we save both segmentation and embedding results for each evaluation. Hence you can directly use these results as the inference outputs for the model, which is trained by a specific number of iterations, without the need to perform the inference process explicitly.

If you want to do inference only with pretrained model, check this [demo](demo_for_ytvis_2019/my_demo_multi_seq_ytvis_kitti_pretrain.py) on YouTube-VIS 2019 for reference.

## Notes
- Recommended project organization: 4 conda environments for the 4 components (BoxMOTS, GMA, SSIS, StrongSORT) to avoid package conflicts.
