# Overview
Our project includes four parts: main model, data association method, optical flow model, and shadow detection model.

## Main Model
Main model generates detection, segmentation, and object embedding results.

## Data Association Method
We use [DeepSORT](https://github.com/dyhBUPT/StrongSORT) for data association, based on both motion and appearance information.

## Optical Flow Model
We use [GMA](https://github.com/zacjiang/GMA) to generate optical flow results for the KITTI MOTS and BDD100K MOTS training sets. Optical flow results are used to train the main model.

## Shadow Detection Model
We use [SSIS](https://github.com/stevewongv/SSIS) to detect the shadow and remove it from the car-like object's segmentation result. Shadow detection results are used in the inference process.

## TODO
- [x] Repo setup.
- [x] Add code of main model.

## Acknowledgements
- Thanks [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for the BoxInst implementation.
- Thanks [StrongSORT](https://github.com/dyhBUPT/StrongSORT) for the DeepSORT implementation.
- Thanks [GMA](https://github.com/zacjiang/GMA) for the optical flow model.
- Thanks [SSIS](https://github.com/stevewongv/SSIS) for the shadow detection model.
