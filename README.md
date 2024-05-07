# Overview
This is the implementation of the BoxMOTS work. This project includes four parts: main model, data association method, optical flow model, and shadow detection model.

## Main Model
Main model generates detection, segmentation, and object embedding results. This part is contained in the [boxmots](boxmots) folder. Please go to the [README](boxmots/README.md) file under that folder for usage details.

## Data Association Method
We use [DeepSORT](https://github.com/dyhBUPT/StrongSORT) for data association, based on both motion and appearance information. This part is contained in the [StrongSORT](StrongSORT) folder. Please go to the the [README](StrongSORT/README.md) file under that folder for usage details.

## Optical Flow Model
We use the [GMA](https://github.com/zacjiang/GMA) method to generate optical flow results for the KITTI MOTS and BDD100K MOTS training sets. Optical flow results are used to train the main model. This part is contained in the [GMA](GMA) folder. Please go to the the [README](GMA/README.md) file under that folder for usage details.

## Shadow Detection Model
We use the [SSIS](https://github.com/stevewongv/SSIS) method to detect the shadow and remove it from the car-like object's segmentation result. Shadow detection results are used in the inference process. This part is contained in the [SSIS](SSIS) folder. Please go to the [README](SSIS/README.md) file under that folder for usage details.

## TODO
- [x] Repo setup.
- [x] Add code of main model.

## Acknowledgements
- Thanks [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for the BoxInst implementation.
- Thanks [StrongSORT](https://github.com/dyhBUPT/StrongSORT) for the DeepSORT implementation.
- Thanks [GMA](https://github.com/zacjiang/GMA) for the optical flow model.
- Thanks [SSIS](https://github.com/stevewongv/SSIS) for the shadow detection model.
