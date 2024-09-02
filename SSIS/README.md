# SSIS Usage
## Installation
Please check the [my_install](my_install.md) doc for installation details.
## Perform shadow detection
1. Download the shadow detection model `model_ssisv2_final.pth` from the official [repo](https://github.com/stevewongv/SSIS), and put it under `SSIS/tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/`.
2. Run [my_script.sh](my_script.sh)/[my_script_bdd.sh](my_script_bdd.sh) to get shadow detection results of the KITTI/BDD dataset. Remember to change the dataset path to yours.
## Notes
- Shadow detection is performed on the validation set as a refinement technique. It is not used in the model training process.
- The shadow detection results are available under the folder `SSIS/demo/my_json_result` for reference.
