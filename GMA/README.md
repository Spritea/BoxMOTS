# GMA Usage
## Installation
Please check the [my_install](my_install.md) doc for installation details.
## Extract Optical Flow
1. Download the optical flow model `gma-sintel.pth` from the official [repo](https://github.com/zacjiang/GMA/tree/main/checkpoints).
2. Run [my_script.sh](my_script.sh) to extract optical flow information of the MOTS/BDD dataset. Remember to change the optical flow model and dataset path to yours. Both the optical flow visualization images and the optical flow data will be generated.
## Notes
- The generated optical flow data is large. Ensure your hard disk has enough space.
