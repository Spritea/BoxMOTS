# Steps to use GMA
Create data: 2024/02/19. Test server: tesla.

This folder does not need build. Only the environment is needed.
## 1. Clone this folder
Git clone this private GitHub folder with PAT key.

Command: `git clone https://<pat>@github.com/<your account or organization>/<repo>.git`

## 2. Follow the official repo [readme](https://github.com/Spritea/GMA?tab=readme-ov-file#environments) to set up the environment

### 2.1 Create a conda env
Command: `conda create -n myenv python=3.8`

### 2.2 Install PyTorch 1.8.0 with cuda 11.1
Command: `conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`

### 2.3 Install other required python packages
Command: `pip install matplotlib imageio einops scipy opencv-python`

### 2.4 Install other needed python packages not listed above
The python package `tqdm` is needed to run our own script, like `my_eval_and_save_bidirect.py`.

Command: `pip install tqdm`