# Steps to install this modified SSIS folder
Create date: 2024/03/02. Test server: tesla.

## 1. Clone this folder
Git clone this private GitHub folder with PAT key.

Command: `git clone https://<pat>@github.com/<your account or organization>/<repo>.git`

Note that we need to clone my own SSIS folder, where the file `adet/layers/csrc/ml_nms/ml_nms.cu` is modified compared to the official SSIS project. Because we use `pytorch=1.8.1` with `cuda=11.1`, we have to use the old version `adet/layers/csrc/ml_nms/ml_nms.cu` file of old version `AdelaiDet` project.

GitHub issues of SSIS have mentioned [this](https://github.com/stevewongv/SSIS/issues/6). If we want to directly use the official SSIS project, we have to use `pytorch=1.11+`, but this pytorch version requires `cuda=11.3+`, which is not installed in the server tesla. We have installed `cuda=11.1` on the server tesla.

## 2. Create a conda env
Command: `conda create -n myenv python=3.8`

## 3. Pip install PyTorch 1.8.1 with cuda 11.1
Command: `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

We have to use `pytorch=1.8.1`, because `kornia=0.6.7` requires `pytorch=1.8.1+`, and `pytorch=1.8.0` cannot work after try.

If we use `pytorch=1.8.0`, we need to use `kornia=0.5.6`. This is mentioned by the updated `requirement.txt` in the official SSIS project, but I have not tried yet.

Use pip to install `pytorch=1.8.1` is because using conda will show `solving environment` forever.

## 4. Install Detectron 2 for PyToch 1.8 and cuda 11.1
Command: `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html`

## 5. Install numpy 1.23.2
Command: `pip install numpy==1.23.2`

## 6. Install old version pillow
Command: `pip install pillow==9.2.0`

The default installed latest pillow version 10.0+ will report error `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'`, and we need to install old version pillow to fix this issue.

## 7. Install skimage 0.19.3
Command: `pip install scikit-image==0.19.3`

## 8. Install kornia 0.6.7
Command: `pip install kornia==0.6.7`

## 9. Install pycocotools
Command: `pip install pycocotools`

## 10. Install opencv-python
Command: `pip install opencv-python`

Ths opencv-python package is needed for demo program.

## 11. Install old version pysobatools
Command: `pip install git+https://github.com/stevewongv/InstanceShadowDetection.git@50764eb336f3194db382054fe537956dd8449c01#subdirectory=PythonAPI`

If we use the official SSIS project with `python=3.8`, it will report error:

`/usr/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`

We need to install a specific version of `pysobatools` to fix this. This problem is mentioned in the official SSIS project GitHub issues [here](https://github.com/stevewongv/SSIS/issues/13).

THe author says using `python=3.10` with the latest version SSIS project should work in this [issue](https://github.com/stevewongv/InstanceShadowDetection/issues/24), but I have not tried yet.

## 12. Build the downloaded folder
Command: `cd SSIS` and then `python setup.py build develop`.

