# StrongSORT Usage
## Installation
This part is based on the [StrongSORT](https://github.com/dyhBUPT/StrongSORT) project. Please first check the installation steps on the StrongSORT project to build the conda environment.
We only use the DeepSORT method of the project.

## Overview
After we get the inference result (instance segmentaion result) of the main model, we need to generate the object trajectories. It includes following steps.
1. Convert coco format instance segmentation results to txt file.
2. Convert ReID results to box and features needed by StrongSORT.
3. Use the DeepSORT method implemented in the StrongSORT project to link the same objects across frames. This step is termed **Data Association**.
4. Match masks of the instance segmentation result to boxes in the tracks of the data association result.
5. Combine results of different classes together, and remove overlapping pixels. Overlapping pixels are not allowed in the MOTS task. This step generates the final MOTS results, which are in fact mask-based trajectories.

Then you can do the evaluation with the MOTS results.

## Usage
We provide two different ways to use this code, described below.
### One program per function
Programs for different functions are in the [my_code](my_code) folder for KITTI. Check this [description](my_code/the_readme_pipeline.txt) to understand the function of each python program.

For BDD dataset, check the [my_code_bdd](my_code_bdd) folder, and its [description](my_code_bdd/readme_bdd.txt).
### One script for all functions
Use the [script](my_code_pipeline/my_pipeline.sh) to finish all 5 steps with one operation for KITTI. It call python programs in the [my_code_pipeline](my_code_pipeline) folder. Ensure these paths in ths script are correct on your local device.

For BDD dataset, check this [script](my_code_pipeline_bdd/my_bdd_pipeline.sh) and the [my_code_pipeline_bdd](my_code_pipeline_bdd) folder.
