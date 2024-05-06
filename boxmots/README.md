# BoxMOTS Usage
## Installation
Please check the [my_install](my_install.md) doc for installation details.

## Training
Please use the [my_script_kitti](my_script_kitti.sh) script to train the model on KITTI MOTS. Use the [my_script_bdd](my_script_bdd.sh) script to train the model on BDD100K MOTS.

## Inference
Note that we perform evaluation on the validation set in the training process, and we save both segmentation and embedding results for each evaluation. Hence you can directly use these results as the inference outputs for the model, which is trained by a specific number of iterations, without the need to perform the inference process explicitly.