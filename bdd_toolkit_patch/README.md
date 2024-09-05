# BDD Toolkit Patch Usage
This patch involves two modifications to the official toolkit: label format convertion, and MOTS evaluation.

## Official BDD Toolkit
Download the official dataset [toolkit](https://github.com/bdd100k/bdd100k) with commit `b7e1781` for data format convertion and result evaluation. Install `scalabel=0.3.0` required by the toolkit. We need to modify source files of the two packages.

## Label format convertion
- Related files: [to_coco.py](bdd100k/label/to_coco.py).
- Usage: For bdd100k package, replace `bdd100k/bdd100k/label/to_coco.py` with [to_coco.py](bdd100k/label/to_coco.py).
- Description: Modify the `file_name` value of the `images` dict of the converted COCO label, and other small adjustments for usage convenience.
## MOTS evaluation
- Related files: [run.py](bdd100k/eval/run.py), [mot.py](scalabel/eval/mot.py), [mots.py](scalabel/eval/mots.py).
- Usage: For bdd100k package, replace `bdd100k/bdd100k/eval/run.py` with [run.py](bdd100k/eval/run.py). For scalable package, replace `scalabel/eval/mot.py` with [mot.py](scalabel/eval/mot.py), and replace `scalabel/eval/mots.py` with [mots.py](scalabel/eval/mots.py).
- Description: The official BDD Toolkit does not compute the `sMOTSA` metric. Therefore we add the computation of this metric to the toolkit, referring to its definition in this [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Voigtlaender_MOTS_Multi-Object_Tracking_and_Segmentation_CVPR_2019_paper.html).
