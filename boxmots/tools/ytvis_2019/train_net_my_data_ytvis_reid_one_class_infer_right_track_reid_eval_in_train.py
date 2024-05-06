# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader,build_detection_train_loader_no_shuffle,build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import YTVIS2019DatasetMapperWithBasisReID, YTVIS2019DatasetMapperWithBasisReIDEval
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
)
from my_code.my_evaluator import YTVIS2019ReIDInferAndEvalInTrain
# below by cws.
# for build my own dataset, the dataset is already in coco format.
from detectron2.data.datasets import bdd_register_coco_instances_reid

ytvis_2019_classes = ["person", "giant_panda", "lizard", "parrot", "skateboard",
                      "sedan", "ape", "dog", "snake", "monkey", 
                      "hand", "rabbit", "duck", "cat", "cow",
                      "fish", "train", "horse", "turtle", "bear", 
                      "motorbike", "giraffe", "leopard", "fox", "deer",
                      "owl", "surfboard", "airplane", "truck", "zebra",
                      "tiger", "elephant", "snowboard", "boat", "shark",
                      "mouse", "frog", "eagle", "earless_seal", "tennis_racket"]
bdd_register_coco_instances_reid("ytvis_2019_train", {"thing_classes" : ytvis_2019_classes}, "my_dataset/youtube_vis_2019/label_coco_format/train_coco_format.json", "my_dataset/youtube_vis_2019/train/JPEGImages/")
bdd_register_coco_instances_reid("ytvis_2019_val", {"thing_classes" : ytvis_2019_classes}, "my_dataset/youtube_vis_2019/label_coco_format/val_from_train_coco_format.json", "my_dataset/youtube_vis_2019/train/JPEGImages/")
a=MetadataCatalog.get("ytvis_2019_train")


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    # add by cws.
    # to get iter count as class variable.
    my_iter_count=0
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                Trainer.my_iter_count=self.iter
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = YTVIS2019DatasetMapperWithBasisReID(cfg, True)
        # return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader_no_shuffle(cfg, mapper=mapper)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        # add coco evaluator.
        evaluator_list=[]
        output_folder_coco = os.path.join(cfg.OUTPUT_DIR, "inference","iter_"+str(cls.my_iter_count).zfill(7))
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder_coco))
        # below is ReID Infer and Eval evaluator.
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "reid_infer_and_eval_out","iter_"+str(cls.my_iter_count).zfill(7))
        reid_evaluator=YTVIS2019ReIDInferAndEvalInTrain(output_dir=output_folder)
        evaluator_list.append(reid_evaluator)
        
        return DatasetEvaluators(evaluator_list)
        
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # by cws.
    # for ReID eval, use DatasetMapperWithBasisReIDEval to get gt boxes on val set.
    # for ReID infer, use default build_test_loader without DatasetMapperWithBasisReIDEval.
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = YTVIS2019DatasetMapperWithBasisReIDEval(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name,mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
