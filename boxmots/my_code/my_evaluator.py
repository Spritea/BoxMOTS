from detectron2.evaluation.evaluator import DatasetEvaluator
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist

class ReIDInfer(DatasetEvaluator):
    """
    This is for ReID inference. 
    The ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        reid_feats=outputs[1]['reid_feats']
        instances=outputs[0]['instances']
        # for ReID feat, need to get predicted boxes corrsponding to original img size.
        for inst_count in range(len(reid_feats)):
            assert len(reid_feats)==len(instances.pred_boxes),\
                "#reid_feats must be equal to #pred_boxes"
            pred_box_ori_size=instances.pred_boxes[inst_count]
            box_score=instances.scores[inst_count]
            pred_class=instances.pred_classes[inst_count]
            reid_feat=reid_feats[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
            inst_one["box_score"]=box_score.cpu().numpy()
            inst_one["pred_class"]=pred_class.cpu().numpy()
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions.append(inst_one)
            
    def evaluate(self):
        """
        This simply save the ReID feat of gt boxes to a disk file.
        """
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_infer_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions, f)
        # below for following compatible.
        return None


class ReIDEval(DatasetEvaluator):
    """
    This is for ReID evaluation. The input is gt boxes in val set.
    Note for inference in practice, the ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=outputs[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions.append(inst_one)
            
    def evaluate(self):
        """
        This simply save the ReID feat of gt boxes to a disk file.
        """
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions, f)
        # below for following compatible.
        return {}

class ReIDEvalInTrain(DatasetEvaluator):
    """
    This is for ReID evaluation. The input is gt boxes in val set.
    Note for inference in practice, the ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        reid_feats=outputs[1]['reid_feats']
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=reid_feats[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions.append(inst_one)
            
    def evaluate(self):
        """
        This simply save the ReID feat of gt boxes to a disk file.
        """
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions, f)
        # below compute top-n acc.
        out_all=self._predictions
        track_id_np=np.array([x["track_id"] for x in out_all])
        # below get reid feat only.
        reid_feat_np=np.array([x["reid_feat"] for x in out_all])
        # below del nan.
        no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
        reid_feat_np=reid_feat_np[no_nan_id,:]
        track_id_np=track_id_np[no_nan_id]
        
        USE_NORM=False
        if not USE_NORM:
            dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
            dist_id=np.argsort(dist_mat,axis=1)
        else:
            # [:,None] is to expand norm dimension for division.
            reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
            dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
            dist_id=np.argsort(dist_mat_norm,axis=1)
        # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
        track_id_sort_all=[]
        for inst_one_id in dist_id:
            track_id_sort_one=track_id_np[inst_one_id]
            track_id_sort_all.append(track_id_sort_one)
        
        # below get top-n acc.
        top_n=1
        track_id_np=np.array(track_id_sort_all)
        target=track_id_np[:,0]
        obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
        # below check whether any object only appears one time.
        # if so, they line should be removed from track_id_np.
        if obj_min_appear>1:
            print("Objects at least appear {:} time".format(obj_min_appear))
        if top_n==1:
            track_id_top_n=track_id_np[:,1]
            correct_count=np.count_nonzero((track_id_top_n==target))
        else:
            track_id_top_n=track_id_np[:,1:top_n+1]
            target=np.reshape(target,(len(target),-1))
            correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))
            
        top_n_acc=correct_count/len(track_id_np)
        return {f"Top {top_n} Acc":top_n_acc}

class ReIDInferAndEvalInTrain(DatasetEvaluator):
    """
    This is for ReID inference. 
    The ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions_reid_infer = []
        self._predictions_reid_eval = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        # below for reid infer pred.
        # reid_feats=outputs[1]['reid_feats']
        reid_feats_infer=outputs[1]['reid_infer_pred']['reid_feats']
        instances=outputs[0]['instances']
        # for ReID feat, need to get predicted boxes corrsponding to original img size.
        for inst_count in range(len(reid_feats_infer)):
            assert len(reid_feats_infer)==len(instances.pred_boxes),\
                "#reid_feats must be equal to #pred_boxes"
            pred_box_ori_size=instances.pred_boxes[inst_count]
            box_score=instances.scores[inst_count]
            pred_class=instances.pred_classes[inst_count]
            reid_feat=reid_feats_infer[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
            inst_one["box_score"]=box_score.cpu().numpy()
            inst_one["pred_class"]=pred_class.cpu().numpy()
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_infer.append(inst_one)

        # below for reid eval gt.
        reid_feats_eval=outputs[1]['reid_eval_gt']
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=reid_feats_eval[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_eval.append(inst_one)

    def evaluate(self):
        # below for reid infer pred.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_infer_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_infer, f)
                
        # below for reid eval gt.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_eval, f)
                
        def get_pkl_and_del_nan_by_category(out_all,category='both'):
            cat_dict={'car':0,'ped':1}
            # with open(file_name,'rb') as f:
            #     out_all=pickle.load(f)
            if category!='both':
                out_all=list(filter(lambda x:x['gt_class']==cat_dict[category],out_all))
            track_id_np=np.array([x["track_id"] for x in out_all])
            # below get reid feat only.
            reid_feat_np=np.array([x["reid_feat"] for x in out_all])
            # below del nan.
            no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
            reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
            track_id_np_no_nan=track_id_np[no_nan_id]
            return track_id_np_no_nan,reid_feat_np_no_nan

        def run_reid_eval(track_id_np,reid_feat_np,top_n):
            USE_NORM=False
            if not USE_NORM:
                dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
                dist_id=np.argsort(dist_mat,axis=1)
            else:
                # [:,None] is to expand norm dimension for division.
                reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
                dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
                dist_id=np.argsort(dist_mat_norm,axis=1)
            # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
            track_id_sort_all=[]
            for inst_one_id in dist_id:
                track_id_sort_one=track_id_np[inst_one_id]
                track_id_sort_all.append(track_id_sort_one)
            
            # below get top-n acc.
            # top_n=1
            track_id_np=np.array(track_id_sort_all)
            target=track_id_np[:,0]
            obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
            # below check whether any object only appears one time.
            # if so, they line should be removed from track_id_np.
            if obj_min_appear>1:
                print("Objects at least appear {:} time".format(obj_min_appear))
            if top_n==1:
                track_id_top_n=track_id_np[:,1]
                correct_count=np.count_nonzero((track_id_top_n==target))
            else:
                track_id_top_n=track_id_np[:,1:top_n+1]
                target=np.reshape(target,(len(target),-1))
                correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))
            top_n_acc=correct_count/len(track_id_np)
            return top_n_acc
        
        # below compute top-n acc.
        top_n=1
        out_all=self._predictions_reid_eval
        track_id_np,reid_feat_np=get_pkl_and_del_nan_by_category(out_all)
        top_n_acc_all=run_reid_eval(track_id_np,reid_feat_np,top_n)
        track_id_np_car,reid_feat_np_car=get_pkl_and_del_nan_by_category(out_all,category='car')
        top_n_acc_car=run_reid_eval(track_id_np_car,reid_feat_np_car,top_n)
        track_id_np_ped,reid_feat_np_ped=get_pkl_and_del_nan_by_category(out_all,category='ped')
        top_n_acc_ped=run_reid_eval(track_id_np_ped,reid_feat_np_ped,top_n)

        return {'ReID Metrics':{f"Top {top_n} Acc All":top_n_acc_all,f"Top {top_n} Acc Car":top_n_acc_car,
                f"Top {top_n} Acc Ped":top_n_acc_ped}}

class BDDReIDInferAndEvalInTrain(DatasetEvaluator):
    """
    This is for ReID inference. 
    The ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions_reid_infer = []
        self._predictions_reid_eval = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        # below for reid infer pred.
        # reid_feats=outputs[1]['reid_feats']
        reid_feats_infer=outputs[1]['reid_infer_pred']['reid_feats']
        instances=outputs[0]['instances']
        # for ReID feat, need to get predicted boxes corrsponding to original img size.
        for inst_count in range(len(reid_feats_infer)):
            assert len(reid_feats_infer)==len(instances.pred_boxes),\
                "#reid_feats must be equal to #pred_boxes"
            pred_box_ori_size=instances.pred_boxes[inst_count]
            box_score=instances.scores[inst_count]
            pred_class=instances.pred_classes[inst_count]
            reid_feat=reid_feats_infer[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
            inst_one["box_score"]=box_score.cpu().numpy()
            inst_one["pred_class"]=pred_class.cpu().numpy()
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_infer.append(inst_one)

        # below for reid eval gt.
        reid_feats_eval=outputs[1]['reid_eval_gt']
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=reid_feats_eval[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_eval.append(inst_one)

    def evaluate(self):
        # below for reid infer pred.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_infer_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_infer, f)
                
        # below for reid eval gt.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_eval, f)
                
        def get_pkl_and_del_nan_by_category(out_all,category='all'):
            # with open(file_name,'rb') as f:
            #     out_all=pickle.load(f)
            if category!='all':
                out_all=list(filter(lambda x:x['gt_class']==cat_dict[category],out_all))
            track_id_np=np.array([x["track_id"] for x in out_all])
            # below get reid feat only.
            reid_feat_np=np.array([x["reid_feat"] for x in out_all])
            # below del nan.
            no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
            reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
            track_id_np_no_nan=track_id_np[no_nan_id]
            return track_id_np_no_nan,reid_feat_np_no_nan

        def run_reid_eval(track_id_np,reid_feat_np,top_n):
            USE_NORM=False
            if not USE_NORM:
                dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
                dist_id=np.argsort(dist_mat,axis=1)
            else:
                # [:,None] is to expand norm dimension for division.
                reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
                dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
                dist_id=np.argsort(dist_mat_norm,axis=1)
            # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
            track_id_sort_all=[]
            for inst_one_id in dist_id:
                track_id_sort_one=track_id_np[inst_one_id]
                track_id_sort_all.append(track_id_sort_one)
            
            # below get top-n acc.
            # top_n=1
            track_id_np=np.array(track_id_sort_all)
            target=track_id_np[:,0]
            # don't use instance that only appears 1 time as query image.
            track_id_np=track_id_np[np.count_nonzero(track_id_np==(target[:,None]),axis=1)>1]
            target=track_id_np[:,0]
            obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
            # below check whether any object only appears one time.
            # if so, they line should be removed from track_id_np.
            if obj_min_appear>1:
                print("Objects at least appear {:} time".format(obj_min_appear))
            if top_n==1:
                track_id_top_n=track_id_np[:,1]
                correct_count=np.count_nonzero((track_id_top_n==target))
            else:
                track_id_top_n=track_id_np[:,1:top_n+1]
                target=np.reshape(target,(len(target),-1))
                correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))
            top_n_acc=correct_count/len(track_id_np)
            return top_n_acc
        
        # below compute top-n acc.
        top_n=1
        out_all=self._predictions_reid_eval
        cat_dict={'pedestrian':0,'rider':1,'car':2,'truck':3,
                  'bus':4,'motorcycle':6,'bicycle':7}
        acc_list=[]
        for cat in cat_dict.keys():
            track_id_np_cat,reid_feat_np_cat=get_pkl_and_del_nan_by_category(out_all,category=cat)
            acc=run_reid_eval(track_id_np_cat,reid_feat_np_cat,top_n)
            acc_list.append(acc)
        acc_mean=np.mean(acc_list)
        acc_by_cat_dict={}
        for ctr,k in enumerate(cat_dict.keys()):
            acc_by_cat_dict[f"Top {top_n} Acc "+k]=acc_list[ctr]
        acc_by_cat_dict[f"Top {top_n} Mean Acc by class"]=acc_mean
        # return {'ReID Metrics':{f"Top {top_n} Mean Acc by class":acc_mean,f"Top {top_n} Acc Pedestrian":acc_list[0],
        #         f"Top {top_n} Acc Rider":acc_list[1],f"Top {top_n} Acc Car":acc_list[2],
        #         f"Top {top_n} Acc Truck":acc_list[3],f"Top {top_n} Acc Bus":acc_list[4],
        #         f"Top {top_n} Acc Motorcycle":acc_list[5],f"Top {top_n} Acc Bicycle":acc_list[6]}}
        return {'ReID Metrics':acc_by_cat_dict}

class BDDReIDInferAndEvalInTrainClassSeven(DatasetEvaluator):
    """
    This is for ReID inference. 
    The ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions_reid_infer = []
        self._predictions_reid_eval = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        # below for reid infer pred.
        # reid_feats=outputs[1]['reid_feats']
        reid_feats_infer=outputs[1]['reid_infer_pred']['reid_feats']
        instances=outputs[0]['instances']
        # for ReID feat, need to get predicted boxes corrsponding to original img size.
        for inst_count in range(len(reid_feats_infer)):
            if len(reid_feats_infer)!=len(instances.pred_boxes):
                # TODO: here could raise error on BDD data, need to further check.
                print('kk')
            assert len(reid_feats_infer)==len(instances.pred_boxes),\
                "#reid_feats must be equal to #pred_boxes"
            pred_box_ori_size=instances.pred_boxes[inst_count]
            box_score=instances.scores[inst_count]
            pred_class=instances.pred_classes[inst_count]
            reid_feat=reid_feats_infer[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
            inst_one["box_score"]=box_score.cpu().numpy()
            inst_one["pred_class"]=pred_class.cpu().numpy()
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_infer.append(inst_one)

        # below for reid eval gt.
        reid_feats_eval=outputs[1]['reid_eval_gt']
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=reid_feats_eval[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_eval.append(inst_one)

    def evaluate(self):
        # below for reid infer pred.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_infer_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_infer, f)
                
        # below for reid eval gt.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_eval, f)
                
        def get_pkl_and_del_nan_by_category(out_all,category='all'):
            # with open(file_name,'rb') as f:
            #     out_all=pickle.load(f)
            if category!='all':
                out_all=list(filter(lambda x:x['gt_class']==cat_dict[category],out_all))
            track_id_np=np.array([x["track_id"] for x in out_all])
            # below get reid feat only.
            reid_feat_np=np.array([x["reid_feat"] for x in out_all])
            # below del nan.
            no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
            reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
            track_id_np_no_nan=track_id_np[no_nan_id]
            return track_id_np_no_nan,reid_feat_np_no_nan

        def run_reid_eval(track_id_np,reid_feat_np,top_n):
            USE_NORM=False
            if not USE_NORM:
                dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
                dist_id=np.argsort(dist_mat,axis=1)
            else:
                # [:,None] is to expand norm dimension for division.
                reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
                dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
                dist_id=np.argsort(dist_mat_norm,axis=1)
            # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
            track_id_sort_all=[]
            for inst_one_id in dist_id:
                track_id_sort_one=track_id_np[inst_one_id]
                track_id_sort_all.append(track_id_sort_one)
            
            # below get top-n acc.
            # top_n=1
            track_id_np=np.array(track_id_sort_all)
            target=track_id_np[:,0]
            # don't use instance that only appears 1 time as query image.
            track_id_np=track_id_np[np.count_nonzero(track_id_np==(target[:,None]),axis=1)>1]
            target=track_id_np[:,0]
            obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
            # below check whether any object only appears one time.
            # if so, they line should be removed from track_id_np.
            if obj_min_appear>1:
                print("Objects at least appear {:} time".format(obj_min_appear))
            if top_n==1:
                track_id_top_n=track_id_np[:,1]
                correct_count=np.count_nonzero((track_id_top_n==target))
            else:
                track_id_top_n=track_id_np[:,1:top_n+1]
                target=np.reshape(target,(len(target),-1))
                correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))
            top_n_acc=correct_count/len(track_id_np)
            return top_n_acc
        
        # below compute top-n acc.
        top_n=1
        out_all=self._predictions_reid_eval
        # since detectron2 applies mapping, need to change the cat_dict.
        # cat_dict={'pedestrian':0,'rider':1,'car':2,'truck':3,
        #           'bus':4,'motorcycle':6,'bicycle':7}
        cat_dict={'pedestrian':0,'rider':1,'car':2,'truck':3,
                  'bus':4,'motorcycle':5,'bicycle':6}
        acc_list=[]
        for cat in cat_dict.keys():
            track_id_np_cat,reid_feat_np_cat=get_pkl_and_del_nan_by_category(out_all,category=cat)
            acc=run_reid_eval(track_id_np_cat,reid_feat_np_cat,top_n)
            acc_list.append(acc)
        acc_mean=np.mean(acc_list)
        acc_by_cat_dict={}
        for ctr,k in enumerate(cat_dict.keys()):
            acc_by_cat_dict[f"Top {top_n} Acc "+k]=acc_list[ctr]
        acc_by_cat_dict[f"Top {top_n} Mean Acc by class"]=acc_mean
        # return {'ReID Metrics':{f"Top {top_n} Mean Acc by class":acc_mean,f"Top {top_n} Acc Pedestrian":acc_list[0],
        #         f"Top {top_n} Acc Rider":acc_list[1],f"Top {top_n} Acc Car":acc_list[2],
        #         f"Top {top_n} Acc Truck":acc_list[3],f"Top {top_n} Acc Bus":acc_list[4],
        #         f"Top {top_n} Acc Motorcycle":acc_list[5],f"Top {top_n} Acc Bicycle":acc_list[6]}}
        return {'ReID Metrics':acc_by_cat_dict}

class YTVIS2019ReIDInferAndEvalInTrain(DatasetEvaluator):
    """
    This is for ReID inference. 
    The ReID branch input should be the box output of detection branch.
    """
    def __init__(self,output_dir) -> None:
        self._output_dir = output_dir
    def reset(self):
        self._predictions_reid_infer = []
        self._predictions_reid_eval = []
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        input_data=inputs[0]
        img_info={}
        img_info["file_name"]=input_data["file_name"]
        img_info['height']=input_data["height"]
        img_info["width"]=input_data["width"]
        img_info["image_id"]=input_data["image_id"]
        # below for reid infer pred.
        # reid_feats=outputs[1]['reid_feats']
        reid_feats_infer=outputs[1]['reid_infer_pred']['reid_feats']
        instances=outputs[0]['instances']
        # for ReID feat, need to get predicted boxes corrsponding to original img size.
        for inst_count in range(len(reid_feats_infer)):
            assert len(reid_feats_infer)==len(instances.pred_boxes),\
                "#reid_feats must be equal to #pred_boxes"
            pred_box_ori_size=instances.pred_boxes[inst_count]
            box_score=instances.scores[inst_count]
            pred_class=instances.pred_classes[inst_count]
            reid_feat=reid_feats_infer[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["pred_box_ori_size"]=pred_box_ori_size.tensor.cpu().numpy()
            inst_one["box_score"]=box_score.cpu().numpy()
            inst_one["pred_class"]=pred_class.cpu().numpy()
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_infer.append(inst_one)

        # below for reid eval gt.
        reid_feats_eval=outputs[1]['reid_eval_gt']
        for inst_count in range(len(input_data["instances"].gt_classes)):
            gt_class_one=input_data["instances"].gt_classes[inst_count]
            track_id_one=input_data["instances"].track_ids[inst_count]
            reid_feat=reid_feats_eval[inst_count]
            inst_one={}
            inst_one["img_info"]=img_info
            inst_one["gt_class"]=int(gt_class_one.cpu().numpy())
            inst_one["track_id"]=int(track_id_one.cpu().numpy())
            inst_one["reid_feat"]=reid_feat.cpu().numpy()
            self._predictions_reid_eval.append(inst_one)

    def evaluate(self):
        # below for reid infer pred.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_infer_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_infer, f)
                
        # below for reid eval gt.
        if self._output_dir:
            os.makedirs(self._output_dir,exist_ok=True)
            file_path = os.path.join(self._output_dir, "reid_eval_out.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self._predictions_reid_eval, f)
                
        def get_pkl_and_del_nan_by_category(out_all,category='all'):
            # with open(file_name,'rb') as f:
            #     out_all=pickle.load(f)
            if category!='all':
                out_all=list(filter(lambda x:x['gt_class']==cat_dict[category],out_all))
            track_id_np=np.array([x["track_id"] for x in out_all])
            # below get reid feat only.
            reid_feat_np=np.array([x["reid_feat"] for x in out_all])
            # below del nan.
            no_nan_id=np.unique(np.where(~np.isnan(reid_feat_np))[0])
            reid_feat_np_no_nan=reid_feat_np[no_nan_id,:]
            track_id_np_no_nan=track_id_np[no_nan_id]
            return track_id_np_no_nan,reid_feat_np_no_nan

        def run_reid_eval(track_id_np,reid_feat_np,top_n):
            USE_NORM=False
            if not USE_NORM:
                dist_mat=cdist(reid_feat_np,reid_feat_np,metric='euclidean')
                dist_id=np.argsort(dist_mat,axis=1)
            else:
                # [:,None] is to expand norm dimension for division.
                reid_feat_norm=reid_feat_np/np.linalg.norm((reid_feat_np),axis=1)[:,None]
                dist_mat_norm=cdist(reid_feat_norm,reid_feat_norm,metric='euclidean')
                dist_id=np.argsort(dist_mat_norm,axis=1)
            # track_id_sort=np.take_along_axis(track_id_np,dist_id,axis=1)
            track_id_sort_all=[]
            for inst_one_id in dist_id:
                track_id_sort_one=track_id_np[inst_one_id]
                track_id_sort_all.append(track_id_sort_one)
            
            # below get top-n acc.
            # top_n=1
            track_id_np=np.array(track_id_sort_all)
            target=track_id_np[:,0]
            # don't use instance that only appears 1 time as query image.
            track_id_np=track_id_np[np.count_nonzero(track_id_np==(target[:,None]),axis=1)>1]
            target=track_id_np[:,0]
            obj_min_appear=min(np.count_nonzero(track_id_np==(target[:,None]),axis=1))
            # below check whether any object only appears one time.
            # if so, they line should be removed from track_id_np.
            if obj_min_appear>1:
                print("Objects at least appear {:} time".format(obj_min_appear))
            if top_n==1:
                track_id_top_n=track_id_np[:,1]
                correct_count=np.count_nonzero((track_id_top_n==target))
            else:
                track_id_top_n=track_id_np[:,1:top_n+1]
                target=np.reshape(target,(len(target),-1))
                correct_count=np.count_nonzero((track_id_top_n==target).any(axis=1))
            top_n_acc=correct_count/len(track_id_np)
            return top_n_acc
        
        # below compute top-n acc.
        top_n=1
        out_all=self._predictions_reid_eval
        # cat_dict={'pedestrian':0,'rider':1,'car':2,'truck':3,
        #           'bus':4,'motorcycle':6,'bicycle':7}
        cat_dict = {"person":0, "giant_panda":1, "lizard":2, "parrot":3, "skateboard":4,
                    "sedan":5, "ape":6, "dog":7, "snake":8, "monkey":9, 
                    "hand":10, "rabbit":11, "duck":12, "cat":13, "cow":14,
                    "fish":15, "train":16, "horse":17, "turtle":18, "bear":19, 
                    "motorbike":20, "giraffe":21, "leopard":22, "fox":23, "deer":24,
                    "owl":25, "surfboard":26, "airplane":27, "truck":28, "zebra":29,
                    "tiger":30, "elephant":31, "snowboard":32, "boat":33, "shark":34,
                    "mouse":35, "frog":36, "eagle":37, "earless_seal":38, "tennis_racket":39}
        acc_list=[]
        for cat in cat_dict.keys():
            track_id_np_cat,reid_feat_np_cat=get_pkl_and_del_nan_by_category(out_all,category=cat)
            acc=run_reid_eval(track_id_np_cat,reid_feat_np_cat,top_n)
            acc_list.append(acc)
        acc_mean=np.mean(acc_list)
        acc_by_cat_dict={}
        for ctr,k in enumerate(cat_dict.keys()):
            acc_by_cat_dict[f"Top {top_n} Acc "+k]=acc_list[ctr]
        acc_by_cat_dict[f"Top {top_n} Mean Acc by class"]=acc_mean
        # return {'ReID Metrics':{f"Top {top_n} Mean Acc by class":acc_mean,f"Top {top_n} Acc Pedestrian":acc_list[0],
        #         f"Top {top_n} Acc Rider":acc_list[1],f"Top {top_n} Acc Car":acc_list[2],
        #         f"Top {top_n} Acc Truck":acc_list[3],f"Top {top_n} Acc Bus":acc_list[4],
        #         f"Top {top_n} Acc Motorcycle":acc_list[5],f"Top {top_n} Acc Bicycle":acc_list[6]}}
        return {'ReID Metrics':acc_by_cat_dict}
