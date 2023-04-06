import os
import cv2
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from detectron2.engine import default_argument_parser

import sys
sys.path.append('../')
from evaluator.voc_eval_offical import voc_evaluate

def get_inference_output_dir(output_dir_name,
                             test_dataset_name,
                             inference_config_name,
                             image_corruption_level):
    return os.path.join(
        output_dir_name,
        'inference',
        test_dataset_name,
        os.path.split(inference_config_name)[-1][:-5],
        "corruption_level_" + str(image_corruption_level))

def set_up_parse():
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args.add_argument("--dataset-dir", type=str,
                      default="temp",
                      help="path to dataset directory")
    args.add_argument("--test-dataset", type=str,
                      default="",
                      help="test dataset")
    
    args.add_argument(
        '--outputdir',
        type=str,
        default='../output'
    )

    args.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed to be used for all scientific computing libraries")

    # Inference arguments, will not be used during training.
    args.add_argument(
        "--inference-config",
        type=str,
        default="",
        help="Inference parameter: Path to the inference config, which is different from training config. Check readme for more information.")
    args.add_argument(
        "--image-corruption-level",
        type=int,
        default=0,
        help="Inference parameter: Image corruption level between 0-5. Default is no corruption, level 0.")

    return args.parse_args()



class Eval:
    def __init__(self):
        self.args = set_up_parse()
        modelname = (self.args.config_file).split("/")[2].split(".")[0]
        cfg_OUTPUT_DIR = "../data/VOC-Detection/faster-rcnn/{}/random_seed_0".format(modelname)

        # load groundtruth
        self.img_path = self.args.dataset_dir + "/val2017/"
        self.test_OODdataset = self.args.test_dataset  
        if self.test_OODdataset == "coco_extended_ood_val":
            self.gt_OODcoco_api = COCO(self.args.dataset_dir + "/annotations/instances_val2017_extended_ood.json")
        elif self.test_OODdataset == "coco_mixed_val":
            self.gt_OODcoco_api = COCO(self.args.dataset_dir + "/annotations/instances_val2017_mixed_OOD.json")
            self.gt_IDcoco_api = COCO(self.args.dataset_dir + "/annotations/instances_val2017_mixed_ID.json")
        # load prediction
        OOD_inference_output_dir = get_inference_output_dir(
            cfg_OUTPUT_DIR,
            self.test_OODdataset,
            self.args.inference_config,
            self.args.image_corruption_level)
        OOD_prediction_file_name = os.path.join(
            OOD_inference_output_dir,
            'coco_instances_results_idood.json')
        self.OOD_res_coco_api = self.gt_OODcoco_api.loadRes(OOD_prediction_file_name)
        imgidlist = list(self.gt_OODcoco_api.imgs.keys())
        self.imgidlist = []
        for imgID in imgidlist:
            gt_list_this_img = self.gt_OODcoco_api.loadAnns(self.gt_OODcoco_api.getAnnIds(imgIds=imgID))
            if len(gt_list_this_img) == 0:
                continue
            self.imgidlist.append(imgID)
        print("img num: {}\n".format(len(self.imgidlist)))

    def readdata(self, gt_coco_api, res_coco_api):
        self.res = {81: {}}
        self.OOD_gt = {}
        for imgID in self.imgidlist:
            # read prediction
            res_list_this_img = res_coco_api.loadAnns(res_coco_api.getAnnIds(imgIds=[imgID]))
            res_list_this_img = [res for res in res_list_this_img if res["category_id"]==81]  # only read OOD
            if len(res_list_this_img)==0:
                img_res = np.array([])
            else:
                # xyhw -> xyxy
                img_res = np.array([res['bbox'] + [res[self.sort_scores_name]]  for res in res_list_this_img])
                img_res[:, 2] = img_res[:, 0] + img_res[:, 2]
                img_res[:, 3] = img_res[:, 1] + img_res[:, 3]
            self.res[81].update({imgID: img_res})
            # read groundtruth
            gt_list_this_img = gt_coco_api.loadAnns(gt_coco_api.getAnnIds(imgIds=[imgID]))
            img_gt = np.array([gti['bbox'] + [81] for gti in gt_list_this_img])
            img_gt[:, 2] = img_gt[:, 0] + img_gt[:, 2]
            img_gt[:, 3] = img_gt[:, 1] + img_gt[:, 3]
            self.OOD_gt.update({imgID: img_gt})

    def run(self):
        self.sort_scores_name = "complete_scores"
        self.readdata(self.gt_OODcoco_api, self.OOD_res_coco_api)
        recall, precision, ap, rec, prec, state, det_image_files = voc_evaluate(self.res, self.OOD_gt, 81)
        print("Ap: ", ap)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", 2 * (precision * recall) / (precision + recall))


eva = Eval()
eva.run()