import os
import cv2
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from detectron2.engine import default_argument_parser
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou

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
    
    args.add_argument("--dataset-dir", type=str,
                      default="temp",
                      help="path to dataset directory")

    args.add_argument(
        '--outputdir',
        type=str,
        default='../output'
    )

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

def cumTpFp(gtRects, detRects, scores, label, overlapRatio):
    # det_state: [label, score, tp, fp], tp, fp = 0 or 1

    # gtRect: xmin, ymin, xmax, ymax
    det_state = [(label, 0., 0, 1)] * len(detRects)
    iou_max = 0
    maxIndex = -1
    blockIdx = -1
    for cnt in range(len(det_state)):
        det_state[cnt] = (label, scores[cnt], 0, 1) 
    visited = [0] * len(gtRects)
    if len(detRects) != len(scores):
        print("Num of scores does not match detection results!")
    if len(detRects) == 0:
        return det_state
    detRects[:, 2] = detRects[:, 0] + detRects[:, 2]
    detRects[:, 3] = detRects[:, 1] + detRects[:, 3]
    gtRects[:, 2] = gtRects[:, 0] + gtRects[:, 2]
    gtRects[:, 3] = gtRects[:, 1] + gtRects[:, 3]
    iou_matrix = pairwise_iou(Boxes(gtRects.cuda()), Boxes(detRects.cuda())).cpu()
    for indexDet, deti in enumerate(detRects):
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for indexGt, gti in enumerate(gtRects):
            iou = iou_matrix[indexGt][indexDet].item()
            if iou > iou_max:
                iou_max = iou
                maxIndex = indexDet
                blockIdx = indexGt
        if iou_max >= overlapRatio and visited[blockIdx] == 0:
            det_state[maxIndex] = (label, scores[indexDet], 1, 0)
            visited[blockIdx] = 1
    return det_state, iou_matrix

class Eval:
    def __init__(self):
        self.iou_threshold = 0.5
        self.args = set_up_parse()
        modelname = (self.args.config_file).split("/")[2].split(".")[0]

        self.test_IDdataset = "voc_completely_annotation_pretest"
        self.gt_IDcoco_api = COCO(self.args.dataset_dir + "/voc0712_train_completely_annotation200.json")
        
        self.cfg_OUTPUT_DIR = "../data/VOC-Detection/faster-rcnn/{}/random_seed_0".format(modelname)
        ID_inference_output_dir = get_inference_output_dir(
            self.cfg_OUTPUT_DIR,
            self.test_IDdataset,
            self.args.inference_config,
            self.args.image_corruption_level)
        print(ID_inference_output_dir)
        ID_prediction_file_name = os.path.join(
            ID_inference_output_dir,
            'voc_instances_results_pretest.json')
        self.ID_res_coco_api = self.gt_IDcoco_api.loadRes(ID_prediction_file_name)
        

    def scanVOC(self, gt_coco_api, res_coco_api):
        energy_container = []
        imgidlist = list(gt_coco_api.imgs.keys())
        for imgID in imgidlist:
            res_list_this_img = res_coco_api.imgToAnns[imgID]
            res_boxes = np.array([resbox_i['bbox'] for resbox_i in res_list_this_img])
            res_score = np.array([resbox_i['score'] for resbox_i in res_list_this_img])

            gt_list_this_img = gt_coco_api.imgToAnns[imgID]
            gt_boxes = np.array([gti['bbox'] for gti in gt_list_this_img])
            if len(gt_boxes) == 0:
                print("There are no gtboxes in {}\n".format(gt_coco_api.imgs[imgID]['file_name']))
                continue
            if len(res_boxes) == 0:
                continue
            
            det_state, iou_matrix = cumTpFp(torch.Tensor(gt_boxes), torch.Tensor(res_boxes), res_score, 0, self.iou_threshold)
            for i in range(len(det_state)):
                inter_feat = res_list_this_img[i]["inter_feat"]
                if det_state[i][2] == 1:
                    energy_i = torch.logsumexp(torch.Tensor(inter_feat[:-1]), dim=0).item()
                    energy_container.append(energy_i)
                    
        return energy_container

    def thresh(self, data):
        data = np.array(data)
        # x, y(ratio)
        datamax = max(data)
        print("max energy: {}".format(datamax))
        print("mean energy: {}".format(data.mean()))

        datamin = min(data)
        stride = 1e-3
        deltamin = 100
        recall95_thresh = 0
        for thresh in np.arange(datamin, datamax, stride):
            y = np.where(data >= thresh)[0].shape[0] / data.shape[0]
            delta = abs(y - 0.95)
            if delta < deltamin:
                deltamin = delta
                recall95_thresh = thresh
        print("thresh: {}".format(recall95_thresh))

        print("\n")
        print("The threshold gamma is written in " + self.cfg_OUTPUT_DIR + "/inference/energy_threshold.txt")
        with open(os.path.join(self.cfg_OUTPUT_DIR, 'inference', "energy_threshold.txt"), "w") as f:
            f.write("{}\n".format(recall95_thresh))


    def run(self):
        ID_energy = self.scanVOC(self.gt_IDcoco_api, self.ID_res_coco_api)
        self.thresh(ID_energy)

eva = Eval()
eva.run()