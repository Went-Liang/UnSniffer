import os
import cv2
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from pycocotools.coco import COCO
from detectron2.engine import default_argument_parser
from voc_eval_offical import voc_ap

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

# We refer to OWOD's implementation of WI 
# https://github.com/JosephKJ/OWOD/blob/9b56fdf1d37c15109f17be36806b72e565ec0647/detectron2/evaluation/pascal_voc_evaluation.py

def voc_eval(detections, annotations, classname, ovthresh=0.5, use_07_metric=True, known_classes=None):
    # detections {81: [np, np, np...]}, np = np.array([[x1,y1,x2,y2,score], [...],...])
    # annotations [np, np, np], np = np.array([[x1,y1,x2,y2,class_id], [...],...])
    # classname = 81
    # extract ground truth objects from the annotations for this class
    class_gt_bboxes = {}
    npos = 0  # number of ground truth bboxes having label classname
    # annotations keyed on image file names or paths or anything that is unique for each image
    for image_name in annotations:
        # for each image list of objects: [[x1,y1, x2,y2, classname], [], ...]
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == classname]
        bbox = np.array(R)
        # difficult is not stored: take it as 0/false
        difficult = np.array([0] * len(R)).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # detections' image file names/paths
    det_image_files = []
    confidences = []
    det_bboxes = []
    # detections should be keyed on class_id (classname)
    class_dict = detections[classname]
    for image_file in class_dict:
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = np.array(det_bboxes)
    confidences = np.array(confidences)

    # number of detections
    num_dets = len(det_image_files)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)

    if det_bboxes.shape[0] == 0:
        return 0., 0., 0.

    # sort detections by confidence
    sorted_ind = np.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(num_dets):
        if det_image_files[d] not in class_gt_bboxes:
            continue
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ## compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    state = [copy.deepcopy(tp), copy.deepcopy(fp)]
    # compute precision recall
    stp = sum(tp)
    recall = stp / npos
    precision = stp / (stp + sum(fp))

    # compute average precision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for image_name in annotations:
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == 81]
        bbox = np.array(R)
        difficult = np.array([0] * len(R)).astype(np.bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[image_name] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 81:
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(num_dets)

    for d in range(num_dets):
        R = unknown_class_recs[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # OSE = is_unk / n_unk
    # logger.info('Number of unknowns detected knowns (for class '+ classname + ') is ' + str(is_unk))
    # logger.info("Num of unknown instances: " + str(n_unk))
    # logger.info('OSE: ' + str(OSE))

    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)
    # print(fp_open_set)
    # print(len(fp_open_set))

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


class WI:
    def __init__(self):
        self.args = set_up_parse()
        modelname = (self.args.config_file).split("/")[2].split(".")[0]
        cfg_OUTPUT_DIR = "../data/VOC-Detection/faster-rcnn/{}/random_seed_0".format(modelname)

        self.img_path = self.args.dataset_dir + "/val2017/"
        self.test_dataset = self.args.test_dataset  
        self.gt_OODcoco_api = COCO(self.args.dataset_dir + "/annotations/instances_val2017_mixed_OOD.json")
        self.gt_IDcoco_api = COCO(self.args.dataset_dir + "/annotations/instances_val2017_mixed_ID.json")

        inference_output_dir = get_inference_output_dir(
            cfg_OUTPUT_DIR,
            self.test_dataset,
            self.args.inference_config,
            self.args.image_corruption_level)
        prediction_file_name = os.path.join(
            inference_output_dir,
            'coco_instances_results_idood.json')
        self.res_coco_api = self.gt_OODcoco_api.loadRes(prediction_file_name)

        self.classes = list(self.gt_IDcoco_api.cats.keys()) + [81]
        self.num_seen_classes = len(self.classes) - 1

        imgidlist = list(self.gt_OODcoco_api.imgs.keys())
        self.imgidlist = []
        for imgID in imgidlist:
            gt_list_this_img = self.gt_OODcoco_api.loadAnns(self.gt_OODcoco_api.getAnnIds(imgIds=imgID))
            if len(gt_list_this_img) == 0:
                continue
            self.imgidlist.append(imgID)

    def readdata(self):
        # load groundtruth and prediciton
        self.res = {}
        for class_i in self.classes:
            self.res[class_i] = {}
        self.gt = {}
        for imgID in self.imgidlist:
            gtID_list_this_img = self.gt_IDcoco_api.loadAnns(self.gt_IDcoco_api.getAnnIds(imgIds=[imgID]))
            gtOOD_list_this_img = self.gt_OODcoco_api.loadAnns(self.gt_OODcoco_api.getAnnIds(imgIds=[imgID]))
            img_gt = np.array([gti['bbox'] + [gti["category_id"]] for gti in gtID_list_this_img] + 
                              [gti['bbox'] + [81] for gti in gtOOD_list_this_img])
            img_gt[:, 2] = img_gt[:, 0] + img_gt[:, 2]
            img_gt[:, 3] = img_gt[:, 1] + img_gt[:, 3]
            self.gt.update({imgID: img_gt})

            res_list_this_img = self.res_coco_api.loadAnns(self.res_coco_api.getAnnIds(imgIds=[imgID]))
            for class_i in self.classes:
                res_list = [res for res in res_list_this_img if res["category_id"]==class_i]
                if len(res_list)==0:
                    img_res = np.array([])
                else:
                    # xyhw
                    img_res = np.array([res['bbox'] + [res[self.sort_scores_name]]  for res in res_list])
                    img_res[:, 2] = img_res[:, 0] + img_res[:, 2]
                    img_res[:, 3] = img_res[:, 1] + img_res[:, 3]
                self.res[class_i].update({imgID: img_res})

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou
            
    def run(self):
        self.sort_scores_name = "complete_scores"
        self.readdata()

        aps = defaultdict(list)  # iou -> ap per class
        recs = defaultdict(list)
        precs = defaultdict(list)
        all_recs = defaultdict(list)
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        fp_os = defaultdict(list)
        for class_i in self.classes:
            thresh = 50
            rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set = voc_eval(self.res, self.gt, class_i)
            aps[thresh].append(ap * 100)
            unk_det_as_knowns[thresh].append(unk_det_as_known)
            num_unks[thresh].append(num_unk)
            all_precs[thresh].append(prec)
            all_recs[thresh].append(rec)
            tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
            fp_os[thresh].append(fp_open_set)
            try:
                recs[thresh].append(rec[-1] * 100)
                precs[thresh].append(prec[-1] * 100)
            except:
                recs[thresh].append(0)
                precs[thresh].append(0)
        wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os)
        print('Wilderness Impact: ' + str(wi))
        print('Wilderness Impact@recall0.8: ' + str(wi[0.8]))


eva = WI()
eva.run()