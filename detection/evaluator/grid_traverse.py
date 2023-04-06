import os
import cv2
import copy
import torch
import numpy as np
import torchvision
import torch_geometric
import matplotlib.pyplot as plt
from operator import itemgetter
from pycocotools.coco import COCO
from detectron2.engine import default_argument_parser
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou
import sys
sys.path.append('../')
from ncut_torch import torch_ncut_detection

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

def cumTpFp(gt_num, res_num, scores, label, overlapRatio, iou_matrix):
    # det_state: [label, score, tp, fp], tp, fp = 0 or 1

    # gtRect: xmin, ymin, xmax, ymax
    det_state = [[label, 0., 0, 1]] * res_num
    iou_max = 0
    maxIndex = -1
    blockIdx = -1
    for cnt in range(len(det_state)):
        det_state[cnt] = [label, scores[cnt].item(), 0, 1]  # 更新score值
    visited = [0] * gt_num
    if res_num != len(scores):
        print("Num of scores does not match detection results!")
    if res_num == 0:
        return det_state

    for indexDet in range(res_num):
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for indexGt in range(gt_num):
            iou = iou_matrix[indexGt][indexDet].item()
            if iou > iou_max:
                iou_max = iou
                maxIndex = indexDet
                blockIdx = indexGt
        if iou_max >= overlapRatio and visited[blockIdx] == 0:
            det_state[maxIndex] = [label, scores[indexDet].item(), 1, 0]
            visited[blockIdx] = 1
    return det_state

def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


class Eval:
    def __init__(self):
        self.iou_threshold = 0.5
        self.args = set_up_parse()
        modelname = (self.args.config_file).split("/")[2].split(".")[0]
        self.cfg_OUTPUT_DIR = "../data/VOC-Detection/faster-rcnn/{}/random_seed_0".format(modelname)

        self.test_dataset = "voc_completely_annotation_pretest"
        self.gt_coco_api = COCO(self.args.dataset_dir + "/voc0712_train_completely_annotation200.json")

        inference_output_dir = get_inference_output_dir(
            self.cfg_OUTPUT_DIR,
            self.test_dataset,
            self.args.inference_config,
            self.args.image_corruption_level)
        prediction_file_name = os.path.join(
            inference_output_dir,
            'voc_instances_results_pretest.json')
        self.res_coco_api = self.gt_coco_api.loadRes(prediction_file_name)

        self.savepath = inference_output_dir
        

    def compute_iou(self, gtRects, detRects):
        gtRects = gtRects.cuda()
        detRects = detRects.cuda()
        detRects[:, 2] = detRects[:, 0] + detRects[:, 2]
        detRects[:, 3] = detRects[:, 1] + detRects[:, 3]
        gtRects[:, 2] = gtRects[:, 0] + gtRects[:, 2]
        gtRects[:, 3] = gtRects[:, 1] + gtRects[:, 3]
        iou_matrix = pairwise_iou(Boxes(gtRects), Boxes(detRects)).cpu()
        return iou_matrix

    def readdata(self, gt_coco_api, res_coco_api):
        self.imgidlist = list(gt_coco_api.imgs.keys())
        self.img_dict = {}
        for imgID in self.imgidlist:
            gt_list_this_img = gt_coco_api.loadAnns(gt_coco_api.getAnnIds(imgIds=[imgID]))
            if len(gt_list_this_img) == 0:
                print("There are no gtboxes in {}\n".format(gt_coco_api.loadImgs([imgID])[0]['file_name']))
                continue
            self.img_dict[imgID] = {}
            # xyhw
            self.img_dict[imgID].update({"gt_num": len(gt_list_this_img)})
            self.img_dict[imgID].update({"gt_boxes": torch.Tensor([gti['bbox'] for gti in gt_list_this_img]).reshape(-1, 4)})

            res_list_this_img = res_coco_api.loadAnns(res_coco_api.getAnnIds(imgIds=[imgID]))
            res_list_this_img_pred_id = [res for res in res_list_this_img if res["category_id"]!=81]
            self.img_dict[imgID].update({"ID_res": res_list_this_img_pred_id})

            res_list_this_img = [res for res in res_list_this_img if res["category_id"]==81]
            self.img_dict[imgID].update({"res_num": len(res_list_this_img)})
            # xyhw
            self.img_dict[imgID].update({"res_boxes": torch.Tensor([res['bbox'] for res in res_list_this_img]).reshape(-1, 4)})
            self.img_dict[imgID].update({"res_scores": torch.Tensor([res['score'] for res in res_list_this_img])})
            self.img_dict[imgID].update({"res_energy": torch.Tensor([torch.logsumexp(torch.Tensor(res["inter_feat"][:-1]), dim=0).item() for res in res_list_this_img])})
            self.img_dict[imgID].update({"res_complete_scores": torch.Tensor([res['complete_scores'] for res in res_list_this_img])})
            self.img_dict[imgID].update({"res_complete_feat": torch.Tensor([res['complete_feat'] for res in res_list_this_img])})

            iou_matrix = self.compute_iou(self.img_dict[imgID]["gt_boxes"], self.img_dict[imgID]["res_boxes"])
            self.img_dict[imgID].update({"IOU": iou_matrix})
    

    def compute_det_state(self):
        for imgID in self.imgidlist:
            det_state = cumTpFp(self.img_dict[imgID]["gt_num"], self.img_dict[imgID]["res_num"], \
                    self.img_dict[imgID][self.sort_scores_name] if self.sort_scores_name != "" else torch.ones(self.img_dict[imgID]["res_num"]),\
                     0, self.iou_threshold, self.img_dict[imgID]["IOU"])
            det_state = torch.Tensor(det_state)
            self.img_dict[imgID].update({"det_state": det_state})

    def update(self, imgID, keepidx):
        self.img_dict[imgID]["res_num"] = keepidx.shape[0]
        self.img_dict[imgID]["res_boxes"] = self.img_dict[imgID]["res_boxes"][keepidx]
        self.img_dict[imgID]["res_scores"] = self.img_dict[imgID]["res_scores"][keepidx]
        self.img_dict[imgID]["res_energy"] = self.img_dict[imgID]["res_energy"][keepidx]
        self.img_dict[imgID]["res_complete_scores"] = self.img_dict[imgID]["res_complete_scores"][keepidx]
        self.img_dict[imgID]["res_complete_feat"] = self.img_dict[imgID]["res_complete_feat"][keepidx]
        self.img_dict[imgID]["IOU"] = self.img_dict[imgID]["IOU"][:, keepidx]

        det_state = cumTpFp(self.img_dict[imgID]["gt_num"], self.img_dict[imgID]["res_num"], \
            self.img_dict[imgID][self.sort_scores_name] if self.sort_scores_name != "" else torch.ones(self.img_dict[imgID]["res_num"]), \
                0, self.iou_threshold, self.img_dict[imgID]["IOU"])
        self.img_dict[imgID]["det_state"] = torch.Tensor(det_state)

    def CumSum(self):
        fp_copy = sorted(self.fp, key=itemgetter(0), reverse=True)  
        cumsum = []
        cumPre = 0
        fp_th = 0
        fp_th_num = 0
        for index, pair in enumerate(fp_copy):
            cumPre += (fp_copy[index][1])
            cumsum.append(cumPre)  
            if fp_copy[index][0] > self.eval_threshold:
                fp_th_num += 1
                if fp_copy[index][1] == 1:  # false positive
                    fp_th += 1
        fppw = float(fp_th) / float(fp_th_num)  # FP
        return cumsum, fp_th, fppw

    def CumSum_tp(self):
        tp_copy = sorted(self.tp, key=itemgetter(0), reverse=True)
        cumsum = [] 
        cumPre = 0 
        tp_th = 0 
        tp_th_num = 0 
        for index, pair in enumerate(tp_copy):
            cumPre += (tp_copy[index][1])
            cumsum.append(cumPre)
            if tp_copy[index][0] > self.eval_threshold:
                tp_th_num += 1 
                if tp_copy[index][1] == 1: 
                    tp_th += 1
        tp_precision = float(tp_th) / float(tp_th_num)
        return cumsum, tp_th, tp_precision

    def computeAp(self):
        num = len(self.tp) 
        prec = [] 
        rec = []  
        fpr = []  
        ap = 0  
        if num == 0 or self.all_num_pos == 0:
            return prec, rec, ap
        tp_cumsum, tp_th, tp_precision = self.CumSum_tp()
        fp_cumsum, fp_th, fppw = self.CumSum()
        # Compute precision. Compute recall.
        for i in range(num):
            prec.append(float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))  
            rec.append(float(tp_cumsum[i]) / float(self.all_num_pos))  
            fpr.append(float(fp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))   

        tmp = 0
        
        max_precs = [0.] * 11
        start_idx = num - 1
        j = 10
        while j >= 0:
            i = start_idx
            while i >= 0:
                tmp = j / 10.0
                if rec[i] < tmp:
                    start_idx = i
                    if j > 0:
                        max_precs[j - 1] = max_precs[j]
                        break
                else:
                    if max_precs[j] < prec[i]:
                        max_precs[j] = prec[i]
                i -= 1
            j -= 1
        for iji in range(11):
            ap += max_precs[iji] / 11.0

        recall = float(tp_th) / float(self.all_num_pos)
        precision = tp_precision
        return precision, recall, ap    

    def eval(self):
        state_all = []
        self.tp = []  # tp = [(float, int)]
        self.fp = []  # fp = [(float, int)]
        self.all_num_pos = 0  
        for imgID in self.imgidlist:
            det_state = self.img_dict[imgID]["det_state"]
            self.all_num_pos += self.img_dict[imgID]["gt_num"]  
            state_all.append(det_state)  
        state_all = torch.cat(state_all)
        for state in state_all:
            self.tp.append((state[1].item(), state[2].item())) # scores, tp
            self.fp.append((state[1].item(), state[3].item())) # scores, fp
        precision, recall, ap = self.computeAp()
        print("Ap: ", ap)
        print("precision: ", precision)
        print("recall: ", recall) 
        
        return [ap, precision, recall]
    
    def analy(self, stage="init", eval=True):
        if eval:
            matrix = self.eval()
            return matrix
        

    def remove_by_scores(self, scores_name, threshold, retain_larger, vis_removed_proposals_type=""):
        for imgID in self.imgidlist:
            scores = self.img_dict[imgID][scores_name]
            if retain_larger:
                keepidx = torch.where(scores >= threshold)[0]
            else:
                keepidx = torch.where(scores < threshold)[0]
            if vis_removed_proposals_type != "":
                self.vis_removed(vis_removed_proposals_type, imgID, keepidx)
            self.update(imgID, keepidx)


    def torch_ncut(self, scores_name, thresh):
        for imgID in self.imgidlist:
            scores = self.img_dict[imgID][scores_name]
            if self.img_dict[imgID]["res_num"] == 0:
                continue
            similarity = torch.cosine_similarity(self.img_dict[imgID]["res_complete_feat"].unsqueeze(1), self.img_dict[imgID]["res_complete_feat"].unsqueeze(0), dim=-1)
            clusters = torch_ncut_detection(proposals=self.img_dict[imgID]["res_boxes"], device=torch.device('cuda'), thresh=thresh, sim_matrix=similarity)
            
            clusters_Container = {} 
            for i, cluster_index in enumerate(clusters):
                if cluster_index.item() not in clusters_Container:
                    clusters_Container[cluster_index.item()] = [i]
                else:
                    clusters_Container[cluster_index.item()].append(i)
            
            clusters_scores = [scores[clusters_Container[key]] for key in clusters_Container.keys()]
            
            keepidx = torch.Tensor([clusters_Container[key][torch.argmax(clusters_scores[i])] for i, key in enumerate(clusters_Container.keys())])
            keepidx = torch.tensor(keepidx, dtype=torch.int64)
            self.update(imgID, keepidx)

    def run(self, ncut_thres):
        self.torch_ncut(scores_name="res_complete_scores", thresh=ncut_thres)
        matrix = self.analy("ncut")
        return matrix

    def grid(self):
        self.iou_threshold = 0.5
        self.sort_scores_name = "res_complete_scores"
        self.eval_threshold = 0.5
        
        self.readdata(self.gt_coco_api, self.res_coco_api)
        self.compute_det_state()

        self.analy("1init")
        self.remove_by_scores("res_complete_scores", 0.5, retain_larger=True)
        self.analy("3_removeby_complete_scores50")
        self.img_dict_copy = copy.deepcopy(self.img_dict)

        maxap = 0
        maxap_precision = 0
        maxap_recall = 0
        best_ncut_thres = 0
        with open(os.path.join(self.savepath, "threshold.txt"), "w") as f:
            for ncut_thres in np.arange(0.68, 0.72, 0.01):
                self.img_dict = copy.deepcopy(self.img_dict_copy)
                matrix = self.run(ncut_thres)
                ap, precision, recall = matrix[0], matrix[1], matrix[2]
                if ap > maxap:
                    maxap = ap
                    maxap_precision = precision
                    maxap_recall = recall
                    best_ncut_thres = ncut_thres


                f.write("{} {} {} {}\n".format(ncut_thres, ap, precision, recall))
                print("{} {} {} {}\n".format(ncut_thres, ap, precision, recall))
                print("best thres: {} {} {} {}\n".format(best_ncut_thres, maxap, maxap_precision, maxap_recall))


            f.write("best thres: {} {} {} {}\n".format(best_ncut_thres, maxap, maxap_precision, maxap_recall))
            print("final best thres: {} {} {} {}\n".format(best_ncut_thres, maxap, maxap_precision, maxap_recall))
            
        print("\n")
        print(self.cfg_OUTPUT_DIR + "/inference/ncut_threshold.txt")
        with open(os.path.join(self.cfg_OUTPUT_DIR, 'inference', "ncut_threshold.txt"), "w") as f:
            f.write("{}\n".format(best_ncut_thres))            
                

eva = Eval()
eva.grid()