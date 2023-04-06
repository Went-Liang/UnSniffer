import os
import torch
import torchvision
from inference.ncut_torch import torch_ncut_detection

def update(outputs, keep):
    for key in outputs.keys():
        try:
            outputs[key] = outputs[key][keep]
        except:
            continue
    # return outputs

def remove_by_scores(scores, threshold, retain_larger):
    if retain_larger:
        keepidx = torch.where(scores >= threshold)[0]
    else:
        keepidx = torch.where(scores < threshold)[0]
    return keepidx

def torch_ncut_top(scores, res_boxes, similarity, thresh):
    if res_boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64)
    
    clusters = torch_ncut_detection(proposals=res_boxes, device=torch.device('cuda'), thresh=thresh, sim_matrix=similarity)
    clusters_Container = {} 
    for i, cluster_index in enumerate(clusters):
        if cluster_index.item() not in clusters_Container:
            clusters_Container[cluster_index.item()] = [i]
        else:
            clusters_Container[cluster_index.item()].append(i)
   
    clusters_scores = [scores[clusters_Container[key]] for key in clusters_Container.keys()]
    
    keepidx = torch.Tensor([clusters_Container[key][torch.argmax(clusters_scores[i])] for i, key in enumerate(clusters_Container.keys())])
    return torch.tensor(keepidx, dtype=torch.int64)

def NMS(res_boxes, scores=None, iou_threshold=0.5):
    if scores == None:
        scores = torch.ones(res_boxes.shape[0])
    keepidx = torchvision.ops.nms(boxes=res_boxes, scores=scores, iou_threshold=iou_threshold)
    return keepidx

def topK(scores, topk_num):
    topk_num = min(topk_num, len(scores))
    keepidx = torch.sort(scores, descending=True)[1][:topk_num]
    return keepidx