"""
VOC style object detection evaluation in memory, without saving/loading detection/annotation files
Adapted from https://github.com/GOATmessi7/RFBNet/blob/master/data/voc_eval.py, by Bharath Hariharan
See voc_evaluate() function below (the main function to do VOC evaluation).
Author: Muhammet Bastan, mubastan@gmail.com, 06 November 2018
Original file header:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
"""

import numpy
import copy
def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default: False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in numpy.arange(0.0, 1.1, 0.1):
            if numpy.sum(rec >= t) == 0:
                p = 0
            else:
                p = numpy.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = numpy.concatenate(([0.], rec, [1.]))
        mpre = numpy.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = numpy.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_evaluate(detections, annotations, cid, ovthresh=0.5, use_07_metric=True):
    """
    Top level function that does the PASCAL VOC evaluation.
    :param detections: Bounding box detections dictionary, keyed on class id (cid) and image_file,
                       dict[cid][image_file] = numpy.array([[x1,y1,x2,y2,score], [...],...])
    :param annotations: Ground truth annotations, keyed on image_file,
                       dict[image_file] = numpy.array([[x1,y1,x2,y2,score], [...],...])
    :param cid: Class ID (0 is typically reserved for background, but this function does not care about the value)
    :param ovthresh: Intersection over union overlap threshold, above which detection is considered as correct,
                       if it matches to a ground truth bounding box along with its class label (cid)
    :param use_07_metric: Whether to use VOC 2007 metric
    :return: recall, precision, ap (average precision)
    """
    # detections {81: [np, np, np...]}, np = numpy.array([[x1,y1,x2,y2,score], [...],...])
    # annotations [np, np, np], np = numpy.array([[x1,y1,x2,y2,class_id], [...],...])
    # cid = 81


    # extract ground truth objects from the annotations for this class
    class_gt_bboxes = {}
    npos = 0  # number of ground truth bboxes having label cid
    # annotations keyed on image file names or paths or anything that is unique for each image
    for image_name in annotations:
        # for each image list of objects: [[x1,y1, x2,y2, cid], [], ...]
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == cid]
        bbox = numpy.array(R)
        # difficult is not stored: take it as 0/false
        difficult = numpy.array([0] * len(R)).astype(numpy.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # detections' image file names/paths
    det_image_files = []
    confidences = []
    det_bboxes = []
    # detections should be keyed on class_id (cid)
    class_dict = detections[cid]
    for image_file in class_dict:
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = numpy.array(det_bboxes)
    confidences = numpy.array(confidences)

    # number of detections
    num_dets = len(det_image_files)
    tp = numpy.zeros(num_dets)
    fp = numpy.zeros(num_dets)

    if det_bboxes.shape[0] == 0:
        return 0., 0., 0.

    # sort detections by confidence
    sorted_ind = numpy.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(num_dets):
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -numpy.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ## compute overlaps
            # intersection
            ixmin = numpy.maximum(BBGT[:, 0], bb[0])
            iymin = numpy.maximum(BBGT[:, 1], bb[1])
            ixmax = numpy.minimum(BBGT[:, 2], bb[2])
            iymax = numpy.minimum(BBGT[:, 3], bb[3])
            iw = numpy.maximum(ixmax - ixmin + 1., 0.)
            ih = numpy.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # IoU
            overlaps = inters / uni
            ovmax = numpy.max(overlaps)
            jmax = numpy.argmax(overlaps)

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
    fp = numpy.cumsum(fp)
    tp = numpy.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / numpy.maximum(tp + fp, numpy.finfo(numpy.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return recall, precision, ap, rec, prec, state, det_image_files