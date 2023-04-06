"""
Probabilistic Detectron Inference Script
"""
import core
import json
import os
import sys
import torch
import tqdm
from shutil import copyfile
import numpy as np

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, MetadataCatalog

# Project imports
from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from evaluator import compute_average_precision
from inference.inference_utils import instances_to_json, get_inference_output_dir, build_predictor
from detectron2.structures import BoxMode, Instances, RotatedBoxes, Boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from detectron2.data.detection_utils import read_image


def extract_ID(outputs):
    inst_ID = Instances((outputs.image_size[0], outputs.image_size[1]))
    pred_classes = outputs.pred_classes
    keepindex = torch.where(pred_classes!=81)

    inst_ID.pred_boxes = outputs.pred_boxes[keepindex]
    inst_ID.scores = outputs.scores[keepindex]
    inst_ID.pred_classes = outputs.pred_classes[keepindex]
    inst_ID.pred_cls_probs = outputs.pred_cls_probs[keepindex]
    inst_ID.inter_feat = outputs.inter_feat[keepindex]
    inst_ID.det_labels = outputs.det_labels[keepindex]
    inst_ID.pred_boxes_covariance = outputs.pred_boxes_covariance[keepindex]
    inst_ID.complete_scores = outputs.complete_scores[keepindex]
    return inst_ID


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads#
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)
    cat_mapping_dict.update({81:81})

    # Build predictor
    predictor = build_predictor(cfg)
    test_data_loader = build_detection_test_loader(
        cfg, dataset_name=args.test_dataset)

    final_output_list_idood = [] # test mixed dataset, include ood, id
    final_output_list_id = [] # test mixed dataset, include id


    dataname = args.test_dataset.split("_")[0]
    if not args.eval_only:
        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data_loader)) as pbar:
                for idx, input_im in enumerate(test_data_loader):
                    outputs = predictor(input_im, args.pretest)
                    
   
                    final_output_list_idood.extend(
                        instances_to_json(
                            outputs,
                            input_im[0]['image_id'],
                            cat_mapping_dict))
                    
                    outputs_ID = extract_ID(outputs)
                    final_output_list_id.extend(
                        instances_to_json(
                            outputs_ID,
                            input_im[0]['image_id'],
                            cat_mapping_dict))

                    pbar.update(1)


        big_inference_output_dir = inference_output_dir
        if args.pretest:
            with open(os.path.join(big_inference_output_dir, '{}_instances_results_pretest.json'.format(dataname)), 'w') as fp:
                json.dump(final_output_list_idood, fp, indent=4, separators=(',', ': '))
        else:

            with open(os.path.join(big_inference_output_dir, '{}_instances_results_idood.json'.format(dataname)), 'w') as fp:
                json.dump(final_output_list_idood, fp, indent=4, separators=(',', ': '))

        with open(os.path.join(big_inference_output_dir, '{}_instances_results_id.json'.format(dataname)), 'w') as fp:
            json.dump(final_output_list_id, fp, indent=4, separators=(',', ': '))

    if 'ood' not in args.test_dataset:
        compute_average_precision.main(args, cfg, dataname)



if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    # args.num_machines = 8

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
