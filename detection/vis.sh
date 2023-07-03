cd evaluator/

python visualization.py --dataset-dir COCO_DATASET_ROOT --test-dataset coco_extended_ood_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

python visualization.py --dataset-dir COCO_DATASET_ROOT --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

cd ..