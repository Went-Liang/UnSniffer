# Unknown Sniffer for Object Detection: Don’t Turn a Blind Eye to Unknown Objects (CVPR 2023)


#### [Wenteng Liang](https://github.com/Went-Liang)<sup>\*</sup>, [Feng Xue](https://xuefeng-cvr.github.io/)<sup>\*</sup>, [Yihao Liu](https://github.com/howtoloveyou), [Guofeng Zhong](), [Anlong Ming](https://teacher.bupt.edu.cn/mal) ####

(:star2: denotes equal contribution)


### [`Paper`](https://arxiv.org/abs/2303.13769) [`Bilibili`](https://www.bilibili.com/video/BV1xM4y1z7Hv/?buvid=XYC2EDBCCC2B3C4802E4AAD1035EFACB2AC57&is_story_h5=false&mid=vL1Nha2VQkhwiq6%2FLPmtbA%3D%3D&plat_id=147&share_from=ugc&share_medium=android&share_plat=android&share_session_id=a280f047-3ced-4b9d-acb2-40244f9a55fb&share_source=WEIXIN&share_tag=s_i&timestamp=1679647440&unique_k=2n8pmaV&up_id=253369834&vd_source=668f39404189897ee2f8d0c7596f9f4e) [`Youtube`](https://www.youtube.com/watch?v=AI2mfO2CycM) [`Slides`](https://docs.google.com/presentation/d/1YUxG_NnjeIiSZjHpIgS9wtETqZQ1MD0s/edit?usp=sharing&ouid=104225774732865902245&rtpof=true&sd=true) [`Project`](https://xuefengbupt.github.io/project_page/unsniffer_cvpr23.html)

# Introduction

The recently proposed open-world object and open-set detection have achieved a breakthrough in finding never-seen-before objects and distinguishing them from known ones. However, their studies on knowledge transfer from known classes to unknown ones are not deep enough, resulting in the scanty capability for detecting unknowns hidden in the background. In this paper, we propose the unknown sniffer (UnSniffer) to find both unknown and known objects. Firstly, the generalized object confidence (GOC) score is introduced, which only uses known samples for supervision and avoids improper suppression of unknowns in the background. Significantly, such confidence score learned from known objects can be generalized to unknown ones. Additionally, we propose a negative energy suppression loss to further suppress the non-object samples in the background. Next, the best box of each unknown is hard to obtain during inference due to lacking their semantic information in training. To solve this issue, we introduce a graph-based determination scheme to replace hand-designed non-maximum suppression (NMS) post-processing. Finally, we present the Unknown Object Detection Benchmark, the first publicly benchmark that encompasses precision evaluation for unknown detection to our knowledge. Experiments show that our method is far better than the existing state-of-the-art methods.

[//]: # (![]&#40;https://xuefengbupt.github.io/project_page/img/unsniffer_demo1.gif&#41;)

[//]: # (![]&#40;https://xuefengbupt.github.io/project_page/img/unsniffer_demo2.gif&#41;)

<div id="demo" style="display: flex; justify-content: center; margin-bottom: 20px;">

<img src="unsniffer_demo1.gif" width="305" style="margin-right: 30px;">

<img src="unsniffer_demo2.gif" width="400" style="margin-left: 30px;">

</div>

# Requirements
```bash
pip install -r requirements.txt
```

In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Dataset Preparation

The datasets can be downloaded using this [link](https://drive.google.com/drive/folders/1Mh4xseUq8jJP129uqCvG9cSLdjqdl0Jo?usp=sharing).

**PASCAL VOC**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

Please download the JPEGImages data from the [link](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing) provided by [VOS](https://github.com/deeplearning-wisc/vos).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         ├── voc0712_train_completely_annotation200.json
         └── val_coco_format.json

**COCO**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_coco_ood.json
            ├── instances_val2017_mixed_ID.json
            └── instances_val2017_mixed_OOD.json
         ├── train2017
         └── val2017

# Training
```bash
python train_net.py --dataset-dir VOC_DATASET_ROOT --num-gpus 2 --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --random-seed 0 --resume
```
The pretrained models for Pascal-VOC can be downloaded from [UnSniffer](https://drive.google.com/file/d/1kp60e6nh0iIOPd41f4JI6Yo9r_r7MqRo/view?usp=sharing). Please put the model in UnSniffer/detection/data/VOC-Detection/faster-rcnn/UnSniffer/random_seed_0/.

# Pretesting
The function of this process is to obtain the threshold, which only uses part of the training data.
```bash
sh pretest.sh
```

# Evaluation on the VOC
```bash
python apply_net.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_custom_val  --config-file VOC-Detection/faster-rcnn/UnSniffer.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

# Evaluation on the COCO-OOD
```bash
sh test_ood.sh
```

# Evaluation on the COCO-Mix

```bash
sh test_mixed.sh
```

# Visualize prediction results
```bash
sh vis.sh
```

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


# Citation

If you use UnSniffer, please consider citing:

    @inproceedings{liang2023unknown,
    title={Unknown Sniffer for Object Detection: Don't Turn a Blind Eye to Unknown Objects},
    author={Liang, Wenteng and Xue, Feng and Liu, Yihao and Zhong, Guofeng and Ming, Anlong},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
    }

**Acknowledgments:**

UnSniffer builds on previous works code base such as [VOS](https://github.com/deeplearning-wisc/vos) and [OWOD](https://github.com/JosephKJ/OWOD). If you found UnSniffer useful please consider citing these works as well.
