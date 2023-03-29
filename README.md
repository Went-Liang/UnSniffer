# Unknown Sniffer for Object Detection: Don’t Turn a Blind Eye to Unknown Objects (CVPR 2023)

[`Paper`](https://arxiv.org/abs/2303.13769) [`Video`](https://www.bilibili.com/video/BV1xM4y1z7Hv/?buvid=XYC2EDBCCC2B3C4802E4AAD1035EFACB2AC57&is_story_h5=false&mid=vL1Nha2VQkhwiq6%2FLPmtbA%3D%3D&plat_id=147&share_from=ugc&share_medium=android&share_plat=android&share_session_id=a280f047-3ced-4b9d-acb2-40244f9a55fb&share_source=WEIXIN&share_tag=s_i&timestamp=1679647440&unique_k=2n8pmaV&up_id=253369834&vd_source=668f39404189897ee2f8d0c7596f9f4e) [`Youtube`](https://www.youtube.com/watch?v=AI2mfO2CycM) 

#### [Wenteng Liang](https://github.com/Went-Liang)<sup>\*</sup>, [Feng Xue](https://github.com/XuefengBUPT)<sup>\*</sup>, [Yihao Liu](https://github.com/howtoloveyou), [Guofeng Zhong](), [Anlong Ming](https://teacher.bupt.edu.cn/mal) ####

(:star2: denotes equal contribution)

# Introduction

The recently proposed open-world object and open-set detection have achieved a breakthrough in finding never-seen-before objects and distinguishing them from known ones. However, their studies on knowledge transfer from known classes to unknown ones are not deep enough, resulting in the scanty capability for detecting unknowns hidden in the background. In this paper, we propose the unknown sniffer (UnSniffer) to find both unknown and known objects. Firstly, the generalized object confidence (GOC) score is introduced, which only uses known samples for supervision and avoids improper suppression of unknowns in the background. Significantly, such confidence score learned from known objects can be generalized to unknown ones. Additionally, we propose a negative energy suppression loss to further suppress the non-object samples in the background. Next, the best box of each unknown is hard to obtain during inference due to lacking their semantic information in training. To solve this issue, we introduce a graph-based determination scheme to replace hand-designed non-maximum suppression (NMS) post-processing. Finally, we present the Unknown Object Detection Benchmark, the first publicly benchmark that encompasses precision evaluation for unknown detection to our knowledge. Experiments show that our method is far better than the existing state-of-the-art methods.


# Todo

We are organizing the code and dataset and will open-source them as soon as possible.



# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.



**Acknowledgments:**

UnSniffer builds on previous works code base such as [VOS](https://github.com/deeplearning-wisc/vos). If you found UnSniffer useful please consider citing these works as well.

