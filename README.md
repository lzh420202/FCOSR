# FCOSR: A Simple Anchor-free Rotated Detector for Aerial Object Detection

> **[FCOSR: A Simple Anchor-free Rotated Detector for Aerial Object Detection](#)**<br>
> arXiv preprint ([arXiv:2111.10780](https://arxiv.org/abs/2111.10780)).

This implement is modified from [mmdetection](https://github.com/open-mmlab/mmdetection). 
We also refer to the codes of 
[ReDet](https://github.com/csuhan/ReDet), 
[PIoU](https://github.com/clobotics/piou), 
and [ProbIoU](https://github.com/ProbIOU/probiou-sample).

In the process of implementation, 
we find that only Python code processing will produce huge memory overhead on Nvidia devices.
Therefore, we directly write the label assignment module proposed in this paper in the form of CUDA extension of Pytorch.
The program could not work effectively when we migrate it to cuda 11 (only support cuda10).
By applying CUDA expansion, the memory utilization is improved and a lot of unnecessary calculations are reduced.
We also try to train FCOSR-M on 2080ti (4 images per device), which can basically fill memory of graphics card.

## Install

Please refer to [install.md](./install.md) for installation and dataset preparation.


## Getting Started

Please see [get_started.md](./get_started.md) for the basic usage.

## Model Zoo

![benchmark](resources/FCOSR/sp_vs_acc.png)

**The <font color='red'>password</font> of baiduPan is <font color='red' bolder>ABCD</font>**

### FCOSR serise DOTA 1.0 result.FPS(2080ti) [Detail](./Details.md#fcosr-serise-dota-10-result)

|Model|backbone|MS|Sched.|Param.|Input|GFLOPs|FPS|mAP|download|
|:-|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCOSR-S|Mobilenet v2|-|3x|7.32M|1024×1024|101.42|23.7|74.05|[model](https://pan.baidu.com/s/1kYN7oE2I8naX2zVQo02iWQ)/[cfg](configs/fcosrbox/fcosr_mobilenetv2_fpn_3x_dota10_single.py)|
|FCOSR-S|Mobilenet v2|✓|3x|7.32M|1024×1024|101.42|23.7|76.11|[model](https://pan.baidu.com/s/1hbYZmNU8WXG_EmlpgpkQ3A)/[cfg](configs/fcosrbox/fcosr_mobilenetv2_fpn_3x_dota10_ms.py)|
|FCOSR-M|ResNext50-32x4|-|3x|31.4M|1024×1024|210.01|14.6|77.15|[model](https://pan.baidu.com/s/1HItBizh5FpxGdwWzONloRw)/[cfg](configs/fcosrbox/fcosr_rx50_32x4d_fpn_3x_dota10_single.py)|
|FCOSR-M|ResNext50-32x4|✓|3x|31.4M|1024×1024|210.01|14.6|79.25|[model](https://pan.baidu.com/s/1J1ZrLyj8XI7rU0M9ULF5Bg)/[cfg](configs/fcosrbox/fcosr_rx50_32x4d_fpn_3x_dota10_ms.py)|
|FCOSR-L|ResNext101-64x4|-|3x|89.64M|1024×1024|445.75|7.9|77.39|[model](https://pan.baidu.com/s/1W0c-2_xKpg5DogqlBgcU9w)/[cfg](configs/fcosrbox/fcosr_rx101_64x4d_fpn_3x_dota10_single.py)|
|FCOSR-L|ResNext101-64x4|✓|3x|89.64M|1024×1024|445.75|7.9|78.80|[model](https://pan.baidu.com/s/1WK48mkbHBYgF7gfvh45g9g)/[cfg](configs/fcosrbox/fcosr_rx101_64x4d_fpn_3x_dota10_ms.py)|

### FCOSR serise DOTA 1.5 result. FPS(2080ti) [Detail](./Details.md#fcosr-serise-dota-15-result)

|Model|backbone|MS|Sched.|Param.|Input|GFLOPs|FPS|mAP|download|
|:-|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCOSR-S|Mobilenet v2|-|3x|7.32M|1024×1024|101.42|23.7|66.37|[model](https://pan.baidu.com/s/1qcW_DFF0mMrx4YzrCKTSOw)/[cfg](configs/fcosrbox/fcosr_mobilenetv2_fpn_3x_dota15_single.py)|
|FCOSR-S|Mobilenet v2|✓|3x|7.32M|1024×1024|101.42|23.7|73.14|[model](https://pan.baidu.com/s/1R2vW6tAKd091btm566nPRQ)/[cfg](configs/fcosrbox/fcosr_mobilenetv2_fpn_3x_dota15_ms.py)|
|FCOSR-M|ResNext50-32x4|-|3x|31.4M|1024×1024|210.01|14.6|68.74|[model](https://pan.baidu.com/s/1BC7pSmzA9y1n2sExIP__dw)/[cfg](configs/fcosrbox/fcosr_rx50_32x4d_fpn_3x_dota15_single.py)|
|FCOSR-M|ResNext50-32x4|✓|3x|31.4M|1024×1024|210.01|14.6|73.79|[model](https://pan.baidu.com/s/1ubCywgEoH-hssptVDAipyQ)/[cfg](configs/fcosrbox/fcosr_rx50_32x4d_fpn_3x_dota15_ms.py)|
|FCOSR-L|ResNext101-64x4|-|3x|89.64M|1024×1024|445.75|7.9|69.96|[model](https://pan.baidu.com/s/1__vt9AII-6SqxR8UU5VxlA)/[cfg](configs/fcosrbox/fcosr_rx101_64x4d_fpn_3x_dota15_single.py)|
|FCOSR-L|ResNext101-64x4|✓|3x|89.64M|1024×1024|445.75|7.9|75.41|[model](https://pan.baidu.com/s/1bFzKSpJDnVh3tu3By-YHGA)/[cfg](configs/fcosrbox/fcosr_rx101_64x4d_fpn_3x_dota15_ms.py)|

### FCOSR serise HRSC2016 result. FPS(2080ti)

|Model|backbone|Rot.|Sched.|Param.|Input|GFLOPs|FPS|AP50(07)|AP75(07)|AP50(12)|AP75(12)|download|
|:-|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCOSR-S|Mobilenet v2|✓|40k iters|7.29M|800×800|61.57|35.3|90.08|76.75|92.67|75.73|[model](https://pan.baidu.com/s/1HhJITM3jRCtFzAhGc7lv3Q)/[cfg](configs/fcosrbox/fcosr_mobilenetv2_fpn_40k_hrsc2016.py)|
|FCOSR-M|ResNext50-32x4|✓|40k iters|31.37M|800×800|127.87|26.9|90.15|78.58|94.84|81.38|[model](https://pan.baidu.com/s/1_aXC_IO9XT9ygwwvUZW1fw)/[cfg](configs/fcosrbox/fcosr_rx50_32x4d_fpn_40k_hrsc2016.py)|
|FCOSR-L|ResNext101-64x4|✓|40k iters|89.61M|800×800|271.75|15.1|90.14|77.98|95.74|80.94|[model](https://pan.baidu.com/s/16u940TyDTewCd4KvDkc1GA)/[cfg](configs/fcosrbox/fcosr_rx101_64x4d_fpn_40k_hrsc2016.py)|

### Lightweight FCOSR test result on Jetson Xavier NX (DOTA 1.0 single-scale). [Detail](./Details.md#lightweight-fcosr-test-result-on-jetson-xavier-nx-dota-10-single-scale)

|Model|backbone|Head channels|Sched.|Param|Size|Input|GFLOPs|FPS|mAP|onnx|TensorRT|
|:-|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCOSR-lite|Mobilenet v2|256|3x|6.9M|51.63MB|1024×1024|101.25|7.64|74.30|Wait|[rtr](https://pan.baidu.com/s/1Bhg3hfJJlc2-iJBPi79zFA)|
|FCOSR-tiny|Mobilenet v2|128|3x|3.52M|23.2MB|1024×1024|35.89|10.68|73.93|Wait|[rtr](https://pan.baidu.com/s/1WPLV7xjXkMLSes5Jf8cFgw)|
