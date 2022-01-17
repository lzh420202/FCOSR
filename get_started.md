# Getting started

We are glad that you use FCOSR. This guide mainly provides the basic usage of FCOSR. Our code has been modified based on mmdetection, so it is slightly different from the way you are used to. Please read it carefully.<br>
For installation instructions, please see [install.md](install.md).

## Prepare DOTA dataset.

It is recommended to symlink the dataset root to `FCOSR/data`.

Here, we give an example for single scale data preparation of DOTA1.0.

First, make sure your initial data are in the following structure.
```
data/dota10
├── train
│   ├──images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```
Split the original images and create COCO format json. 
```
python DOTA_devkit/prepare_dota.py path_to_dota path_to_split
```
Then you will get data in the following structure
```
dota10_1024
├── test1024
│   ├── DOTA_test1024.json
│   └── images
└── trainval1024
     ├── DOTA_trainval1024.json
     └── images
```
The arguments of prepare_dota.py in the list.

|Arguments|type|default|discribe|
|-|-|-|-|
|srcpath|str|None|source path|
|dstpath|str|None|DOTA save path.|
|-s, --subsize|int|1024|patch size|
|-g, --gaps|int|200|gap|
|-t, --type|str|"dota10"|dota version|
|-p, --processor-num|int|16|multiprocessor number|
|-d, --drop-threshold|float|0.5|Drop samples threshold|
|--scales|list[float]|None|None means single scale|


## Inference with pretrained models


### Test a dataset

- [x] specified GPU testing (unify single GPU and multi GPU)

You can use the following commands to test a dataset.

```shell
python analyse_result.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
**Note:** This command will generate a result file (.pkl) and DOTA detection result in a folder.

The arguments of analyse_result.py in the list.

|Arguments|type|default|discribe|
|-|-|-|-|
|config|str|None|config file path|
|checkpoint|str|None|checkpoint path|
|-d, --draw|None|False|Draw result switch|
|-n, --draw-num|str or int|'all'|Draw number|
|-s, --source-dir|str|None|Source origin image dir|
|-p, --processor-num|int|16|multiprocessor number|
|-D, --draw-dir|str|None|Draw result folder path|
|-g, --gpus|str|"0"|GPUs, use common to separate multi devices.|


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**\*Important\***: The default learning rate in config files is for 8 GPUs.
If you use less or more than 8 GPUs, you need to set the learning rate proportional
to the GPU num, e.g., 0.01 for 4 GPUs and 0.04 for 16 GPUs.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.


### Train with multiple GPUs

```shell
# GPU_order is a string, just like "0,1,2,3". You can specify the device.
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_order} [optional arguments]
```

Optional arguments are:

- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.


### How to use benchmark

Speed
```shell
# modify and run shell script file analyse_inference_speed.sh
./analyse_inference_speed.sh
```

### Transfer to onnx and TensorRT

transfer path: checkpoint(.pth) -> onnx format(.onnx) -> tensorrt engine(.trt)

```shell
# checkpoint(.pth) -> onnx format(.onnx)
python tools/deployment/pytorch2onnx.py config_file checkpoint_file --output-file output_path --shape 1024 1024
```

onnx -> tensorrt ([onnx-tensorrt 8.0](https://github.com/onnx/onnx-tensorrt/tree/release/8.0))

The TensorRT inference implement can be found in my other [repository](https://github.com/lzh420202/TensorRT_Inference)