# Install

You can create runtime based on docker. We will provide a dockerfile to help you, just wait. Now, you can build FCOSR on your device through following steps.

## Recommend system environments:
 - Ubuntu 18.04/20.04
 - python 3.7
 - cuda 10.0/10.1/10.2
 - cudnn 8.2 (optional, required for TensorRT deployment)
 - TensorRT 8.0 GA (8.0.1, optional)
 - pytorch 1.3.1/1.5.1
 - torchvision 0.4.2/0.6.1
 - mmdetection 2.15.1
 - mmcv-full 1.3.9
 - DOTA_devkit and shapely

**Note:** pytorch->onnx require torch>=1.5.1. Unfortunately, the CUDA components we designed in the cuda11 version are not working effectively.

## Install FCOSR

1. Install pytorch

    Download torch installation file from website (offline): 
    ```
    https://download.pytorch.org/whl/{cuda_version}/torch_stable.html
    ```
    **Note:** the website should be changed follow your cuda version. For example, cuda10.0 equal "cu110", cuda10.1 equal "cu101".
    install torch and torchvision.
    ```shell
    pip install torch-xxx.whl torchvision-xxx.whl
    ```

2. Install mmcv-full.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11.0` and `PyTorch 1.7.0`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

    Optionally you can compile mmcv from source if you need to develop both mmcv and mmdet. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

3. Install DOTA_devkit
    ```shell
    sudo apt update && sudo apt install swig -y
    cd DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    pip install shapely
    ```
    Adding DOTA_devkit path to PYTHONPATH (environment var).<br>
    **Note:** You can find DOTA_devkit in our implement, which includes datasets create function.

4. Install FCOSR.
    ```shell
    git clone https://github.com/lzh420202/FCOSR.git
    cd FCOSR
    pip install -r requirements/build.txt
    python setup.py develop
    ```

## TensorRT

This implement migrate from pytorch to TensorRT follow the path: pytorch -> onnx -> TensorRT.
Our code provide tools transfer pytorch to onnx. The conversion from onnx to tensorrt needs to be performed on the deployed devices, so we only provide the installation of the conversion model tool on Jetson Xavier NX.

### Recommend system environments:
 - Jetson Xavier NX / Jetson AGX Xavier
 - python 3.6
 - JetPack 4.6
 - cmake 3.14 or higher
 - protobuf 3.0.9
 - CUDA 10.2 (from JetPack)
 - cuDNN 8.2.1 (from JetPack)
 - OpenCV 4.1.1 (from JetPack)
 - TensorRT 8.0.1.6 (from JetPack)
 - DOTA_devkit and shapely
 - Cython, numpy, pucuda, and tqdm


1. System prepare

    You should successfully build the system on the device, which is the first step for the model to run on it. In this step, you should complete the installation of jetpack and cmake. There are many tutorials about brushing machines on the Internet, so we skip it.

2. Install protobuf

    ```shell
    sudo apt update && sudo apt install libprotoc-dev protobuf-compiler -y
    pip install protobuf==3.0.0
    ```

3. Install DOTA_devkit

    see above

4. Install onnx-tensorrt

    see https://github.com/onnx/onnx-tensorrt/tree/release/8.0