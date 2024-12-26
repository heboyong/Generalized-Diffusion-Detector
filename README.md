<div align="center">
    <h2>
        Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection
    </h2>
</div>
<br>

## Introduction

This repository is the code implementation of the paper **Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection** , which is based on the [MMDetection](https://github.com/open-mmlab/mmdetection) project.

    Domain generalization (DG) for object detection aims to enhance detectors' performance in unseen scenarios. This task remains challenging due to complex variations in real-world applications. Recently, diffusion models have demonstrated remarkable capabilities in diverse scene generation, which inspires us to explore their potential for improving DG tasks. Instead of generating images, our method extracts multi-step intermediate features during the diffusion process to obtain domain-invariant features for generalized detection. Furthermore, we propose an efficient knowledge transfer framework that enables detectors to inherit the generalization capabilities of diffusion models through feature and object-level alignment, without increasing inference time. We conduct extensive experiments on six challenging DG benchmarks. The results demonstrate that our method achieves substantial improvements of 14.0% mAP over existing DG approaches across different domains and corruption types. Notably, our method even outperforms most domain adaptation methods without accessing any target domain data. Moreover, the diffusion-guided detectors show consistent improvements of 15.9% mAP on average compared to the baseline. Our work aims to present an effective approach for domain-generalized detection and provide potential insights for robust visual recognition in real-world scenarios.

## Installation
### Requirements
- Linux system, Windows is not tested
- Python 3.8+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.0.0
- CUDA 11.7 or higher, recommended 11.8
- MMCV 2.0 or higher, recommended 2.1.0
- MMDetection 3.0 or higher, recommended 3.3.0
- diffusers 0.30.0 or higher, recommended 0.30.0
### Environment Installation

It is recommended to use conda for installation. The following commands will create a virtual environment named `GDD` and install PyTorch and MMCV. In the following installation steps, the default installed CUDA version is **11.8**. 
If your CUDA version is not 11.8, please modify it according to the actual situation.
Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

**Step 1**: Create a virtual environment named `GDD` and activate it.

```shell
conda create -n GDD python=3.10 -y
conda activate GDD
```

**Step 2**: Install [PyTorch2.x](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3**: Install [MMDetection-3.x](https://mmdetection.readthedocs.io/en/latest/get_started.html).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.1.0"
mim install mmdet=3.3.0
```

**Step 4**: Prepare for [Stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with diffusers.

```shell
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```
And you should move the **Stable-diffusion-1.5** to the same dir as our **GDD**. Then:

```shell
pip install diffusers==0.30.0
```
The configuration steps for (SD-2.1, SD-3-M) follow the same procedure as previously described.

## Dataset Preparation

### Cross-domian datasets for DG detection

- Image and annotation download link: [Cityscapes, FoggyCityscapes, RainyCityscapes](https://www.cityscapes-dataset.com).
- Image and annotation download link: [BDD 100k](https://bdd-data.berkeley.edu/).
- Image and annotation download link: [SIM10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix).
- Image and annotation download link: [VOC 07+12](http://host.robots.ox.ac.uk/pascal/VOC/).
- Image and annotation download link: [Clipart, Comic, Watercolor](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).
- Image and annotation download link: [Diverse Weather Benchmark](https://github.com/AmingWu/Single-DGOD).

## Code of our GDD

**Important code directories**：

- `DG`：The root directory of our config file for GDD.
- `DG/_base_/dg_setting`：Training and lr config of GDD.
- `DG/_base_/datasets`：Dataset config of GDD.
- `DG/Ours`：Detector config of GDD.
- [`mmdet/models/backbones/dift_encoder.py`](mmdet/models/backbones/dift_encoder.py)：Code and setting of  diffusion backbone.
- [`mmdet/models/detectors/Z_domain_detector.py`](mmdet/models/detectors/Z_domain_detector.py)：Main code of GDD.
- [`mmdet/datasets/transforms/albu_domain_adaption.py`](mmdet/datasets/transforms/albu_domain_adaption.py)：Domain augmentation code.

## Model Training
<!-- 
### Diffusion detector training -->
The models are trained for 20,000 steps on two 3090 GPUs, with a batch size of 16 (For Diverse Weather Benchmark, we use eight 3090 GPUs with a total batch size of 16). 
If your settings are different from ours, please modify the training steps and default learning rate settings in [training config](DG/_base_/dg_setting).
<!-- Or You can use the trained models that we provide [google drive](https://drive.google.com/drive/folders/1_I1nXXdgL8aaoT-XKQ9FNX2A8nPkmR1Z) -->

#### Multi-gpu Training
```shell
sh ./tools/dist_train.sh ${CHECKPOINT_FILE} ${GPU_NUM}  # CHECKPOINT_FILE is the configuration file you want to use, GPU_NUM is the number of GPUs used
```
For example:
```shell
sh ./tools/dist_train.sh DG/Ours/cityscapes/diffusion_guided_detector_cityscapes.py  2  
```

## Model Testing
#### Multi-gpu Testing：

Note: Please change the code [here](DG/Ours/cityscapes/diffusion_guided_detector_cityscapes.py) ***detector.dift_model.config*** and ***detector.dift_model.pretrained_model*** as *None* before test, to prevent using settings and weights related to diffusion models.

We provide a convenient way to quickly perform DG testing.

```shell
sh ./tools/dist_test_dg.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  # CONFIG_FILE is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

## Trained models

We provide all trained models here [google drive](https://drive.google.com/drive/folders/1dHZ1p0gaKg-RahlEcAtbaqne9JI_4-iL?usp=sharing) and [baidu link](https://pan.baidu.com/s/1iYllO-xIrw7rTElBFyrbPg?pwd=kbe2) (code: <font color=Red>kbe2</font>), each corresponding to its respective config file in [`DG/Ours`](DG/Ours).

