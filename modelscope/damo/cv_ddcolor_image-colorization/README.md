---
tasks:
  - image-colorization
widgets:
  - task: image-colorization
    inputs:
      - type: image
    examples:
      - name: 1
        inputs:
          - name: image
            data: git://resources/demo.jpg
      - name: 2
        inputs:
          - name: image
            data: git://resources/demo2.jpg
      - name: 3
        inputs:
          - name: image
            data: git://resources/demo3.jpg
    inferencespec:
      cpu: 4
      memory: 16000
      gpu: 1
      gpu_memory: 16000
model-type:
  - ddcolor
domain:
  - cv
frameworks:
  - pytorch
backbone:
  - unet
metrics:
  - FID
  - Colorfulness
  - PSNR
customized-quickstart: False
finetune-support: True
license: Apache License 2.0
tags:
  - ICCV 2023
  - Alibaba
  - Image colorization
  - Old photo restoration
datasets:
  evaluation:
  - damo/imagenet-val5k-image
  test:
  - modelscope/image-colorization-dataset
studios:
  - damo/old_photo_restoration
---

# DDColor 图像上色模型

该模型为黑白图像上色模型，输入一张黑白图像，实现端到端的全图上色，返回上色处理后的彩色图像。

[**English Version**](#ddcolor-for-image-colorization) **|** [**中文版本**](#ddcolor-图像上色模型)

## [Paper](https://arxiv.org/abs/2212.11613) ｜ [Github](https://github.com/piddnad/DDColor)

## 模型描述

DDColor 是最新的 SOTA 图像上色算法，能够对输入的黑白图像生成自然生动的彩色结果。

算法整体流程如下图，使用 UNet 结构的骨干网络和图像解码器分别实现图像特征提取和特征图上采样，并利用 Transformer 结构的颜色解码器完成基于视觉语义的颜色查询，最终聚合输出彩色通道预测结果。

![ofa-image-caption](./resources/ddcolor_arch.jpg)

## 模型期望使用方式和适用范围

该模型适用于多种格式的图像输入，给定黑白图像，生成上色后的彩色图像；给定彩色图像，将自动提取灰度通道作为输入，生成重上色的图像。

### 如何使用

在 ModelScope 框架上，提供输入图片，即可以通过简单的 Pipeline 调用来使用图像上色模型。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, 
                       model='damo/cv_ddcolor_image-colorization')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg'
result = img_colorization(img_path)
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差

- 本算法模型使用自然图像数据集进行训练，对于分布外场景（例如漫画等）可能产生不恰当的上色结果；
- 对于低分辨率或包含明显噪声的图像，算法可能无法得到理想的生成效果。

## 训练数据介绍

模型使用公开数据集 [ImageNet](https://www.image-net.org/) 训练，其训练集包含 128 万张自然图像。

## 模型训练流程

### 预处理

将 ImageNet 数据集的 RGB 彩色图像转为 Lab 色彩空间，并提取其中的 L 通道，得到相应的灰度图片。

### 模型训练代码
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.dataset_cls.custom_datasets.image_colorization import \
    ImageColorizationDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_ddcolor_image-colorization'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_train = MsDataset.load(
    'imagenet-val5k-image',
    namespace='damo',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
dataset_val = MsDataset.load(
    'imagenet-val5k-image',
    namespace='damo',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

dataset_train = ImageColorizationDataset(
    dataset_train, config.dataset, is_train=True)
dataset_val = ImageColorizationDataset(
    dataset_val, config.dataset, is_train=False)

kwargs = dict(
    model=model_id,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
trainer.train()
```

## 数据评估及结果

本算法主要在 [ImageNet](https://www.image-net.org/) 和 [COCO-Stuff](https://github.com/nightrome/cocostuff)上测试。

| Val Name          | FID  | Colorfulness |
|:-----------------:|:----:|:------------:|
| ImageNet (val50k) | 3.92 | 38.26        |
| ImageNet (val5k)  | 0.96 | 38.65        |
| COCO-Stuff        | 5.18 | 38.48        |

## 引用

如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：

```
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```

# DDColor for Image Colorization

[**English Version**](#ddcolor-for-image-colorization) **|** [**中文版本**](#ddcolor-图像上色模型)

## [Paper](https://arxiv.org/abs/2212.11613) ｜ [Github](https://github.com/piddnad/DDColor)

## Model Description

DDColor is the latest state-of-the-art (SOTA) image colorization algorithm, capable of generating natural and vivid colored results from input black-and-white images.

The overall workflow of the algorithm is shown in the figure below, a UNet structure backbone network and an image decoder is used to extract image features and upsample feature maps, respectively. A Transformer-structured color decoder completes the color query inference based on visual semantics, ultimately aggregating and outputting predicted color channels.

![ofa-image-caption](./resources/ddcolor_arch.jpg)

## Expected Usage and Application Scope of the Model

This model is applicable to a variety of image formats, generating a colored image from a given black-and-white image; for a given color image, it will automatically extract the grayscale channel as input to generate a re-colorized image.

### How to Use

With the ModelScope library, providing an input picture, you can use the image colorization model through a simple pipeline call.

#### Code Example

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, 
                       model='damo/cv_ddcolor_image-colorization')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg'
result = img_colorization(img_path)
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### Model Limitations and Potential Biases

- This algorithm model is trained using a natural image dataset, which may produce inappropriate colorization results for out-of-distribution scenes (such as comics, etc.);
- For low-resolution images or those containing noticeable noise, the algorithm may not achieve the desired generative effects.

## Training Data Introduction

The model is trained on the publicly available dataset [ImageNet](https://www.image-net.org/), which contains 1.28 million natural images in its training set.

## Model Training Process

### Preprocessing

Convert the RGB color images in the ImageNet dataset to the Lab color space, and extract the L channel to obtain corresponding grayscale images.

### Model Training Code Using ModelScope

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.dataset_cls.custom_datasets.image_colorization import \
    ImageColorizationDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile


tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_ddcolor_image-colorization'
cache_path = snapshot_download(model_id)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_train = MsDataset.load(
    'imagenet-val5k-image',
    namespace='damo',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
dataset_val = MsDataset.load(
    'imagenet-val5k-image',
    namespace='damo',
    subset_name='default',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

dataset_train = ImageColorizationDataset(
    dataset_train, config.dataset, is_train=True)
dataset_val = ImageColorizationDataset(
    dataset_val, config.dataset, is_train=False)

kwargs = dict(
    model=model_id,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
trainer.train()
```

## Data Evaluation and Results

The algorithm was mainly tested on [ImageNet](https://www.image-net.org/) and [COCO-Stuff](https://github.com/nightrome/cocostuff).

| Val Name          | FID  | Colorfulness |
|:-----------------:|:----:|:------------:|
| ImageNet (val50k) | 3.92 | 38.26        |
| ImageNet (val5k)  | 0.96 | 38.65        |
| COCO-Stuff        | 5.18 | 38.48        |

## Citation

If you find this model helpful, please consider citing the following paper:

```
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```