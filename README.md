# SplitMixer: Fat Trimmed From MLP-like Models
PyTorch implementation of the SplitMixer MLP model for visual recognition

by [Ali Borji](https://github.com/aliborji) and [Sikun Lin](https://github.com/sklin93)

Arxiv link: TBD





## Code overview

The most important code is in [`splitmixer.py`](https://github.com/aliborji/splitmixer/blob/main/pytorch-image-models/timm/models/splitmixer.py). We trained SplitMixers (on ImageNet) using the `timm` framework, which we copied from [here](http://github.com/rwightman/pytorch-image-models).

For CIFAR-{10,100} trainings or standalone model definitions, please refer to the [cifar notebook](https://github.com/aliborji/splitmixer/blob/main/splitmixer-cifar.ipynb).

Inside `pytorch-image-models`, we have made the following modifications: 

- Added ConvMixers
  - added `timm/models/splitmixer.py`
  - modified `timm/models/__init__.py`


## Evaluation


### CIFAR-10

Patch Size p=2, Kernel Size k=5


| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|  0.60 | 152.6 | 94.17 |
|SplitMixer-I 256/8|  0.28 | 71.8 | 93.91 |
|SplitMixer-II 256/8|  0.17 | 46.2 | 92.25 |
|SplitMixer-III 256/8|  0.17 | 79.8 | 92.52 |
|SplitMixer-IV 256/8|  0.31 | 79.8 | 93.38 |



### CIFAR-100

Patch Size p=2, Kernel Size k=5


| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|  0.62 | 152.6 | 73.92 |
|Splitixer-I 256/8|  0.30 | 71.9 | 72.88 |
|SplitMixer-II 256/8|  0.19 | 46.2 | 70.44 | 
|SplitMixer-III 256/8|  0.19 | 79.8 | 70.89 |
|SplitMixer-IV 256/8|  0.32 | 79.8 | 71.75 |



### Flowers102

Patch Size p=7, Kernel Size k=7

| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|   0.70  |696 | 60.47 |
|Splitixer-I 256/8| 0.34 |331 | 62.03|
|SplitMixer-II 256/8|  0.24 | 229 | 59.33 |
|SplitMixer-III 256/8|  0.24 | 363 | 59.00 |
|SplitMixer-IV 256/8|   0.37 | 363 | 61.51 |



### Foods101

Patch Size p=7, Kernel Size k=7


| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|   0.70   | 696  |  74.59 | 
|Splitixer-I 256/8|  0.34  |  331  |  73.56  | 
|SplitMixer-II 256/8 | 0.24  |  229  |  71.74   | 
|SplitMixer-III 256/8|  0.24  |  363  |  72.78  | 
|SplitMixer-IV 256/8|   0.37  |  363  |  72.92  | 



### ImageNet

Stay Tuned!





## Citation

If you use this code in your research, please cite this project.

```
@inproceedings{borji2022SplitMixer,
title={SplitMixer: Fat Trimmed From MLP-like Models},
author={Ali Borji and Sikun Lin},
booktitle={IArxiv},
year={2022},
url={https://openreview.net/forum?id=H1ebhnEYDH}
}```

