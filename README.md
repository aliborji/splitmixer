# SplitMixer: Fat Trimmed From MLP-like Models
PyTorch implementation of the SplitMixer MLP model for visual recognition

by Ali Borji and Sikun Lin 

Arxiv link: TBD





### Code overview
The most important code is in `convmixer.py`. We trained ConvMixers using the `timm` framework, which we copied from [here](http://github.com/rwightman/pytorch-image-models).

### SplitMixer is integrated into the [`timm` framework itself](https://github.com/rwightman/pytorch-image-models). You can see the PR [here](https://github.com/rwightman/pytorch-image-models/pull/910).

Inside `pytorch-image-models`, we have made the following modifications: 

- Added ConvMixers
  - added `timm/models/convmixer.py`
  - modified `timm/models/__init__.py`
- Added "OneCycle" LR Schedule
  - added `timm/scheduler/onecycle_lr.py`
  - modified `timm/scheduler/scheduler.py`
  - modified `timm/scheduler/scheduler_factory.py`
  - modified `timm/scheduler/__init__.py`
  - modified `train.py` (added two lines to support this LR schedule)



## Evaluation


### CIFAR-10

Patch Size p=2, Kernel Size k=5


| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|  0.594 | 152.6 | 94.17 |
|SplitMixer-I 256/8|  0.276 | 71.8 | 93.91 |
|SplitMixer-II 256/8|  0.175 | 46.2 | 92.25 |
|SplitMixer-III 256/8|  0.175 | 79.8 | 92.52 |
|SplitMixer-IV 256/8|  0.307 | 79.8 | 93.38 |



### CIFAR-100

Patch Size p=2, Kernel Size k=5


| Model Name | Params (M) | FLOPS (M) | Acc | 
|------------|:-----------:|:----------:|:----------:|
|ConvMixer-256/8|  0.617 | 152.6 | 73.92 |
|Splitixer-I 256/8|  0.298 | 71.9 | 72.88 |
|SplitMixer-II 256/8|  0.186 | 46.2 | 70.44 | 
|SplitMixer-III 256/8|  0.186 | 79.8 | 70.89 |
|SplitMixer-IV 256/8|  0.318 | 79.8 | 71.75 |



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






## Rest!!!




You can evaluate ConvMixer-1536/20 as follows:

```
python validate.py --model convmixer_1536_20 --b 64 --num-classes 1000 --checkpoint [/path/to/convmixer_1536_20_ks9_p7.pth.tar] [/path/to/ImageNet1k-val]
```

You should get a `81.37%` accuracy.

### Training
If you had a node with 10 GPUs, you could train a ConvMixer-1536/20 as follows (these are exactly the settings we used):

```
sh distributed_train.sh 10 [/path/to/ImageNet1k] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model convmixer_1536_20 
    -b 64 
    -j 10 
    --opt adamw 
    --epochs 150 
    --sched onecycle 
    --amp 
    --input-size 3 224 224
    --lr 0.01 
    --aa rand-m9-mstd0.5-inc1 
    --cutmix 0.5 
    --mixup 0.5 
    --reprob 0.25 
    --remode pixel 
    --num-classes 1000 
    --warmup-epochs 0 
    --opt-eps=1e-3 
    --clip-grad 1.0
```


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

