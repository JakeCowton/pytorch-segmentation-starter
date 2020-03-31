# Semantic segmentation reference training scripts

This folder contains reference training scripts for semantic segmentation.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

## Data Science Bowl 2018

```
python train.py --dataset dsb --imageset stage1_train --model deeplabv3_resnet101 --output-dir ./runs/deeplab -j 50 --data-path /path/to/data-science-bowl-2018


python train.py --dataset dsb --imageset stage1_train --model fcn_resnet101 --output-dir ./runs/fcn -j 50 --data-path /path/to/data-science-bowl-2018
```

## Benchmark datasets

### fcn_resnet101
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model fcn_resnet101 --aux-loss
```

### deeplabv3_resnet101
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet101 --aux-loss
```
