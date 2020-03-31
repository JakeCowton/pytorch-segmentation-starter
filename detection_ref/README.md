# Object detection reference training scripts

## Data Science Bowl

This will train MaskRCNN on DSB18 and store them in the runs folder. Other arguments for this can be found at the end of `train.py`.

```
python train.py --dataset dsb --imageset stage1_train --data-path /path/to/data-science-bowl-2018 --model maskrcnn_resnet50_fpn --output-dir ./runs/maskrcnn -j 50
```

## Benchmark datasets

### Faster R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

