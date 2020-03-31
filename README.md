# A generic Instance and Semantic Segmentation Starter Repo

Semantic and instance segmentation implementations that operate on VOC, COCO, & Data Science Bowl 2018 (implemented as a custom dataset). If you wish to modify this for your own dataset, use the corresponding `dsb_dataset.py` as a template.

This was developed in order to carry out instance and semantic segmentation for finding nucliei in micorscope images of blood cells in order to do blood cancer detection.

## Supported datasets

- [COCO](http://cocodataset.org)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC)
- [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018/data)
Nucleus detection in blood cells

## Supported models

- [FCN ResNet101](https://arxiv.org/abs/1411.4038)
- [DeepLabV3 ResNet101](https://arxiv.org/abs/1706.05587)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Demo Scripts

See the respective README files in `detection_ref` and `segmentation_ref`.
