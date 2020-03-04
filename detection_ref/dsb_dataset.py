from os import path, listdir
import math

import numpy as np
from PIL import Image

from skimage.measure import label, regionprops

import torch
from torch.utils.data import Dataset

class DSB(Dataset):

    def __init__(self, root_path="/home/jake/data/data-science-bowl-2018",
                 image_set="stage1_train", transforms=None):
        super(DSB, self).__init__()
        self.root = root_path
        self.image_set = image_set

        self.path = path.join(self.root, self.image_set)

        self.img_ids = [f for f in sorted(listdir(self.path))]

        self.transforms = transforms

    def get_height_and_width(self, index):
        image_id = self.img_ids[index]
        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")
        return img.height, img.width

    def generate_bbox_from_mask(self, mask):
        label_img = label(mask, connectivity=1)
        try:
            props = regionprops(label_img)[0]
        except Exception:
            print("Region not found")
            return
        return props.bbox

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        image_id = self.img_ids[index]

        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")

        boxes = []
        labels = []
        masks = []
        area = []
        is_crowd = []

        mask_path = path.join(self.path, image_id, "masks")

        for f in listdir(mask_path):
            # Load mask
            mask = np.array(Image.open(path.join(mask_path, f)))
            # Normalise mask
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            box = self.generate_bbox_from_mask(mask)
            if box:
                boxes.append(box)
                labels.append(1)
                masks.append(np.array(mask))
                area.append(np.count_nonzero(mask == 1))
                is_crowd.append(0)
            else:
                continue

        target = {
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels),
            "masks": torch.tensor(masks),
            "image_id": torch.tensor([index]),
            "area": torch.tensor(area),
            "is_crowd": torch.tensor(is_crowd)
        }

        img, target = self.transforms(img, target)

        return img, target
