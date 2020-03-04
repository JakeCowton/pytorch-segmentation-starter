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

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        image_id = self.img_ids[index]

        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")

        masks = [np.array(Image.open(path.join(self.path, image_id, "masks", f))) \
                 for f in listdir(path.join(self.path, image_id, "masks"))]

        masks = np.array(masks)

        # Combine the masks
        masks = sum(masks)

        # Normalise the mask
        mask = (masks - masks.min()) / (masks.max() - masks.min())

        # Conver back to PIL.Image
        mask = Image.fromarray(mask)

        img, mask = self.transforms(img, mask)

        return img, mask
