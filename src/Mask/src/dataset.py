import os
import numpy as np
import torch
from PIL import Image

class WildParkMaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path, transforms):
        self.img_path = img_path
        self.ann_path = ann_path
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(img_path)))
        print(len(self.imgs))