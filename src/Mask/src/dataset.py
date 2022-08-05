import os
from cv2 import idct
import torch
import json
import numpy as np
import cv2
from PIL import Image
import transforms as T

class WildParkMaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path, split, transforms):
        self.split = split
        self.img_path = os.path.join(img_path, f'{split}/')
        self.ann_path = os.path.join(ann_path, f'{split}.json')
        self.transforms = transforms

        with open(self.ann_path) as f:
            self.ann_data = json.load(f)
        
        self.imgs = self.ann_data['images']
        self.anns = self.ann_data['annotations']
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.imgs[idx]['file_name'])
        img = Image.open(image_path)    
        width, height = img.size

        boxes, masks = self.__get_anns_of_image(self.imgs[idx]['id'], width, height)

        num_objs = len(masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __get_anns_of_image(self, idx, width, height):
        masks = []
        boxes = []
        annotations = []

        for annotation in self.anns:
            if annotation['image_id'] == idx:
                annotations.append(annotation.copy())

        for ann in annotations:
            mask = np.zeros((width, height, 1), np.uint8)

            x_val = np.array(ann['segmentation'][::2]).reshape((len(ann['segmentation'][::2]), 1))
            y_val = np.array(ann['segmentation'][1::2]).reshape((len(ann['segmentation'][::2]), 1))

            points = np.concatenate((x_val.astype(int), y_val.astype(int)), axis=1)
            mask = cv2.drawContours(mask, [points], 0, (1), thickness=cv2.FILLED)
            

            box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
            masks.append(mask)
            boxes.append(box)

        return boxes, masks
        
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)