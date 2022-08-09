import random
import torch
import random
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as T

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            if "masks" in target:
                for i in range(len(target['masks'])):
                    
                    mask = torch.permute(target["masks"][i], (2, 0, 1))
                    target["masks"][i] = torch.permute(mask.flip(-1), (1, 2, 0))

                    box = get_bbox_from_mask(target['masks'][i])
                    target['boxes'][i] = box

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class GaussianBlur(object):
    def __init__(self, kernel, sigma) -> None:
        self.kernel = kernel
        self.sigma = sigma
    def __call__(self, image, target):
        image = F.gaussian_blur(image, self.kernel, self.__get_sigma())
        return image, target

    def __get_sigma(self):
        return random.uniform(self.sigma[0], self.sigma[1])

class RandomRotate(object):
    def __init__(self, degrees) -> None:
        self.degrees = degrees
    
    def __call__(self, image, target):
        angle = self.__get_angle()
        image = F.rotate(image, angle)

        for i in range(len(target['masks'])):
            mask = torch.permute(target['masks'][i], (2, 0, 1))
            mask = torch.permute(F.rotate(mask, angle), (1, 2, 0))
            target['masks'][i] = mask

            target['boxes'][i] = get_bbox_from_mask(mask)

        return image, target

    def __get_angle(self):
        return random.uniform(self.degrees[0], self.degrees[1])

class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, image, target):
        image = self.jitter(image)

        return image, target


def get_bbox_from_mask(mask):
    size = mask.shape
    x1 = y1 = x2 = y2 = None
    
    points = []

    for col in range(len(mask)):
        for row in range(len(mask[col])):
            if mask[col][row] >= 1:
                points.append(row)
                points.append(col)
    
    x1 = min(points[::2])
    y1 = min(points[1::2])
    x2 = max(points[::2])
    y2 = max(points[1::2])

    return torch.tensor([x1, y1, x2, y2])

