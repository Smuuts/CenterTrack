import torch
from model import get_model_instance_segmentation
from dataset import WildParkMaskDataset, get_transform
import cv2

colors = [
    (200, 0, 0),
    (0, 200, 0),
    (0, 0, 200),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
]

def draw_mask(img, mask, alpha, color, beta):
    img = img.copy()
    for i in range(len(color)):
        img[i] = img[i] * alpha + mask[0] * color[i] * beta
    
    return img

img_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/'
ann_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/'

dataset_test = WildParkMaskDataset(img_path, ann_path, 'train', get_transform(train=True))

img, target = dataset_test[11000]

img = img.detach().cpu().numpy() * 255
i = 0

new_img = img
mask_amt = len(target['masks'])
print(f'{mask_amt} masks')
for i in range(len(target['masks'])):
    mask = torch.permute(target['masks'][i].detach().cpu(), (2, 0, 1)).numpy()
    new_img = draw_mask(new_img, mask, 1, colors[i%len(colors)], 1)

    box = target['boxes'][i].detach().cpu().numpy().astype(int)
    new_img = new_img.transpose(1, 2, 0).copy()
    new_img = cv2.rectangle(new_img, (box[0], box[1]), (box[2], box[3]), color=colors[i%len(colors)], thickness=5)
    new_img = new_img.transpose(2, 0, 1).copy()

    i+=1

new_img = new_img.transpose(1, 2, 0)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('test.png', new_img)