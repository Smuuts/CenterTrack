import torch
from model import get_model_instance_segmentation
from dataset import WildParkMaskDataset, get_transform
import cv2

def draw_mask(img, mask, alpha, color, beta):
    img = img.copy()
    for i in range(len(color)):
        img[i] = img[i] * alpha + mask[0] * color[i] * beta
    
    return img

img_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/'
ann_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/'

dataset_test = WildParkMaskDataset(img_path, ann_path, 'test', get_transform(train=False))

model_path = '/home/smuuts/Documents/uni/PG/CenterTrack/src/Mask/models/test.pth'

model = get_model_instance_segmentation(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

img, _ = dataset_test[1]
predictions = model([img])

img = img.detach().numpy() * 255
mask1 = predictions[0]['masks'][0].detach().numpy()
# mask2 = predictions[0]['masks'][1].detach().numpy()

new_img = draw_mask(img, mask1, 1, (255, 0, 0), 0.6)
# new_img = draw_mask(new_img, mask2, 1, (0, 255, 0), 0.6)
new_img = new_img.transpose(1, 2, 0)

cv2.imwrite('test.png', new_img)