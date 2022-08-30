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
        img[i] = img[i] * alpha + mask * color[i] * beta
    return img

img_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/'
ann_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/'

dataset_test = WildParkMaskDataset(img_path, ann_path, 'test', get_transform(train=False))

model_path = '/home/smuuts/Documents/uni/PG/CenterTrack/Mask/models/model_1.pth'

model = get_model_instance_segmentation(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

img, target = dataset_test[4]

predictions = model([img])


img = img.detach().cpu().numpy() * 255
i = 0


new_img = img
mask_amt = len(predictions[0]['masks'])

print(f'{mask_amt} masks predicted')
for i in range(len(predictions[0]['masks'])):
    mask = predictions[0]['masks'][i]
    mask = mask.detach().cpu().numpy()
    new_img = draw_mask(new_img, mask, 1, colors[i%len(colors)], 1)
    i+=1

new_img = new_img.transpose(1, 2, 0)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('test.png', new_img)