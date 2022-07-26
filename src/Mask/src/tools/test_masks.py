import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_image_from_id(data, path, image_id):
    filename = ''
    for image in data:
        if image['id'] == image_id:
            filename = image['file_name']
            break
    
    if filename == '':
        print(f'ERROR: File with id: {image_id} not found!')
        exit()
    
    return cv2.imread(os.path.join(path, filename))

split = 'train'
img_dest = f'/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/{split}/'
ann_dest = f'/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/{split}.json'

with open(ann_dest) as f:
    ann_data = json.load(f)

for i in range(len(ann_data['annotations'])):
    ann = ann_data['annotations'][i]

    x_val = np.array(ann['segmentation'][::2]).reshape((len(ann['segmentation'][::2]), 1))
    y_val = np.array(ann['segmentation'][1::2]).reshape((len(ann['segmentation'][::2]), 1))

    points = np.concatenate((x_val, y_val), axis=1)
    img = get_image_from_id(data=ann_data['images'], path=img_dest, image_id=ann['image_id'])
    
    plt.imshow(img)
    print(i)
    for point in points:
        plt.scatter(point[0], point[1])
    
    plt.show()