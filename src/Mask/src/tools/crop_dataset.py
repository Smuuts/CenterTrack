import cv2
import os
import json
from matplotlib import pyplot as plt

def get_crop_size(box, size, width, height, padding):
    new_box = [box[0]-padding, box[1]-padding, box[2]+2*padding, box[3]+2*padding]

    if new_box[2] < size:
        xDiff = size - new_box[2]
        new_box[0] -= (xDiff/2)
        new_box[2] += xDiff
    
    if new_box[3] < size:
        yDiff = size - new_box[3]
        new_box[1] -= (yDiff/2)
        new_box[3] += yDiff
    
    if new_box[0] < 0:
        new_box[0] = 0
    if new_box[1] < 0:
        new_box[1] = 0

    if (new_box[0] + new_box[2]) > width:
        new_box[0] -= ((new_box[0] + new_box[2]) - width)
    if (new_box[1] + new_box[3]) > height:
        new_box[1] -= ((new_box[1] + new_box[3]) - height)

    return new_box

def fit_box(crop_box, bbox):

    if bbox[0] > (crop_box[0] + crop_box[2]) or bbox[1] > (crop_box[1] + crop_box[3]):
        return None
    if (bbox[0] + bbox[2]) <= crop_box[0] or (bbox[1] + bbox[3]) <= crop_box[1]:
        return None

    new_box = [bbox[0] - crop_box[0], bbox[1] - crop_box[1], bbox[2], bbox[3]]

    if new_box[0] < 0:
        new_box[2] = new_box[2] + new_box[0]
        new_box[0] = 0
    if new_box[1] < 0:
        new_box[3] = new_box[3] + new_box[1]
        new_box[1] = 0
    
    xDiff = new_box[0] + new_box[2] - crop_box[2]
    yDiff = new_box[1] + new_box[3] - crop_box[3]

    if xDiff > 0:
        new_box[2] -= xDiff

    if yDiff > 0:
        new_box[3] -= yDiff

    return new_box

def get_anns_from_img(data, img_id):    
    annotations = []
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            annotations.append(ann)

    return annotations

IMG_WIDTH = 1728
IMG_HEIGHT = 1296

MIN_BOX_SIZE = 512
PADDING = 50

split = 'test'

img_path = f'/home/smuuts/Documents/uni/PG/CenterTrack/data/WildparkDataset/annotated_frames/'
ann_path = f'/home/smuuts/Documents/uni/PG/CenterTrack/data/WildparkDataset/annotations/{split}.json'

img_dest = f'/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/{split}/'
ann_dest = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/'

# create directories if necessary
if not os.path.isdir(img_dest):
    os.makedirs(img_dest)
if not os.path.isdir(ann_dest):
    os.makedirs(ann_dest)

ann_dest = os.path.join(ann_dest, f'{split}.json')

with open(ann_path) as f:
    ann_data = json.load(f)

out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'red deer'}, {'id': 2, 'name': 'fallow deer'}]}

thresh = 0

img_amt = len(ann_data['images'])
cropped_img_count = 0
cropped_ann_count = 0

# ann -> {id, category_id, image_id, track_id, bbox, segmentation}
for j in range(img_amt):
    img_data = ann_data['images'][j]

    done = j/img_amt * 100
    if (done > thresh):
        print(f'{int(done)}% done')
        thresh += 5

    # get image
    img = cv2.imread(os.path.join(img_path, img_data['file_name']))
    annotations = get_anns_from_img(ann_data, img_data['id'])

    for ann in annotations:
        # crop to bounding box
        bbox = ann['bbox']

        crop_box = get_crop_size(bbox, MIN_BOX_SIZE, IMG_WIDTH, IMG_HEIGHT, PADDING)
        cropped_img = img.copy()[int(crop_box[1]):int(crop_box[1] + crop_box[3]), int(crop_box[0]):int(crop_box[0] + crop_box[2])]

        cropped_name = '{0:06d}.png'.format(cropped_img_count)
        cv2.imwrite(os.path.join(img_dest, cropped_name), cropped_img)

        # add image info
        image_info = {'file_name': cropped_name,
                        'id': cropped_img_count,
                        'height': cropped_img.shape[0],
                        'width': cropped_img.shape[1]}

        out['images'].append(image_info)

        for annotation in annotations:
            seg = annotation['segmentation'].copy()

            for i in range(int(len(seg)/2)):
                seg[i * 2] = int(seg[i * 2] - crop_box[0])
                seg[i * 2 + 1] = int(seg[i * 2 + 1] - crop_box[1])

                seg[i * 2] = max(min(crop_box[2], seg[i * 2]), 0)
                seg[i * 2 + 1] = max(min(crop_box[3], seg[i * 2 + 1]), 0)

            xMin = min(seg[::2])
            yMin = min(seg[1::2])

            xMax = max(seg[::2])
            yMax = max(seg[1::2])

            box = [xMin, yMin, xMax-xMin, yMax-yMin]

            if box[2] <= 0 or box[3] <= 0:
                continue

            ann_info = {'id': cropped_ann_count,
                                'category_id': ann['category_id'],
                                'image_id': cropped_img_count,
                                'bbox': box,
                                'segmentation': seg,
                                'conf': 1.0}
            out['annotations'].append(ann_info)
            cropped_ann_count+=1
        
        cropped_img_count += 1

json.dump(out, open(ann_dest, 'w'))
print('done!')