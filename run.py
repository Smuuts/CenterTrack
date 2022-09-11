# general imports
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'using {device} device.')

# Mask R-CNN imports and setup
from Mask.src.model import get_model_instance_segmentation

MODEL_PATH = '/home/smuuts/Documents/uni/PG/CenterTrack/Mask/models/model_7.pth'
model = get_model_instance_segmentation(2, False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model = model.to(device)
# CenterTrack imports and setup
import sys
CENTERTRACK_PATH = '/home/smuuts/Documents/uni/PG/CenterTrack/src/lib'
sys.path.insert(0, CENTERTRACK_PATH)
from detector import Detector
from opts import opts

# initialize detector
opt = opts().init()
detector = Detector(opt)

VIDEO_PATH = '/home/smuuts/Documents/uni/PG/CenterTrack/videos/DW_2022_12_16_VD_00003.mov'
cam = cv2.VideoCapture(VIDEO_PATH)

SAVE_VIDEO = False

if SAVE_VIDEO == True:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/{}.mp4'.format('tracked'),fourcc, 30, (
        opt.video_w, opt.video_h))

# colors for masks and boxes
colors = [
    (200, 200, 0),
    (200, 200, 0),
    (200, 200, 0),
    
    (0, 0, 200),
    (255, 255, 0),
    (255, 0, 255),
    (128, 255, 0),
]

# helper function to draw mask on given image
def draw_mask(img, mask, alpha, color, beta):
    img = np.transpose(img.copy(), (2, 0, 1))

    maximum = np.ones(mask.shape) * 255
    
    for i in range(len(color)):
        img[i] = np.minimum(maximum, img[i] * alpha + mask * color[i] * beta)
    
    return np.transpose(img, (1, 2, 0))

def get_cropped_img(img, box, size, width, height, padding):
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

    if (new_box[0] + new_box[2]) > height:
        new_box[0] -= ((new_box[0] + new_box[2]) - height)
    if (new_box[1] + new_box[3]) > width:
        new_box[1] -= ((new_box[1] + new_box[3]) - width)

    cropped_img = img.copy()[int(new_box[1]):int(new_box[1]+new_box[3]), int(new_box[0]):int(new_box[0]+new_box[2])]

    return new_box, cropped_img

def compare_bbox(box1, box2):
    return sum(box1 - box2)

frame_cnt = 0
while True:
    _, img = cam.read()
    if img is None:
        if SAVE_VIDEO == True:
            out.release()
        print('done!')
        sys.exit(0)
    
    frame_cnt += 1
    ret = detector.run(img)
    results = ret['results']

    i = 0
    for track in results:
        box = [int(val) for val in track['bbox']]

        crop, cropped_img = get_cropped_img(img, [box[0], box[1], box[2] - box[0], box[3] - box[1]], 512, 1296, 1728, 50)

        cropped_img = F.to_tensor(cropped_img)
        cropped_img = cropped_img.to(device)
        
        pred = model([cropped_img])[0]

        if len(pred['boxes']) != 0:
            box_predictions = [bb.detach().cpu().numpy() for bb in pred['boxes']]
            for i in range(len(box_predictions)):
                box_predictions[i][0] = box_predictions[i][0] + crop[0]
                box_predictions[i][1] = box_predictions[i][1] + crop[1]
                box_predictions[i][2] = box_predictions[i][2] + crop[0]
                box_predictions[i][3] = box_predictions[i][3] + crop[1]

            box_scores = [compare_bbox(box, bb) for bb in box_predictions]
            match = np.argmin(box_scores)

            if box_scores[match] > 15000000:
                mask = None
            else:
                mask = pred['masks'][match].detach().cpu().numpy()
                expanded_mask = np.zeros((1296, 1728))

                expanded_mask[ int(crop[1]):int(crop[1] + crop[3]), int(crop[0]):int(crop[0] + crop[2])] = mask[0]
        else:
            mask = None

        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors[track['class']], thickness=1)
        img = cv2.putText(img, '{}'.format(track['tracking_id']), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[track['class']], thickness=2)

        if type(mask) != type(None):
            img = draw_mask(img, expanded_mask, 1, colors[track['class']], 0.23)

        i+=1
    
    cv2.imshow('tracked', img)
    cv2.waitKey(1)

    if SAVE_VIDEO == True:
        out.write(img)