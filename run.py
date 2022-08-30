# general imports
import torch
import cv2

# Mask R-CNN imports
from Mask.src.model import get_model_instance_segmentation

# CenterTrack imports
import sys
CENTERTRACK_PATH = '/home/smuuts/Documents/uni/PG/CenterTrack/src/lib'
sys.path.insert(0, CENTERTRACK_PATH)
from detector import Detector
from opts import opts

# initialize detector
opt = opts().init()
detector = Detector(opt)

VIDEO_PATH = '/home/smuuts/Documents/uni/PG/CenterTrack/videos/RW_2022_12_16_VD_00008.mov'
cam = cv2.VideoCapture(VIDEO_PATH)

SAVE_VIDEO = False

if SAVE_VIDEO == True:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/{}.mp4'.format('tracked'),fourcc, 30, (
        opt.video_w, opt.video_h))

# colors for masks and boxes
colors = [
    (200, 0, 0),
    (0, 200, 0),
    (0, 255, 255),
    (0, 0, 200),
    (255, 255, 0),
    (255, 0, 255),
    (128, 255, 0),
]

# helper function to draw mask on given image
def draw_mask(img, mask, alpha, color, beta):
    img = img.copy()
    
    for i in range(len(color)):
        img[i] = img[i] * alpha + mask * color[i] * beta
    return img

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

while True:
    _, img = cam.read()
    if img is None:
        if SAVE_VIDEO == True:
            out.release()
        print('done!')
        sys.exit(0)
        
    ret = detector.run(img)
    results = ret['results']

    i = 0
    for track in results:
        box = [int(val) for val in track['bbox']]

        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors[track['class']], thickness=2)
        img = cv2.putText(img, '{}'.format(track['tracking_id']), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[track['class']], thickness=2)

        i+=1
    
    cv2.imshow('tracked', img)
    cv2.waitKey(1)

    if SAVE_VIDEO == True:
        out.write(img)