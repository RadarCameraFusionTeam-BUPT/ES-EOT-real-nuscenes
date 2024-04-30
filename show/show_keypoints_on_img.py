import os
import numpy as np
import sys
import json
import cv2

# adjustable parameters
scenario = 'turn_left'

# set path
DATA_PATH = os.path.join(os.path.dirname(__file__),\
                '../data/turn_left')

OUT_DIR = os.path.join(DATA_PATH, 'kps_img')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

corner_id = np.array([24, 25, 26, 27])
skeleton_knots_id = np.arange(24)

for f in os.listdir(os.path.join(DATA_PATH, 'images')):
    with open(os.path.join(DATA_PATH, 'kps_json', '{}-keypoints.json'.format(f)), 'r') as fp:
        kps = json.load(fp)
    frame = cv2.imread(os.path.join(DATA_PATH, 'images', f))
    for pix in kps['keypoints']:
        if int(pix[0]) in corner_id:
            cv2.circle(frame, (int(pix[1]), int(pix[2])), 10, (255, 0, 255), -1)
        elif int(pix[0]) in skeleton_knots_id:
            cv2.circle(frame, (int(pix[1]), int(pix[2])), 10, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(OUT_DIR, '{}-keypoints.jpg'.format(f)), frame)