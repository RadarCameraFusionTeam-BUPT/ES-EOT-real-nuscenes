import numpy as np
import os
from config import *

from scipy.spatial.transform import Rotation as Rt

import json

# Get ground truth and tracking result
DATA_PATH = os.path.join(os.path.dirname(__file__),\
                '../data/turn_left')
res = np.load(os.path.join(os.path.dirname(__file__),\
                './ES-EOT_result.npy'), allow_pickle=True)

metrics = {'mATE': [], 'mASE': []}

for frame in range(len(res)):
    ## Draw ground truth
    with open(os.path.join(DATA_PATH, 'label', '{:03d}.json'.format(frame)), 'r') as f:
        label = json.load(f)
        wlh = np.array(label['bbox_wlh'])
        bottom_center = np.mean(label['bottom_corner'], axis=0)

    x_ref = res[frame]['x_ref']
    pos = x_ref[:3]
    theta = x_ref[4:7]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]

    l, w = x_ref[-2:]
    wl_det = np.array([w, l])

    inter_area = np.min([wl_det[0], wlh[0]])*np.min([wl_det[1], wlh[1]])
    area1 = np.prod(wlh[:2])
    area2 = np.prod(wl_det[:2])
    IOU = inter_area / (area1 + area2 - inter_area)

    metrics['mATE'].append(np.linalg.norm(pos-bottom_center))
    metrics['mASE'].append(1-IOU)

print('mATE:', np.mean(metrics['mATE'], axis=0))
print('mASE:', np.mean(metrics['mASE'], axis=0))