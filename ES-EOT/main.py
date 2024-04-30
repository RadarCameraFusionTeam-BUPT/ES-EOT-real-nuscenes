from config import *
from FuncTools import *
import numpy as np
import time
import os, sys
import json

ROOT_PATH = os.path.join(os.path.dirname(__file__), '../')
if not ROOT_PATH in sys.path:
    sys.path.append(ROOT_PATH)
import car_model

DATA_PATH = os.path.join(os.path.dirname(__file__),\
                '../data/turn_left')

idx_list = sorted(os.listdir(os.path.join(DATA_PATH, 'images')))
frame_name = [os.path.splitext(idx)[0] for idx in idx_list]

# Prior
p = np.array([-3.960746805511638, 0.9412338893167501, 12.753004733627492])
theta = np.array([1.33118194e+00, -9.05090221e-01,  8.59973219e-01])

x_ref = np.array([*p, 10, *theta, 0, 0, 0, 4.625, 2.019])

dx = np.zeros(len(x_ref))
P = np.eye(len(x_ref))

mu = np.random.normal(0, 0.1, (N_T, 9))
mu[:, 3:6] = car_model.keypoints
Sigma = np.tile(np.identity(mu.shape[1]), (N_T, 1, 1))

# Concatenate all state
Theta = State(x_ref, dx, P, mu, Sigma)

res = []

for frame in frame_name:
    with open(os.path.join(DATA_PATH, 'radar', '{}.json'.format(frame)), 'r') as f:
        z_r = json.load(f)
        z_r = np.asarray(z_r)
    with open(os.path.join(DATA_PATH, 'kps_json', '{}.jpg-keypoints.json'.format(frame)), 'r') as f:
        keypoints_det = json.load(f)

        z_c = keypoints_det['keypoints']
        z_c = np.asarray(z_c, dtype=np.float64)

    print(frame)

    Theta = update(Theta, z_r, z_c)

    now = dict()
    now['x_ref'] = Theta.x_ref.copy()
    now['P'] = Theta.P.copy()
    now['mu'] = Theta.mu.copy()
    now['Sigma'] = Theta.Sigma.copy()
    res.append(now)

    Theta = predict(Theta)
    

np.save(os.path.join(os.path.dirname(__file__),\
        'ES-EOT_result.npy'), res, allow_pickle = True)