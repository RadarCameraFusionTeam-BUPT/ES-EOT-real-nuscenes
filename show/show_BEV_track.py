import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as Rt
import matplotlib.pyplot as plt


CRN_det_json_path = os.path.join(os.path.dirname(__file__), \
    '../CRN/CRN_det.json')

HVDetFusion_det_json_path = os.path.join(os.path.dirname(__file__), \
    '../HVDetFusion/HVDetFusion.json')

ES_EOT_npy_path = os.path.join(os.path.dirname(__file__), \
    '../ES-EOT/ES-EOT_result.npy')

labels_dir = os.path.join(os.path.dirname(__file__), \
    '../data/turn_left/label')

radar_dir = os.path.join(os.path.dirname(__file__), \
    '../data/turn_left/radar')

es_eot_res = np.load(ES_EOT_npy_path, allow_pickle=True)
with open(CRN_det_json_path, 'r') as fp:
    CRN_res = json.load(fp)

with open(HVDetFusion_det_json_path, 'r') as fp:
    HVDetFusion_res = json.load(fp)

show_frames = np.array([0, 3, 6, 8, 10])

def get_es_eot_bottom_corner(es_eot_frame):
    x_ref = es_eot_frame['x_ref']
    l, w = x_ref[-2:]

    pos = x_ref[:3]
    theta = x_ref[4:7]

    R = Rt.from_rotvec(theta).as_matrix()

    mu = es_eot_frame['mu']
    front = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 0, 1])
    left = R @ np.array([0, 1, 0])
    h = np.max(mu[:, 5]) - np.min(mu[:, 5])

    es_eot_bottom_corner = np.array([
        pos+front*l/2+left*w/2,
        pos+front*l/2-left*w/2,
        pos-front*l/2-left*w/2,
        pos-front*l/2+left*w/2
    ])
    return es_eot_bottom_corner

def get_CRN_bottom_corner(pos, R, s):
    front = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 0, 1])
    left = R @ np.array([0, 1, 0])

    w, l, h = s

    corner_point = np.array([
        pos+front*l/2+left*w/2,
        pos+front*l/2-left*w/2,
        pos-front*l/2-left*w/2,
        pos-front*l/2+left*w/2
    ])

    return corner_point

def get_HVDetFusion_bottom_corner(corners):
    assert len(corners) == 8
    return np.array(corners)[[0, 3, 7, 4]]


def plot_rec(bottom, c, ax, l=None):
    n = len(bottom)
    for i in range(n):
        if i == 0:
            ax.plot([-bottom[i, 1], -bottom[(i+1)%n, 1]], [bottom[i, 0], bottom[(i+1)%n, 0]], color=c, label=l)
        else:
            ax.plot([-bottom[i, 1], -bottom[(i+1)%n, 1]], [bottom[i, 0], bottom[(i+1)%n, 0]], color=c)

gt_bottom = []
es_eot_bottom = []
CRN_bottom = []
HVDetFusion_bottom = []
radar_point = []

for fname in sorted(os.listdir(labels_dir)):
    with open(os.path.join(labels_dir, fname), 'r') as fp:
        label = json.load(fp)
    camera_to_ego_rotation = np.array(label['camera_to_ego_rotation'])
    camera_to_ego_translation = np.array(label['camera_to_ego_translation'])

    camera_to_ego_mat = Rt.from_quat(camera_to_ego_rotation).as_matrix()
    gt_bottom_corner = np.array(label['bottom_corner'])
    gt_bottom_corner = (camera_to_ego_mat @ gt_bottom_corner.T).T + camera_to_ego_translation
    gt_bottom.append(gt_bottom_corner[:, :2])

    fname_no_ext = os.path.splitext(fname)[0]
    idx = int(fname_no_ext)
    es_eot_frame = es_eot_res[idx]

    es_eot_bottom_corner = get_es_eot_bottom_corner(es_eot_frame)
    es_eot_bottom_corner = (camera_to_ego_mat @ es_eot_bottom_corner.T).T + camera_to_ego_translation
    es_eot_bottom.append(es_eot_bottom_corner[:, :2])
    
    CRN_bottom_corner = get_CRN_bottom_corner(CRN_res['pos'][idx], CRN_res['rot'][idx], CRN_res['size'][idx])
    CRN_bottom.append(CRN_bottom_corner[:, :2])

    HVDetFusion_bottom_corner = get_HVDetFusion_bottom_corner(HVDetFusion_res['corners'][idx])
    HVDetFusion_bottom.append(HVDetFusion_bottom_corner[:, :2])

    with open(os.path.join(radar_dir, fname_no_ext+'.json'), 'r') as fp:
        radar = json.load(fp)
    radar = np.atleast_2d(radar)
    radar = (camera_to_ego_mat @ radar.T).T + camera_to_ego_translation

    radar_point.append(radar)

gt_bottom = np.stack(gt_bottom)
es_eot_bottom = np.stack(es_eot_bottom)
CRN_bottom = np.stack(CRN_bottom)
HVDetFusion_bottom = np.stack(HVDetFusion_bottom)

gt_mid = np.mean(gt_bottom, axis=1)

es_eot_mid = np.mean(es_eot_bottom, axis=1)
CRN_mid = np.mean(CRN_bottom, axis=1)
HVDetFusion_mid = np.mean(HVDetFusion_bottom, axis=1)

gt_size = np.stack([
        np.linalg.norm(gt_bottom[:, 0, :] - gt_bottom[:, 1, :], axis=1), 
        np.linalg.norm(gt_bottom[:, 1, :] - gt_bottom[:, 2, :], axis=1)
    ], axis=1)

es_eot_size = np.stack([
        np.linalg.norm(es_eot_bottom[:, 0, :] - es_eot_bottom[:, 1, :], axis=1), 
        np.linalg.norm(es_eot_bottom[:, 1, :] - es_eot_bottom[:, 2, :], axis=1)
    ], axis=1)

CRN_size = np.stack([
        np.linalg.norm(CRN_bottom[:, 0, :] - CRN_bottom[:, 1, :], axis=1), 
        np.linalg.norm(CRN_bottom[:, 1, :] - CRN_bottom[:, 2, :], axis=1)
    ], axis=1)

HVDetFusion_size = np.stack([
        np.linalg.norm(HVDetFusion_bottom[:, 0, :] - HVDetFusion_bottom[:, 1, :], axis=1), 
        np.linalg.norm(HVDetFusion_bottom[:, 1, :] - HVDetFusion_bottom[:, 2, :], axis=1)
    ], axis=1)

#### calculate metrics ####
print('ATE:')
print('\tES-EOT: ', end='')
print(np.mean(np.linalg.norm(es_eot_mid-gt_mid, axis=1)))
print('\tCRN: ', end='')
print(np.mean(np.linalg.norm(CRN_mid-gt_mid, axis=1)))
print('\tHVDetFusion: ', end='')
print(np.mean(np.linalg.norm(HVDetFusion_mid-gt_mid, axis=1)))

intersection_es_eot = np.min(np.vstack([gt_size[:, 0], es_eot_size[:, 0]]), axis=0) * np.min(np.vstack([gt_size[:, 1], es_eot_size[:, 1]]), axis=0)
union_es_eot = np.prod(gt_size, axis=1) + np.prod(es_eot_size, axis=1)

intersection_CRN = np.min(np.vstack([gt_size[:, 0], CRN_size[:, 0]]), axis=0) * np.min(np.vstack([gt_size[:, 1], CRN_size[:, 1]]), axis=0)
union_CRN = np.prod(gt_size, axis=1) + np.prod(CRN_size, axis=1)

intersection_HVDetFusion = np.min(np.vstack([gt_size[:, 0], HVDetFusion_size[:, 0]]), axis=0) * np.min(np.vstack([gt_size[:, 1], HVDetFusion_size[:, 1]]), axis=0)
union_HVDetFusion = np.prod(gt_size, axis=1) + np.prod(HVDetFusion_size, axis=1)

print('ASE:')
print('\tES-EOT: ', end='')
print(np.mean(1-intersection_es_eot/(union_es_eot-intersection_es_eot)))
print('\tCRN: ', end='')
print(np.mean(1-intersection_CRN/(union_CRN-intersection_CRN)))
print('\tHVDetFusion: ', end='')
print(np.mean(1-intersection_HVDetFusion/(union_HVDetFusion-intersection_HVDetFusion)))


#### show BEV in picture ####
fig, ax = plt.subplots()

for i, frame in enumerate(show_frames):
    gt_b, es_eot_b, CRN_b, HVDetFusion_b = gt_bottom[frame], es_eot_bottom[frame], CRN_bottom[frame], HVDetFusion_bottom[frame]
    if i==0:
        plot_rec(gt_b, 'red', ax, 'Ground Truth')
        plot_rec(es_eot_b, 'blue', ax, 'ES-EOT (CTRV+ES)')
        plot_rec(CRN_b, 'green', ax, 'CRN')
        plot_rec(HVDetFusion_b, 'orange', ax, 'HVDetFusion')
        ax.scatter(-radar_point[frame][:, 1], radar_point[frame][:, 0], color='blue', s=10, label='Radar Points')
    else:
        plot_rec(gt_b, 'red', ax)
        plot_rec(es_eot_b, 'blue', ax)
        plot_rec(CRN_b, 'green', ax)
        plot_rec(HVDetFusion_b, 'orange', ax)
        ax.scatter(-radar_point[frame][:, 1], radar_point[frame][:, 0], color='blue', s=10)

ax.plot(-gt_mid[:, 1], gt_mid[:, 0], color='black', marker='*', linestyle='--', linewidth=1, label='Trajectory')

plt.xlim(-15, 20)
plt.ylim(10, 50)
ax.set_aspect(1)
ax.legend()

# plt.grid(True)

# plt.show()

plt.savefig('BEV_picture.jpg', dpi=640, bbox_inches='tight')