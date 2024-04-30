import numpy as np
import os, sys
import matplotlib.pyplot as plt
from config import *

from scipy.spatial.transform import Rotation as Rt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull

import json

### Set label path ###
DATA_PATH = os.path.join(os.path.dirname(__file__),\
                '../data/turn_left')
# Get tracking result
res = np.load(os.path.join(os.path.dirname(__file__),\
                './ES-EOT_result.npy'), allow_pickle=True)

## Show results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ground truth extend
v = np.zeros((8, 3))
skeleton_link = [[0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]]
skeleton = [v[item] for item in skeleton_link]

# Creating Cube Objects
framed = Line3DCollection(skeleton, colors='k', linewidths=0.2, linestyles='-')
ax.add_collection3d(framed)

# Initialized point position
scatter_keypoints = ax.scatter(np.zeros(N_T), np.zeros(N_T), np.zeros(N_T), c='b')
scatrer_base = ax.scatter(np.zeros(N_T), np.zeros(N_T), np.zeros(N_T), c='r', s=5)

# Creating Detected Cube
framed_det = Line3DCollection(skeleton, colors='g', linewidths=0.8, linestyles='-')
ax.add_collection3d(framed_det)

# Set coordinate axis range
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 80)
ax.set_box_aspect((1, 1, 4))

def show(frame):
    global framed, framed_det

    ## Draw ground truth
    with open(os.path.join(DATA_PATH, 'label', '{:03d}.json'.format(frame)), 'r') as f:
        label = json.load(f)
        v = np.array(label['bbox_corners'])
        skeleton = [v[item] for item in skeleton_link]

    framed.set_segments(skeleton)

    ## Draw the estimated shape
    x_ref = res[frame]['x_ref']
    l, w = x_ref[-2:]

    pos = x_ref[:3]
    theta = x_ref[4:7]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]

    R = Rt.from_rotvec(theta).as_matrix()

    front = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 0, 1])
    left = R @ np.array([0, 1, 0])
    h = np.max(mu[:, 5]) - np.min(mu[:, 5])
    v = np.array([
        pos+front*l/2+up*h+left*w/2,
        pos+front*l/2+up*h-left*w/2,
        pos+front*l/2-left*w/2,
        pos+front*l/2+left*w/2,
        pos-front*l/2+up*h+left*w/2,
        pos-front*l/2+up*h-left*w/2,
        pos-front*l/2-left*w/2,
        pos-front*l/2+left*w/2,
    ])
    skeleton = [v[item] for item in skeleton_link]
    framed_det.set_segments(skeleton)
    
    u = (R @ u.T).T + pos
    base = (R @ base.T).T + pos

    scatter_keypoints._offsets3d = (u[:,0], u[:,1], u[:,2])
    scatter_keypoints.set_sizes(np.ones(len(u)) * 10)
    scatrer_base._offsets3d = (base[:,0], base[:,1], base[:,2])
    # scatrer_base.set_sizes(alp * 300)

    ## Car in the middle of the window
    ax.set_xlim(pos[0] - 4, pos[0] + 4)
    ax.set_ylim(pos[1] - 4, pos[1] + 4)
    ax.set_zlim(pos[2] - 4, pos[2] + 4)
    ax.set_box_aspect((1, 1, 1))
    
    return ax,

ani = animation.FuncAnimation(fig, show, frames=range(len(res)), interval=2000)

plt.show()

# ax.azim, ax.elev = -79.66314935064936, 5.649350649350639
# ani.save('animation.gif', writer='pillow', dpi=100)