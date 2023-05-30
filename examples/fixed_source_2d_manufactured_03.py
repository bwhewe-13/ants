########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
########################################################################

import ants
from ants.fixed2d import source_iteration
from ants.utils import manufactured_2d as mms

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cells", type=int)
args = parser.parse_args()

cells_x = args.cells
cells_y = args.cells
angles = angles1 = 4
groups = 1

length_x = 2.
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

length_y = 2.
delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.linspace(0, length_y, cells_y+1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

bc = [0, 0]

params = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles, 
            "groups": groups, 
            "materials": 1,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 3,
            "bc_x": bc,
            "bcdim_x": 3,
            "bc_y": bc,
            "bcdim_y": 3,
            "angular": False
        }

xs_total = np.array([[1.0]])
xs_scatter = np.array([[[0.5]]])
xs_fission = np.array([[[0.0]]])

angle_x, angle_y, angle_w = ants._angle_xy(params)
medium_map = np.zeros((cells_x * cells_y), dtype=np.int32).flatten()

angles = params["angles"]
external = np.zeros((cells_x, cells_y, angles, groups))
x, y = np.meshgrid(centers_x, centers_y)

for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
    # external[:,:,n,0] = eta + mu
    external[:,:,n,0] = 0.5 * (np.exp(-1) - np.exp(1)) \
                        + np.exp(mu) + np.exp(eta)

boundary_x = np.zeros((2, cells_y, angles, groups))
boundary_y = np.zeros((2, cells_x, angles, groups))
for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
    boundary_x[0,:,n,0] = np.exp(mu) + np.exp(eta)
    boundary_x[1,:,n,0] = np.exp(mu) + np.exp(eta)
    boundary_y[0,:,n,0] = np.exp(mu) + np.exp(eta)
    boundary_y[1,:,n,0] = np.exp(mu) + np.exp(eta)

flux = source_iteration(xs_total, xs_scatter, xs_fission, external.flatten(), \
                        boundary_x.flatten(), boundary_y.flatten(), \
                        medium_map, delta_x, delta_y, angle_x, angle_y, \
                        angle_w, params)


exact_angular = mms.solution_mms_03(centers_x, centers_y, angle_x, angle_y)
exact_scalar = np.sum(exact_angular * angle_w[None,None,:,None], axis=(2,3))

fig, ax = plt.subplots()
img = ax.imshow(abs(exact_scalar - flux[:,:,0]), cmap="rainbow", \
                origin="lower") #, vmin=mini, vmax=maxi)
fig.colorbar(img, ax=ax, label="|Exact - Approx|", format="%.02e")
ax.set_title("Scalar Flux, Cells = {}, N = {}".format(cells_x, angles1))
ax.set_xticks(np.arange(-0.5, cells_x+1)[::10])
ax.set_xticklabels(np.round(edges_x, 3)[::10])
ax.set_yticks(np.arange(-0.5, cells_y+1)[::10])
ax.set_yticklabels(np.round(edges_y, 3)[::10])
ax.set_xlabel("x Direction (cm)")
ax.set_ylabel("y Direction (cm)")

plt.show()