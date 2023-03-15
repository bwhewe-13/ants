########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two dimensional, seven energy group problem using multiple materials 
# based off of Hou et al. "C5G7-TD Benchmark for Time-Dependent \
# Heterogeneous Neutron Transport Calculations" (2017).
# 
########################################################################

import ants
from ants.critical2d import power_iteration

import numpy as np
import matplotlib.pyplot as plt

cells_x = 102
cells_y = 102
angles = 4
groups = 7

length_x = 64.26
length_y = 64.26
delta_x = np.repeat(length_x / cells_x, cells_x)
delta_y = np.repeat(length_y / cells_y, cells_y)

edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

bc = [0, 0]

params = {
            "cells_x": cells_x, 
            "cells_y": cells_y, 
            "angles": angles,
            "groups": groups, 
            "materials": 5, 
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 2, 
            "bc_x": bc, 
            "bcdim_x": 0, 
            "bc_y": bc, 
            "bcdim_y": 0, 
            "angular": False
        }

angle_x, angle_w = ants._angle_x(params)
materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
medium_map = ants._medium_map(materials, edges_x)

xs_total = np.array([[1.0], [1.0]])
xs_scatter = np.array([[[0.3]], [[0.9]]])
xs_fission = np.array([[[0.0]], [[0.0]]])

external = ants.externals("mms-05", (cells, angles), \
                          centers_x=centers_x, angle_x=angle_x).flatten()
boundary = ants.boundaries("mms-05", (2, angles), [0, 1], \
                           angle_x=angle_x).flatten()

flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary, medium_map, delta_x, angle_x, angle_w, \
                        params)


exact = mms.solution_mms_05(centers_x, angle_x)

colors = sns.color_palette("hls", angles)

fig, ax = plt.subplots()
for n in range(angles):
    ax.plot(centers_x, flux[:,n,0], color=colors[n], alpha=0.6, \
            label="Angle {}".format(n))
    ax.plot(centers_x, exact[:,n], color=colors[n], ls=":")
ax.plot([], [], c="k", label="Approximate")
ax.plot([], [], c="k", ls=":", label="Analytical")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Angular Flux")
ax.set_title("Manufactured Solutions")
plt.show()
