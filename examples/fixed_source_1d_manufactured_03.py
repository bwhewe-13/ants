########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One dimensional method of manufactured solution problem with scattering
# scattering, unit external source, and an incoming boundary source 
# from (x = 0). Taken from Wang's "Application of the Method of 
# Manufactured Solutions to Verify the Method of Characteristics 
# for Reactor Analysis (2019).
# 
########################################################################

import ants
from ants.fixed1d import source_iteration
from ants.utils import manufactured_1d as mms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cells_x = 100
angles = 4
groups = 1

length = 1.
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

bc_x = [0, 0]

info = {
            "cells_x": cells_x,
            "angles": angles, 
            "groups": groups, 
            "materials": 1,
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": bc_x,
            "angular": True
        }

angle_x, angle_w = ants.angular_x(info)
medium_map = np.zeros((cells_x), dtype=np.int32)

xs_total = np.array([[1.0]])
xs_scatter = np.array([[[0.9]]])
xs_fission = np.array([[[0.0]]])

# Sources
external = ants.external1d.manufactured_ss_03(centers_x, angle_x)
boundary_x = ants.boundary1d.manufactured_ss_03(angle_x)


flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)

exact = mms.solution_ss_03(centers_x, angle_x)

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
