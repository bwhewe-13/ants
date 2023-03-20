########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One dimensional method of manufactured solution problem with no 
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

cells = 100
angles = 4
groups = 1

length = 1.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

bc = [0, 0]

params = {
            "cells": cells, 
            "angles": angles, 
            "groups": groups, 
            "materials": 1,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 1, 
            "bc": bc,
            "bcdim": 2, 
            "steps": 0, 
            "dt": 0, 
            "adjoint": False, 
            "angular": True
        }

xs_total = np.array([[1.0]])
xs_scatter = np.array([[[0.0]]])
xs_fission = np.array([[[0.0]]])

external = ants.externals(1.0, (cells,))
boundary = ants.boundaries(1.0, (2, angles, groups), [0]).flatten()

angle_x, angle_w = ants._angle_x(params)
medium_map = np.zeros((cells), dtype=np.int32)

flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary, medium_map, delta_x, angle_x, angle_w, \
                        params)



exact = mms.solution_mms_01(centers_x, angle_x)

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
