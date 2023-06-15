
import numpy as np

import ants
from ants.critical1d import power_iteration

# General conditions
cells = 1000
angles = 16
groups = 87

info = {
            "cells_x": cells,
            "angles": angles, 
            "groups": groups, 
            "materials": 3,
            "geometry": 2,
            "spatial": 2,
            "qdim": 2,
            "bc_x": [1, 0],
            "bcdim_x": 1,
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular
angle_x, angle_w = ants._angle_x(info)

# Medium Map
materials = [[0, "uranium-%20%", "0-4"], [1, "uranium-%0%", "4-6"], \
             [2, "stainless-steel-440", "6-10"]]
medium_map = ants._medium_map(materials, edges_x)

# Cross Sections
materials = np.array(materials)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                             delta_x, angle_x, angle_w, info)
# np.save("critical_uranium_sphere", flux)
