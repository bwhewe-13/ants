
import numpy as np

import ants
from ants.critical1d import power_iteration

# General conditions
cells_x = 1000
angles = 8
groups = 618

info = {
            "cells_x": cells_x,
            "angles": angles, 
            "groups": groups, 
            "materials": 3,
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": [0, 1]
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

enrich = "80"

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
layers = [[0, "high-density-polyethyene-618", "0-5"],
             [1, "plutonium-%{}%".format(enrich), "5-6.5"],
             [2, "plutonium-240", "6.5-10"]]
medium_map = ants.spatial1d(layers, edges_x)

# Cross Sections
materials = np.array(layers)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                             delta_x, angle_x, angle_w, info)

# np.save(f"critical_plutonium_{enrich}_slab_flux", flux)
# np.save(f"critical_plutonium_{enrich}_slab_keff", keff)
