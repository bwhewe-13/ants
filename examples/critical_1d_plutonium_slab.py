
import numpy as np
import matplotlib.pyplot as plt
import time

import ants
from ants.critical1d import power_iteration as power


# General conditions
cells = 1000
angles = 8
groups = 618

info = {
            "cells_x": cells,
            "angles": angles, 
            "groups": groups, 
            "materials": 3,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 2, 
            "bc_x": [0, 1],
            "bcdim_x": 1,
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

enrich = "80"

# Angular
angle_x, angle_w = ants._angle_x(info)

# Medium Map
materials = [[0, "high-density-polyethyene-618", "0-5"],
             [1, "plutonium-%{}%".format(enrich), "5-6.5"],
             [2, "plutonium-240", "6.5-10"]]
medium_map = ants._medium_map(materials, edges_x)

# Cross Sections
materials_names = np.array(materials)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials_names)

start = time.time()
flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                    delta_x, angle_x, angle_w, info)
stop = time.time()

np.save(f"critical_plutonium_{enrich}_slab_flux", flux)
np.save(f"critical_plutonium_{enrich}_slab_keff", keff)



