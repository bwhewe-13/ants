
import numpy as np

import ants
from ants.critical1d import power_iteration

# General conditions
cells = 1000
angles = 8
groups = 87

info = {
            "cells_x": cells,
            "angles": angles,
            "groups": groups, 
            "materials": 3,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 2,
            "bc_x": [0, 1],
            "bcdim_x": 1
        }

# Spatial
length = 100.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

enrich = "20"

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
materials = [[0, "high-density-polyethyene-087", "0-45"],
             [1, "uranium-hydride-%{}%".format(enrich), "45-80"],
             [2, "uranium-hydride-%0%", "80-100"]]
medium_map = ants.spatial_map(materials, edges_x)

# Cross Sections
materials = np.array(materials)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                             delta_x, angle_x, angle_w, info)

# np.save(f"critical_uranium_{enrich}_hdpe_slab_flux", flux)
# np.save(f"critical_uranium_{enrich}_hdpe_slab_keff", keff)
