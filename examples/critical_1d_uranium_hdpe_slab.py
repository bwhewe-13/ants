
import ants
from ants.critical1d import power_iteration

import numpy as np
import time

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
            "bcdim_x": 1,
        }

# Spatial
length = 100.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

enrich = "20"

# Angular
angle_x, angle_w = ants._angle_x(info)

# Medium Map
materials = [[0, "high-density-polyethyene-087", "0-45"],
             [1, "uranium-hydride-%{}%".format(enrich), "45-80"],
             [2, "uranium-hydride-%0%", "80-100"]]
medium_map = ants._medium_map(materials, edges_x)

# # Cross Sections
# materials_names = np.array(materials)[:,1]
# xs_total, xs_scatter, xs_fission = ants.materials(groups, materials_names)

from discrete1.keigenvalue import Problem1
_, _, _, _, total, scatter, fission, _, _ = Problem1.steady("hdpe", 0.20)
xs_total = np.array([total[0], total[500], total[-1]])
xs_scatter = np.array([scatter[0], scatter[500], scatter[-1]])
xs_fission = np.array([fission[0], fission[500], fission[-1]])

start = time.time()
flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                             delta_x, angle_x, angle_w, info)
end = time.time()

np.save("critical_uranium_{}_hdpe_slab_flux".format(enrich), flux)
np.save("critical_uranium_{}_hdpe_slab_keff".format(enrich), keff)

print("Elapsed Time:", end - start)

# True Keff - 20% Enriched
# array(0.90833818)