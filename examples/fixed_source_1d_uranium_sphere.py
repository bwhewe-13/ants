
import numpy as np

import ants
from ants.fixed1d import source_iteration


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
            "bcdim_x": 2
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx = ants.energy_grid(groups, 87)

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
materials = [[0, "uranium-%20%", "0-4"], [1, "uranium-%0%", "4-6"], \
             [2, "stainless-steel-440", "6-10"]]
medium_map = ants.spatial_map(materials, edges_x)

# Cross Sections
materials = np.array(materials)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

# External and boundary sources
external = ants.externals(0.0, (cells * groups,))
boundary_x = ants.boundaries("14.1-mev", (2, groups), [1], \
                             energy_grid=edges_g).flatten()

flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
            boundary_x, medium_map, delta_x, angle_x, angle_w, info)
# np.save("time_independent_uranium_sphere", flux)