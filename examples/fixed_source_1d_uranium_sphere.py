
import numpy as np

import ants
from ants.fixed1d import source_iteration


# General conditions
cells_x = 1000
angles = 16
groups = 87

info = {
            "cells_x": cells_x,
            "angles": angles, 
            "groups": groups, 
            "materials": 3,
            "geometry": 2,
            "spatial": 2,
            "bc_x": [1, 0]
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx = ants.energy_grid(87, groups)

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
layers = [[0, "uranium-%20%", "0-4"], [1, "uranium-%0%", "4-6"], \
             [2, "stainless-steel-440", "6-10"]]
medium_map = ants.spatial1d(layers, edges_x)

# Cross Sections
materials = np.array(layers)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

# External and boundary sources
external = np.zeros((cells_x, 1, 1))
boundary_x = ants.boundary1d.deuterium_tritium([0], edges_g)

flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
            boundary_x, medium_map, delta_x, angle_x, angle_w, info)
# np.save("time_independent_uranium_sphere", flux)