
import ants
from ants.timed1d import backward_euler

import numpy as np

# General conditions
cells = 1000
angles = 16
groups = 87
steps = 100

params = {
            "cells": cells, 
            "angles": angles, 
            "groups": groups, 
            "materials": 2,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 3, 
            "bc": [0, 0],
            "bcdim": 1, 
            "steps": steps, 
            "dt": 1e-8,
            "adjoint": False, 
            "angular": True, 
            "bcdecay": 2
        }

# Spatial
length = 10.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
energy_grid, idx_edges = ants._energy_grid(groups, 87)
velocity = ants._velocity(groups, energy_grid)

# Angular
angle_x, angle_w = ants._angle_x(params)

# Medium Map
materials = [[0, "ss440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = ants._medium_map(materials, edges_x)

# Cross Sections
materials = ["stainless-steel-440", "uranium-%20%"]
xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

# External and boundary sources
external = ants.externals(0.0, (cells, angles, groups)).flatten()
boundary = ants.boundaries("14.1-mev", (2, groups), [0], \
                           energy_grid=energy_grid).flatten()



angular_flux = backward_euler(xs_total, xs_scatter, xs_fission, velocity, \
                              external, boundary, medium_map, delta_x, \
                              angle_x, angle_w, params)

scalar_flux = np.sum(flux * angle_w[None,None,:,None], axis=2)
