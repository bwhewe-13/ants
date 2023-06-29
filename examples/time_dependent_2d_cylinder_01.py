########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
########################################################################

import numpy as np
import matplotlib.pyplot as plt

import ants
from ants.timed2d import backward_euler
from ants.critical2d import power_iteration


cells_x = 100
cells_y = 100
angles = 8
groups = 1
steps = 1000

# Spatial Layout
radii = [(0.0, 4.279960)]
radius = max(radii)[1]

delta_x = np.repeat(radius * 2 / cells_x, cells_x)
edges_x = np.linspace(0, radius * 2, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

delta_y = np.repeat(radius * 2 / cells_y, cells_y)
edges_y = np.linspace(0, radius * 2, cells_y+1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Boundary conditions
bc_x = [0, 0]
bc_y = [0, 0]

# Cross Sections
xs_total = np.array([[0.32640], [0.0]])
xs_scatter = np.array([[[0.225216]], [[0.0]]])
xs_fission = np.array([[[2.84*0.0816]], [[0.0]]])

# weight_map = np.load("time_dependent_2d_cylinder_01_weight_map.npy")

# Update cross sections for cylinder
medium_map, xs_total, xs_scatter, xs_fission, weight_map \
    = ants.cylinder2d(radii, xs_total, xs_scatter, xs_fission, delta_x, \
                      delta_y, bc_x, bc_y)#, weight_map=weight_map)
np.save("time_dependent_2d_cylinder_01_weight_map", weight_map)

info = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles, 
            "groups": groups, 
            "materials": xs_total.shape[0],
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 3,
            "bc_x": bc_x,
            "bcdim_x": 1,
            "bcdecay_x": 1, 
            "bc_y": bc_y,
            "bcdim_y": 1, 
            "steps": steps, 
            "dt": 0.1
        }


angle_x, angle_y, angle_w = ants.angular_xy(info)

# Boundary conditions and external source
external = np.zeros((cells_x * cells_y * angles**2 * groups))
boundary_x = np.array([1.0, 0.0])
boundary_y = np.array([0.0, 0.0])

# Velocity
velocity = np.ones((groups), dtype=float)

# info["qdim"] = 2
# flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
#                              delta_x, delta_y, angle_x, angle_y, angle_w, info)

flux = backward_euler(xs_total, xs_scatter, xs_fission, velocity, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)
