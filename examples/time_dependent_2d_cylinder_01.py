########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
########################################################################

import numpy as np

import ants
from ants.timed2d import backward_euler
from ants.critical2d import power_iteration


cells_x = 100
cells_y = 100
angles = 8
groups = 1
steps = 1000

# Spatial Layout
radius = 4.279960
coordinates = [(radius, radius), [radius]]

length_x = length_y = 2 * radius

# Spatial Dimensions
delta_x = np.repeat(length_x / cells_x, cells_x)
delta_y = np.repeat(length_y / cells_y, cells_y)

edges_x = np.linspace(0, length_x, cells_x + 1)
edges_y = np.linspace(0, length_y, cells_y + 1)

# Boundary conditions
bc_x = [0, 0]
bc_y = [0, 0]

# Cross Sections
xs_total = np.array([[0.32640], [0.0]])
xs_scatter = np.array([[[0.225216]], [[0.0]]])
xs_fission = np.array([[[2.84*0.0816]], [[0.0]]])

weight_matrix = ants.weight_cylinder2d(coordinates, edges_x, edges_y)
medium_map, xs_total, xs_scatter, xs_fission \
    = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)

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
