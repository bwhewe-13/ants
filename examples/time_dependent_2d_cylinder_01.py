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
            "bc_x": bc_x,
            "bc_y": bc_y, 
            "steps": steps, 
            "dt": 0.1
        }


angle_x, angle_y, angle_w = ants.angular_xy(info)

# Boundary conditions and external source
external = np.zeros((1, cells_x, cells_y, 1, 1))

boundary_x = np.zeros((2, 1, 1, 1))
boundary_x[0] = 1.
edges_t = np.linspace(0, info["dt"] * steps, steps + 1)
boundary_x = ants.boundary2d.time_dependence_decay_01(boundary_x, edges_t, 0.1)

boundary_y = np.zeros((1, 2, 1, 1, 1))

initial_flux = np.zeros((cells_x, cells_y, angles**2, groups))

flux = backward_euler(initial_flux, xs_total, xs_scatter, xs_fission, \
                      velocity, external, boundary_x, boundary_y, \
                      medium_map, delta_x, delta_y, angle_x, angle_y, \
                      angle_w, info)
