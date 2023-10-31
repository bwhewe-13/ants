########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
########################################################################

import ants
from ants.fixed2d import source_iteration
from ants.utils import manufactured_2d as mms

import numpy as np
import matplotlib.pyplot as plt


cells_x = cells_y = 50
# cells_y = 200
angles = angles1 = 4
# angles1 = 4
groups = 1

length_x = 1.
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

length_y = 1.
delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.linspace(0, length_y, cells_y+1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

bc = [0, 0]

info = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles, 
            "groups": groups, 
            "materials": 1,
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": bc,
            "bc_y": bc,
            "angular": False
        }


# Angular
angle_x, angle_y, angle_w = ants.angular_xy(info)

# Materials
xs_total = np.array([[1.0]])
xs_scatter = np.array([[[0.0]]])
xs_fission = np.array([[[0.0]]])

# Externals
external = np.ones((info["cells_x"], info["cells_y"], 1, 1))
boundary_x, boundary_y = ants.boundary2d.manufactured_ss_02(centers_x, \
                                        centers_y, angle_x, angle_y)

# Layout
medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)

flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)
exact = mms.solution_ss_02(centers_x, centers_y, angle_x, angle_y)

exact = np.sum(exact * angle_w[None,None,:,None], axis=(2,3))