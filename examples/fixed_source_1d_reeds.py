########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One dimensional monoenergetic, multi-region problem. Taken from Reed's
# "New Difference Schemes for the Neutron Transport Equation" (1971).
# 
########################################################################

import ants
from ants.fixed1d import source_iteration

import numpy as np
import matplotlib.pyplot as plt

# General conditions
cells_x = 160
angles = 4
groups = 1

# Different boundary conditions
bc_x = [0, 0]
layout = [[0, "scattering", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"], \
          [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]

# bc_x = [0, 1]
# layout = [[0, "scattering", "0-4"], [1, "vacuum", "4-5"], \
#           [2, "absorber", "5-6"], [3, "source", "6-8"]]

# bc_x = [1, 0]
# layout = [[0, "scattering", "4-8"], [1, "vacuum", "3-4"], \
#           [2, "absorber", "2-3"], [3, "source", "0-2"]]

info = {
            "cells_x": cells_x,
            "angles": angles, 
            "groups": groups, 
            "materials": len(layout),
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": bc_x,
            "angular": True
        }

# Spatial
length = 8. if np.sum(bc_x) > 0.0 else 16.
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Medium Map
materials = np.array(layout)[:,1]
medium_map = ants.spatial1d(layout, edges_x)

# Angular
angle_x, angle_w = ants.angular_x(info)

# Cross Sections
xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])


# Sources
external = ants.external1d.reeds(edges_x, bc_x)
boundary_x = np.zeros((2, 1, 1))


flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
# Convert to scalar flux
if info["angular"]:
    flux = np.sum(flux[:,:,0] * angle_w[None,:], axis=1)

fig, ax = plt.subplots()
ax.plot(centers_x, flux.flatten(), color="r", label="Scalar Flux")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")
ax.set_title("Reed's Solutions")
plt.show()
