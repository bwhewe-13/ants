
import ants
from ants.hybrid1d import bdf1

import numpy as np
import matplotlib.pyplot as plt

# General conditions
cells = 320
angles_u = 8
angles_c = 8
groups_u = 1
groups_c = 1
steps = 100

info_u = {
            "cells_x": cells,
            "angles": angles_u,
            "groups": groups_u,
            "materials": 4,
            "geometry": 1,
            "spatial": 2,
            "qdim": 3,
            "bc_x": [0, 0],
            "bcdim_x": 1,
            "steps": steps,
            "dt": 1.,
            "angular": False
        }

info_c = {
            "cells_x": cells,
            "angles": angles_c,
            "groups": groups_c,
            "materials": 4,
            "geometry": 1,
            "spatial": 2,
            "qdim": 2,
            "bc_x": [0, 0],
            "bcdim_x": 1,
            "steps": steps,
            "dt": 1.,
            "angular": False
        }

# Spatial
length = 16.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx = ants.energy_grid(groups_u, 1)
velocity = ants.energy_velocity(groups_u, None)

# Angular
angle_x, angle_w = ants.angular_x(info_u)

# Medium Map
layers = [[0, "scatter", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"],
             [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
medium_map = ants.spatial1d(layers, edges_x)

# Material Cross Sections
xs_total_u = np.array([[1.0], [0.0], [5.0], [50.0]])
xs_scatter_u = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
xs_fission_u = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])
# Collided cross sections
xs_total_c = xs_total_u.copy()
xs_scatter_c = xs_scatter_u.copy()
xs_fission_c = xs_fission_u.copy()

# External Source and Boundary
external = ants.externals1d("reeds", (cells, angles_u, groups_u), \
                          edges_x=edges_x, bc=[0,0]).flatten()
boundary_x = np.zeros((2,))

flux = bdf1(xs_total_u, xs_scatter_u, xs_fission_u, xs_total_c, \
            xs_scatter_c, xs_fission_c, velocity, external, boundary_x, \
            medium_map, delta_x, edges_g, edges_gidx, info_u, info_c)


fig, ax = plt.subplots()
ax.plot(centers_x, flux[-1,:,0], label="Last Time Step", c="r", alpha=0.6)

ax.set_title("Reed Problem")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")

ax.legend(loc=0, framealpha=1)
ax.grid(which="both")

plt.show()