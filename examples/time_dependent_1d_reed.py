
import ants
from ants.timed1d import backward_euler
from ants.fixed1d import source_iteration

import numpy as np
import matplotlib.pyplot as plt

# General conditions
cells = 320
angles = 8
groups = 1
steps = 100

info = {
            "cells_x": cells,
            "angles": angles,
            "groups": groups,
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

# Spatial
length = 16.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
velocity = np.ones((groups,))

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
materials = [[0, "scatter", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"],
             [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
medium_map = ants.spatial_map(materials, edges_x)

# Material Cross Sections
xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])

# External Source and Boundary
external = ants.externals("reeds", (cells, angles, groups), \
                          edges_x=edges_x, bc=[0,0]).flatten()
boundary_x = np.zeros((2,))

# Time Dependent
dependent = backward_euler(xs_total, xs_scatter, xs_fission, velocity, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)

# Time Independent
xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
independent = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
independent = independent.flatten()


fig, ax = plt.subplots()
ax.plot(centers_x, independent, label="Time Independent", c="k", ls=":")
ax.plot(centers_x, dependent[-1,:,0], label="Time Dependent", c="r", alpha=0.6)

ax.set_title("Reed Problem")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")

ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
plt.show()
