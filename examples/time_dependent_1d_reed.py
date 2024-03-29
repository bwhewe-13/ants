
import ants
from ants.timed1d import backward_euler
from ants.fixed1d import source_iteration

import numpy as np
import matplotlib.pyplot as plt

# General conditions
cells_x = 320
angles = 8
groups = 1
steps = 100

info = {
            "cells_x": cells_x,
            "angles": angles,
            "groups": groups,
            "materials": 4,
            "geometry": 1,
            "spatial": 2,
            "bc_x": [0, 0],
            "steps": steps,
            "dt": 1.,
            "angular": False
        }

# Spatial
length = 16.
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
velocity = np.ones((groups,))

# Angular
angle_x, angle_w = ants.angular_x(info)

# Medium Map
layers = [[0, "scatter", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"],
             [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
medium_map = ants.spatial1d(layers, edges_x)

# Material Cross Sections
xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])

# External Source and Boundary
external = ants.external1d.reeds(edges_x, info["bc_x"])
external = external[None,...].copy()
boundary_x = np.zeros((1, 2, 1, 1))

initial_flux = np.zeros((cells_x, angles, groups))

# Time Dependent
dependent = backward_euler(initial_flux, xs_total, xs_scatter, xs_fission, \
                           velocity, external, boundary_x, medium_map, \
                           delta_x, angle_x, angle_w, info)

# Time Independent
xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
independent = source_iteration(xs_total, xs_scatter, xs_fission, external[0], \
                boundary_x[0], medium_map, delta_x, angle_x, angle_w, info)
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
