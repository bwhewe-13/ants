########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Transient Reed's monoenergetic problem solved with the vectorized
# hybrid method (vhybrid1d). The coarse grid resolution (groups_c,
# angles_c) is specified per time step as integer arrays, allowing the
# angular and energy resolution to vary adaptively across time steps.
#
########################################################################

import matplotlib.pyplot as plt
import numpy as np

import ants
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.vhybrid1d import time_dependent

# General conditions
cells_x = 320
angles_u = 8
angles_c = 4  # coarse angles (can vary per step)
groups = 1  # monoenergetic
steps = 100
dt = 1.0
bc_x = [0, 0]

# Spatial
length = 16.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy grid and neutron velocity (monoenergetic)
edges_g, _ = ants.energy_grid(None, groups)
velocity = ants.energy_velocity(groups, edges_g)

# Angular quadrature (fine grid)
quadrature_u = ants.angular_x(angles_u, bc_x=bc_x)

# Medium map
layers = [
    [0, "scatter", "0-4, 12-16"],
    [1, "vacuum", "4-5, 11-12"],
    [2, "absorber", "5-6, 10-11"],
    [3, "source", "6-10"],
]
medium_map = ants.spatial1d(layers, edges_x)

# Cross sections
mat_data = MaterialData(
    total=np.array([[1.0], [0.0], [5.0], [50.0]]),
    scatter=np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]]),
    fission=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
    velocity=velocity,
)

# Sources (constant in time)
external_ss = ants.external1d.reeds(edges_x, bc_x)
external = ants.external1d.time_dependence_constant(external_ss)
boundary_x = np.zeros((1, 2, 1, 1))

sources = SourceData(
    initial_flux=np.zeros((cells_x, angles_u, groups)),
    external=external,
    boundary_x=boundary_x,
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=1,
)
solver = SolverData()
time_data = TimeDependentData(steps=steps, dt=dt)

# Per-step coarse group and angle counts
# groups_c can equal groups (no energy coarsening for monoenergetic)
vgroups_c = np.array([groups] * steps, dtype=np.int32)
vangles_c = np.array([angles_c] * steps, dtype=np.int32)

flux = time_dependent(
    mat_data,
    vgroups_c,
    sources,
    geometry,
    quadrature_u,
    vangles_c,
    solver,
    time_data,
    edges_g,
)

fig, ax = plt.subplots()
ax.plot(centers_x, flux[-1, :, 0], label="Last Time Step", c="r", alpha=0.6)
ax.set_title("Reed Problem - Vectorized Hybrid Method")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
plt.show()
