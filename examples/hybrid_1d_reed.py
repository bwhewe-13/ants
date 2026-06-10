########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Transient Reed's monoenergetic problem solved with the collision-based
# hybrid method. The uncollided and collided problems use the same single
# energy group and the same angular quadrature for simplicity.
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
from ants.hybrid1d import time_dependent
from ants.utils import hybrid as hytools

# General conditions
cells_x = 320
angles_u = 8
angles_c = 8
groups_u = 1
groups_c = 1
steps = 100
dt = 1.0
bc_x = [0, 0]

# Spatial
length = 16.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy grid (monoenergetic)
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups_u, groups_c)
velocity_u = ants.energy_velocity(groups_u, edges_g)
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

# Angular quadratures
quadrature_u = ants.angular_x(angles_u, bc_x=bc_x)
quadrature_c = ants.angular_x(angles_c, bc_x=bc_x)

# Medium map
layers = [
    [0, "scatter", "0-4, 12-16"],
    [1, "vacuum", "4-5, 11-12"],
    [2, "absorber", "5-6, 10-11"],
    [3, "source", "6-10"],
]
medium_map = ants.spatial1d(layers, edges_x)

# Uncollided cross sections
mat_data_u = MaterialData(
    total=np.array([[1.0], [0.0], [5.0], [50.0]]),
    scatter=np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]]),
    fission=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
    velocity=velocity_u,
)
# Collided cross sections (same as uncollided for monoenergetic)
mat_data_c = MaterialData(
    total=mat_data_u.total.copy(),
    scatter=mat_data_u.scatter.copy(),
    fission=mat_data_u.fission.copy(),
    velocity=velocity_c,
)

# Hybrid energy group indexing
hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

# Sources (constant in time)
external_ss = ants.external1d.reeds(edges_x, bc_x)
external = ants.external1d.time_dependence_constant(external_ss)
boundary_x = np.zeros((1, 2, 1, 1))

sources = SourceData(
    initial_flux=np.zeros((cells_x, angles_u, groups_u)),
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

flux = time_dependent(
    mat_data_u,
    mat_data_c,
    sources,
    geometry,
    quadrature_u,
    quadrature_c,
    solver,
    time_data,
    hybrid_data,
)

fig, ax = plt.subplots()
ax.plot(centers_x, flux[:, 0], label="Last Time Step", c="r", alpha=0.6)
ax.set_title("Reed Problem - Hybrid Method")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
plt.show()
