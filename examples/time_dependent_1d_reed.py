########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Transient version of Reed's monoenergetic multi-region problem,
# comparing the time-dependent solution at the final step to the
# steady-state solution.
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
from ants.fixed1d import fixed_source
from ants.timed1d import time_dependent

# General conditions
cells_x = 320
angles = 8
groups = 1
steps = 100
dt = 1.0
bc_x = [0, 0]

# Spatial
length = 16.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Neutron velocity (arbitrary for monoenergetic)
edges_g, _ = ants.energy_grid(None, groups)
velocity = ants.energy_velocity(groups, edges_g)

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

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

# Time-dependent external source (constant in time)
external_ss = ants.external1d.reeds(edges_x, bc_x)
external = ants.external1d.time_dependence_constant(external_ss)
boundary_x = np.zeros((1, 2, 1, 1))

# Time-dependent problem
td_sources = SourceData(
    initial_flux=np.zeros((cells_x, angles, groups)),
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

flux_td = time_dependent(mat_data, td_sources, geometry, quadrature, solver, time_data)

# Steady-state comparison
ss_sources = SourceData(
    external=external_ss,
    boundary_x=np.zeros((2, 1, 1)),
)
flux_ss = fixed_source(mat_data, ss_sources, geometry, quadrature, solver)

fig, ax = plt.subplots()
ax.plot(centers_x, flux_ss.flatten(), label="Steady-State", c="k", ls=":")
ax.plot(
    centers_x, flux_td[-1, :, 0], label="Time-Dependent (final step)", c="r", alpha=0.6
)
ax.set_title("Reed Problem")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
plt.show()
