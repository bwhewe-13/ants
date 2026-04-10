########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional multigroup (87-group) time-dependent problem in
# spherical geometry. Uranium-20% core with depleted uranium and
# stainless steel shielding. A deuterium-tritium (14.1 MeV) source
# enters at the inner reflective boundary.
#
########################################################################

import numpy as np

import ants
from ants.datatypes import GeometryData, SolverData, SourceData, TimeDependentData
from ants.timed1d import time_dependent

# General conditions
cells_x = 1000
angles = 16
groups = 87
steps = 100
dt = 1e-8
bc_x = [1, 0]  # reflective at origin, vacuum at surface

# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy grid and neutron velocity
edges_g, edges_gidx = ants.energy_grid(87, groups)
velocity = ants.energy_velocity(groups, edges_g)

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

# Medium map
layers = [
    [0, "uranium-%20%", "0-4"],
    [1, "uranium-%0%", "4-6"],
    [2, "stainless-steel-440", "6-10"],
]
medium_map = ants.spatial1d(layers, edges_x)

# Cross sections
mat_data = ants.materials(87, np.array(layers)[:, 1], datatype=True)
mat_data.velocity = velocity

# Time-dependent DT boundary source with exponential decay
edges_t = np.linspace(0, steps * dt, steps + 1)
boundary_x = ants.boundary1d.deuterium_tritium(1, edges_g)
boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)

sources = SourceData(
    initial_flux=np.zeros((cells_x, angles, groups)),
    external=np.zeros((1, cells_x, 1, 1)),
    boundary_x=boundary_x,
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=2,  # spherical geometry
)
solver = SolverData()
time_data = TimeDependentData(steps=steps, dt=dt)

flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
# np.save("time_dependent_uranium_sphere", flux)
