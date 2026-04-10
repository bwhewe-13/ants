########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional multigroup (87-group) time-dependent problem with a
# uranium-20% slab in stainless steel shielding. A deuterium-tritium
# (14.1 MeV) boundary source with exponential decay enters from x = 0.
#
########################################################################

import numpy as np

import ants
from ants.datatypes import GeometryData, SolverData, SourceData, TimeDependentData
from ants.timed1d import time_dependent

# General conditions
cells_x = 1000
angles = 8
groups = 87
steps = 100
dt = 1e-8
bc_x = [0, 0]

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
layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = ants.spatial1d(layers, edges_x)

# Cross sections
mat_data = ants.materials(87, np.array(layers)[:, 1], datatype=True)
mat_data.velocity = velocity

# Time-dependent DT boundary source with exponential decay
edges_t = np.linspace(0, steps * dt, steps + 1)
boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
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
    geometry=1,
)
solver = SolverData()
time_data = TimeDependentData(steps=steps, dt=dt)

flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
# np.save("time_dependent_uranium_slab", flux)
