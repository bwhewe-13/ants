########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional monoenergetic time-dependent problem with a fissile
# cylinder embedded in a void. The cylinder geometry is represented using
# a stochastic weight matrix. A boundary source at x = 0 decays in time.
#
########################################################################

import numpy as np

import ants
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.timed2d import time_dependent

cells_x = 100
cells_y = 100
angles = 8
groups = 1
steps = 1000
dt = 0.1
bc_x = [0, 0]
bc_y = [0, 0]

# Cylinder geometry: radius of critical sphere in 2D slab approximation
radius = 4.279960
coords = [[(radius, radius), (0.0, radius)]]

length_x = length_y = 2 * radius

# Spatial dimensions
delta_x = np.repeat(length_x / cells_x, cells_x)
delta_y = np.repeat(length_y / cells_y, cells_y)
edges_x = np.linspace(0, length_x, cells_x + 1)
edges_y = np.linspace(0, length_y, cells_y + 1)

# Cross sections for fissile cylinder (index 0) and void (index 1)
xs_total = np.array([[0.32640], [0.0]])
xs_scatter = np.array([[[0.225216]], [[0.0]]])
xs_fission = np.array([[[2.84 * 0.0816]], [[0.0]]])

# Stochastic weight matrix to represent curved cylinder on Cartesian grid
weight_matrix = ants.weight_matrix2d(
    edges_x,
    edges_y,
    materials=2,
    N_particles=cells_x * 50_000,
    circles=coords,
    circle_index=[0],
)

weighted = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
medium_map, xs_total, xs_scatter, xs_fission = weighted

# Neutron velocity (monoenergetic, arbitrary units)
edges_g, _, _ = ants.energy_grid(None, groups, groups)
velocity = ants.energy_velocity(groups, edges_g)

# Angular quadrature
quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

mat_data = MaterialData(
    total=xs_total,
    scatter=xs_scatter,
    fission=xs_fission,
    velocity=velocity,
)

# Boundary source at x = 0: step function that turns off after 0.1 s
edges_t = np.linspace(0, dt * steps, steps + 1)
boundary_x_base = np.zeros((2, 1, 1, 1))
boundary_x_base[0] = 1.0
boundary_x = ants.boundary2d.time_dependence_decay_01(boundary_x_base, edges_t, 0.1)

sources = SourceData(
    initial_flux=np.zeros((cells_x, cells_y, angles**2, groups)),
    external=np.zeros((1, cells_x, cells_y, 1, 1)),
    boundary_x=boundary_x,
    boundary_y=np.zeros((1, 2, 1, 1, 1)),
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    delta_y=delta_y,
    bc_x=bc_x,
    bc_y=bc_y,
    geometry=3,  # 2D slab
)
solver = SolverData()
time_data = TimeDependentData(steps=steps, dt=dt)

flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
# np.save("time_dependent_2d_cylinder", flux)
