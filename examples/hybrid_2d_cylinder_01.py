########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional monoenergetic time-dependent problem with a fissile
# cylinder embedded in a void, solved with the collision-based hybrid
# method. The cylinder geometry is represented using a stochastic weight
# matrix. A boundary source at x = 0 decays in time.
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
from ants.hybrid2d import time_dependent
from ants.utils import hybrid as hytools

cells_x = 100
cells_y = 100
angles_u = 8
angles_c = 2
groups_u = 1
groups_c = 1
steps = 100
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

# Neutron velocity (monoenergetic)
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups_u, groups_c)
velocity_u = ants.energy_velocity(groups_u, edges_g)
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

# Coarsen cross sections for collided solve
xs_collided = hytools.coarsen_materials(
    xs_total, xs_scatter, xs_fission, edges_g[edges_gidx_u], edges_gidx_c
)
xs_total_c, xs_scatter_c, xs_fission_c = xs_collided

mat_data_u = MaterialData(
    total=xs_total,
    scatter=xs_scatter,
    fission=xs_fission,
    velocity=velocity_u,
)
mat_data_c = MaterialData(
    total=xs_total_c,
    scatter=xs_scatter_c,
    fission=xs_fission_c,
    velocity=velocity_c,
)

# Angular quadratures (fine and coarse)
quadrature_u = ants.angular_xy(angles_u, bc_x=bc_x, bc_y=bc_y)
quadrature_c = ants.angular_xy(angles_c, bc_x=bc_x, bc_y=bc_y)

# Hybrid energy group indexing
hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

# Boundary source at x = 0: decays to zero at t = 8 s
edges_t = np.linspace(0, dt * steps, steps + 1)
boundary_x_base = np.zeros((2, 1, 1, 1))
boundary_x_base[0] = 1.0
boundary_x = ants.boundary2d.time_dependence_decay_01(boundary_x_base, edges_t, 8.0)

sources = SourceData(
    initial_flux=np.zeros((cells_x, cells_y, angles_u**2, groups_u)),
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
# np.save("hybrid_2d_cylinder", flux)
