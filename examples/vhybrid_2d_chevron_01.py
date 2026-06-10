########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional multigroup time-dependent problem on a chevron geometry
# solved with the vectorized hybrid method. The coarse grid resolution
# (groups_c, angles_c) can vary per time step, allowing adaptive accuracy.
# A decaying deuterium-tritium (14.1 MeV) boundary source enters from y = 0.
#
# Note: Requires weight_matrix_2d_chevron.npy in the working directory.
#
########################################################################

from pathlib import Path

import numpy as np

import ants
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TemporalDiscretization,
    TimeDependentData,
)
from ants.vhybrid2d import time_dependent

cells_x = 90
cells_y = 90
angles_u = 8
angles_c = 4  # coarse angles per step (can vary per step)
groups_u = 87
groups_c = 43  # coarse groups per step (can vary per step)
bc_x = [0, 0]
bc_y = [0, 0]

steps = 50
T = 50e-6
dt = np.round(T / steps, 10)
edges_t = np.round(np.linspace(0, steps * dt, steps + 1), 10)

length_x = 9.0
length_y = 9.0

delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.linspace(0, length_y, cells_y + 1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Energy grid (uncollided uses full 87 groups)
edges_g, edges_gidx_u = ants.energy_grid(87, groups_u)

# Uncollided cross sections
materials = ["uranium-%0.7%", "high-density-polyethyene-087"]
xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials, datatype=False)
velocity_u = ants.energy_velocity(groups_u, edges_g)

# Boundary conditions: DT source at y = 0 with gamma decay
boundary_x_base, boundary_y_base = ants.boundary2d.deuterium_tritium(-1, 0, edges_g)
boundary_x = boundary_x_base[None, ...].copy()
gamma_steps = ants.gamma_time_steps(edges_t)
boundary_y = ants.boundary2d.time_dependence_decay_03(boundary_y_base, gamma_steps)

# Load pre-computed chevron geometry weight matrix
weight_matrix = np.load(Path(__file__).with_name("weight_matrix_2d_chevron.npy"))
data = ants.weight_spatial2d(weight_matrix, xs_total_u, xs_scatter_u, xs_fission_u)
medium_map, xs_total_u, xs_scatter_u, xs_fission_u = data

mat_data_u = MaterialData(
    total=xs_total_u,
    scatter=xs_scatter_u,
    fission=xs_fission_u,
    velocity=velocity_u,
)

# Angular quadrature (fine grid)
quadrature_u = ants.angular_xy(angles_u, bc_x=bc_x, bc_y=bc_y)

# Per-step coarse group and angle counts (uniform here, but can be varied)
vgroups_c = np.array([groups_c] * steps, dtype=np.int32)
vangles_c = np.array([angles_c] * steps, dtype=np.int32)

sources = SourceData(
    initial_flux_x=np.zeros((cells_x + 1, cells_y, angles_u**2, groups_u)),
    initial_flux_y=np.zeros((cells_x, cells_y + 1, angles_u**2, groups_u)),
    external=np.zeros((1, cells_x, cells_y, 1, 1)),
    boundary_x=boundary_x,
    boundary_y=boundary_y,
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
time_data = TimeDependentData(
    steps=steps, dt=dt, time_disc=TemporalDiscretization.TR_BDF2
)

flux = time_dependent(
    mat_data_u,
    vgroups_c,
    sources,
    geometry,
    quadrature_u,
    vangles_c,
    solver,
    time_data,
    edges_g,
)
# np.save(f"flux_vhybrid_chevron_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}", flux)
# np.save(f"flux_vhybrid_chevron_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}", flux)
