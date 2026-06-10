########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional multigroup (87-group) time-dependent problem on a
# chevron geometry: triangular uranium fuel regions in an HDPE moderator
# background. A decaying deuterium-tritium (14.1 MeV) boundary source
# enters from y = 0.
#
# Note: Requires weight_matrix_2d_chevron.npy in the working directory.
# This file encodes the mixed-material cell fractions for the chevron
# geometry and was generated using ants.weight_matrix2d() with the
# triangle and rectangle region definitions below.
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
from ants.timed2d import time_dependent

cells_x = 90
cells_y = 90
angles = 4
groups = 87
bc_x = [0, 0]
bc_y = [0, 0]

steps = 1
T = 1e-6
dt = np.round(T / steps, 10)
edges_t = np.round(np.linspace(0, steps * dt, steps + 1), 10)

length_x = 9.0
length_y = 9.0

delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 10)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 10)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Energy grid and neutron velocity
edges_g, edges_gidx = ants.energy_grid(87, groups)
velocity = ants.energy_velocity(groups, edges_g)

# Cross sections for two materials: uranium and HDPE
materials = ["uranium-%0.7%", "high-density-polyethyene-087"]
xs_total, xs_scatter, xs_fission = ants.materials(87, materials, datatype=False)

# Load pre-computed chevron geometry weight matrix.
# To regenerate the weight matrix, uncomment the code below:
#
# triangle01 = [(0.1, 1.0), (0.1, 3.9), (5.9, 1.0)]
# triangle02 = [(6.0, 1.0), (8.9, 1.0), (8.9, 3.9)]
# triangle03 = [(0.1, 4.9), (0.1, 7.8), (5.9, 4.9)]
# triangle04 = [(6.0, 4.9), (8.9, 4.9), (8.9, 7.8)]
# triangles = np.array([triangle01, triangle02, triangle03, triangle04])
# t_index = [1, 1, 0, 0]
# rectangle01 = [(0, 0), 0.1, 9.0]
# rectangle02 = [(0, 8.9), 9.0, 0.1]
# rectangle03 = [(8.9, 0), 0.1, 9.0]
# rectangles = [rectangle01, rectangle02, rectangle03]
# r_index = [0, 0, 0]
# N_particles = cells_x * cells_y * 40
# weight_matrix = ants.weight_matrix2d(
#     edges_x, edges_y, 2, N_particles=N_particles,
#     triangles=triangles, triangle_index=t_index,
#     rectangles=rectangles, rectangle_index=r_index,
# )
# np.save("weight_matrix_2d_chevron", weight_matrix)

weight_matrix = np.load(Path(__file__).with_name("weight_matrix_2d_chevron.npy"))
data = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
medium_map, xs_total, xs_scatter, xs_fission = data


mat_data = MaterialData(
    total=xs_total,
    scatter=xs_scatter,
    fission=xs_fission,
    velocity=velocity,
)

# Angular quadrature
quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

# Boundary conditions: DT source at y = 0 with gamma decay
boundary_x_base, boundary_y_base = ants.boundary2d.deuterium_tritium(-1, 0, edges_g)
boundary_x = boundary_x_base[None, ...].copy()
gamma_steps = ants.gamma_time_steps(edges_t)
boundary_y = ants.boundary2d.time_dependence_decay_03(boundary_y_base, gamma_steps)

sources = SourceData(
    initial_flux_x=np.zeros((cells_x + 1, cells_y, angles**2, groups)),
    initial_flux_y=np.zeros((cells_x, cells_y + 1, angles**2, groups)),
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

flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
# np.save("time_dependent_2d_chevron", flux)
flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
# np.save("time_dependent_2d_chevron", flux)
