########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional multigroup time-dependent problem with the collision-
# based hybrid method. A uranium-20% slab in stainless steel shielding
# uses 87 uncollided groups and 43 collided groups to reduce cost.
# A deuterium-tritium (14.1 MeV) source decays exponentially in time.
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
from ants.hybrid1d import time_dependent
from ants.utils import hybrid as hytools

# General conditions
cells_x = 1000
angles_u = 8
angles_c = 2
groups_u = 87
groups_c = 43
steps = 5
dt = 1e-8
bc_x = [0, 0]

# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy grids
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)
velocity_u = ants.energy_velocity(groups_u, edges_g)
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

# Angular quadratures
quadrature_u = ants.angular_x(angles_u, bc_x=bc_x)
quadrature_c = ants.angular_x(angles_c, bc_x=bc_x)

# Medium map
layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = ants.spatial1d(layers, edges_x)

# Uncollided cross sections (fine 87-group grid)
mat_data_u = ants.materials(87, np.array(layers)[:, 1], datatype=True)
mat_data_u.velocity = velocity_u

# Collided cross sections (coarsened to 43 groups)
xs_collided = hytools.coarsen_materials(
    mat_data_u.total,
    mat_data_u.scatter,
    mat_data_u.fission,
    edges_g[edges_gidx_u],
    edges_gidx_c,
)


mat_data_c = MaterialData(
    total=xs_collided[0],
    scatter=xs_collided[1],
    fission=xs_collided[2],
    velocity=velocity_c,
)

# Hybrid energy group indexing
hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

# Time-dependent DT boundary source with exponential decay
edges_t = np.linspace(0, steps * dt, steps + 1)
boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)

sources = SourceData(
    initial_flux=np.zeros((cells_x, angles_u, groups_u)),
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

# np.save(
#     f"hybrid_uranium_slab_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}_flux",
#     flux,
# )
# )
