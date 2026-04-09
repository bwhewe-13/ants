import numpy as np

import ants
from ants import hybrid1d, vhybrid1d
from ants.utils import hybrid as hytools

# General conditions
cells_x = 100
angles_u = 4
angles_c = 4
groups_u = 87
groups_c = 10
steps = 5

print("Groups", groups_u, groups_c)
print("Angles", angles_u, angles_c)

info_u = {
    "cells_x": cells_x,
    "angles": angles_u,
    "groups": groups_u,
    "materials": 2,
    "geometry": 1,
    "spatial": 2,
    "bc_x": [0, 0],
    "steps": steps,
    "dt": 1e-8,
}

info_c = {
    "cells_x": cells_x,
    "angles": angles_c,
    "groups": groups_c,
    "materials": 2,
    "geometry": 1,
    "spatial": 2,
    "bc_x": [0, 0],
    "steps": steps,
    "dt": 1e-8,
}


# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)
velocity_u = ants.energy_velocity(groups_u, edges_g)
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

# Angular
angle_xu, angle_wu = ants.angular_x(info_u)
angle_xc, angle_wc = ants.angular_x(info_c)

# Medium Map
layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = ants.spatial1d(layers, edges_x)

# Cross Sections - Uncollided
materials = np.array(layers)[:, 1]
xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials)
# Cross Sections - Collided
xs_collided = hytools.coarsen_materials(
    xs_total_u, xs_scatter_u, xs_fission_u, edges_g[edges_gidx_u], edges_gidx_c
)
xs_total_c, xs_scatter_c, xs_fission_c = xs_collided

# External and boundary sources
external = np.zeros((1, cells_x, 1, 1))
boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
edges_t = np.linspace(0, steps * info_u["dt"], steps + 1)
boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)


# Indexing Parameters
fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

initial_flux = np.zeros((cells_x, angles_u, groups_u))

# Run Hybrid Method
hy_flux = hybrid1d.backward_euler(
    initial_flux,
    xs_total_u.copy(),
    xs_total_c.copy(),
    xs_scatter_u.copy(),
    xs_scatter_c.copy(),
    xs_fission_u.copy(),
    xs_fission_c.copy(),
    velocity_u.copy(),
    velocity_c,
    external.copy(),
    boundary_x,
    medium_map,
    delta_x,
    angle_xu,
    angle_xc,
    angle_wu,
    angle_wc,
    fine_idx,
    coarse_idx,
    factor,
    info_u,
    info_c,
)

groups_range = np.array([groups_c] * steps, dtype=np.int32)
angles_range = np.array([angles_c] * steps, dtype=np.int32)

# Run Hybrid Method
vhy_flux = vhybrid1d.backward_euler(
    groups_range,
    angles_range,
    initial_flux * 0.0,
    xs_total_u,
    xs_scatter_u,
    xs_fission_u,
    velocity_u,
    external,
    boundary_x,
    medium_map,
    delta_x,
    angle_xu,
    angle_wu,
    edges_g,
    info_u,
    info_c,
)

print(np.sum(hy_flux, axis=(1, 2)))
print(np.sum(vhy_flux, axis=(1, 2)))


for tt in range(steps):
    print(
        tt,
        np.sum(np.fabs(hy_flux[tt] - vhy_flux[tt])),
        np.max(np.fabs(hy_flux[tt] - vhy_flux[tt])),
    )
# np.save(
#     f"hybrid_uranium_slab_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}_flux",
#     flux,
# )
