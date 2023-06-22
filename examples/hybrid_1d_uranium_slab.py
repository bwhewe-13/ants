
import numpy as np

import ants
from ants.hybrid1d import backward_euler

# General conditions
cells = 1000
angles_u = 8
angles_c = 2
groups_u = 87
groups_c = 43
steps = 5

print("Groups", groups_u, groups_c)
print("Angles", angles_u, angles_c)

info_u = {
            "cells_x": cells,
            "angles": angles_u,
            "groups": groups_u,
            "materials": 2,
            "geometry": 1,
            "spatial": 2,
            "qdim": 3,
            "bc_x": [0, 0],
            "bcdim_x": 2,
            "steps": steps,
            "dt": 1e-8,
            "bcdecay": 2
        }

info_c = {
            "cells_x": cells,
            "angles": angles_c,
            "groups": groups_c,
            "materials": 2,
            "geometry": 1,
            "spatial": 2,
            "qdim": 2,
            "bc_x": [0, 0],
            "bcdim_x": 1,
            "steps": steps,
            "dt": 1e-8,
            "bcdecay": 2
        }


# Spatial
length = 10.
delta_x = np.repeat(length / cells, cells)
edges_x = np.linspace(0, length, cells+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx = ants.energy_grid(groups_c, 87)
velocity = ants.energy_velocity(groups_u, edges_g)

# Angular
angle_x, angle_w = ants.angular_x(info_u)

# Medium Map
materials = [[0, "stainless-steel-440", "0-4, 6-10"], \
             [1, "uranium-%20%", "4-6"]]
medium_map = ants.spatial_map(materials, edges_x)

# Cross Sections
materials = np.array(materials)[:,1]
xs_total, xs_scatter, xs_fission = ants.materials(groups_u, materials)

# External and boundary sources
external = ants.externals1d(0.0, (cells * angles_u * groups_u,))
boundary_x = ants.boundaries1d("14.1-mev", (2, groups_u), [0], \
                             energy_grid=edges_g).flatten()

flux = backward_euler(xs_total, xs_scatter, xs_fission, velocity, \
                external, boundary_x, medium_map, delta_x, edges_g, \
                edges_gidx, info_u, info_c)
np.save(f"hybrid_uranium_slab_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}_flux", flux)