########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for large multigroup problems (G >= 87) and focuses on fixed
# source, time dependent, and criticality problems.
#
########################################################################

import pytest
import numpy as np

import ants
from ants import hybrid1d
from ants.utils import hybrid as hytools

# Path for reference solutions
PATH = "data/references_multigroup/"

@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), \
                        (8, 43), (2, 43)])
def test_slab_01_bdf1(angles_c, groups_c):
    # General Parameters
    cells_x = 1000
    angles_u = 8
    groups_u = 87
    steps = 5
    # Uncollided flux dictionary
    info_u = {
            "cells_x": cells_x,
            "angles": angles_u,
            "groups": groups_u,
            "materials": 2,
            "geometry": 1,
            "spatial": 2,
            "bc_x": [0, 0],
            "steps": steps,
            "dt": 1e-8
            }
    # Collided flux dictionary
    info_c = {
            "cells_x": cells_x,
            "angles": angles_c,
            "groups": groups_c,
            "materials": 2,
            "geometry": 1,
            "spatial": 2,
            "bc_x": [0, 0],
            "steps": steps,
            "dt": 1e-8
            }
    # Spatial
    length = 10.
    delta_x = np.repeat(length / cells_x, cells_x)
    edges_x = np.linspace(0, length, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    # Energy Grid
    edges_g, edges_gidx = ants.energy_grid(groups_c, 87)
    velocity_u = ants.energy_velocity(groups_u, edges_g)
    velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx)
    # Angular
    angle_xu, angle_wu = ants.angular_x(info_u)
    angle_xc, angle_wc = ants.angular_x(info_c)
    # Medium Map
    layers = [[0, "stainless-steel-440", "0-4, 6-10"], \
                 [1, "uranium-%20%", "4-6"]]
    medium_map = ants.spatial1d(layers, edges_x)
    # Cross Sections - Uncollided
    materials = np.array(layers)[:,1]
    xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(groups_u, materials)
    # Cross Sections - Collided
    xs_total_c, xs_scatter_c, xs_fission_c = hytools.coarsen_materials( \
            xs_total_u, xs_scatter_u, xs_fission_u, edges_g, edges_gidx)
    # External and boundary sources
    initial_flux = np.zeros((cells_x, angles_u, groups_u))
    external = np.zeros((1, cells_x, 1, 1))
    # boundary_x = ants.boundaries1d("14.1-mev", (2, groups_u), [0], \
    #                              energy_grid=edges_g)[None,:,None,:]
    edges_t = np.linspace(0, info_u["dt"] * steps, steps + 1)
    boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
    boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)
    # Indexing Parameters
    fine_idx, coarse_idx, factor = hytools.indexing(groups_u, groups_c, \
                                                    edges_g, edges_gidx)
    # Run Hybrid Method
    flux = hybrid1d.backward_euler(initial_flux, xs_total_u, xs_total_c, \
                xs_scatter_u, xs_scatter_c, xs_fission_u, xs_fission_c, \
                velocity_u, velocity_c, external, boundary_x, medium_map, \
                delta_x, angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, \
                coarse_idx, factor, info_u, info_c)
    # Load Reference flux
    params = f"g87g{groups_c}_n8n{angles_c}_flux.npy"
    reference = np.load(PATH + "hybrid_uranium_slab_backward_euler_" + params)
    # Compare each time step
    for tt in range(steps):
        assert np.isclose(flux[tt], reference[tt]).all()
