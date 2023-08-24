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
from ants.utils.hybrid import hybrid_coarsen_materials as coarsen

# Path for reference solutions
PATH = "data/references_multigroup/"

@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.backward_euler
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), \
                        (8, 43), (2, 43)])
def test_slab_01_backward_euler(angles_c, groups_c):
    # General Parameters
    cells = 1000
    angles_u = 8
    groups_u = 87
    steps = 5
    # Uncollided flux dictionary
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
            "bcdecay_x": 2
            }
    # Collided flux dictionary
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
            "bcdecay_x": 2
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
    layers = [[0, "stainless-steel-440", "0-4, 6-10"], \
                 [1, "uranium-%20%", "4-6"]]
    medium_map = ants.spatial1d(layers, edges_x)
    # Cross Sections - Uncollided
    materials = np.array(layers)[:,1]
    xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(groups_u, materials)
    # Cross Sections - Collided
    xs_total_c, xs_scatter_c, xs_fission_c = coarsen(xs_total_u, \
                        xs_scatter_u, xs_fission_u, edges_g, edges_gidx)
    # External and boundary sources
    external = ants.externals1d(0.0, (cells * angles_u * groups_u,))
    boundary_x = ants.boundaries1d("14.1-mev", (2, groups_u), [0], \
                                 energy_grid=edges_g).flatten()
    # Run Hybrid Method
    flux = hybrid1d.backward_euler(xs_total_u, xs_scatter_u, xs_fission_u, \
                        xs_total_c, xs_scatter_c, xs_fission_c, velocity, \
                        external, boundary_x, medium_map, delta_x, edges_g, \
                        edges_gidx, info_u, info_c)
    # Load Reference flux
    params = f"g87g{groups_c}_n8n{angles_c}_flux.npy"
    reference = np.load(PATH + "hybrid_uranium_slab_backward_euler_" + params)
    # Compare each time step
    for tt in range(steps):
        assert np.isclose(flux[tt], reference[tt]).all()
