########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for fixed source problems (time dependent and time independent)
#
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed1d import source_iteration

@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "boundary"), [(True, [0, 0]), (True, [0, 1]), \
                (True, [1, 0]), (False, [0, 0]), (False, [0, 1]), (False, [1, 0])])
def test_time_independent_reed(angular, boundary):
    info = {"cells_x": 1600, "angles": 2, "groups": 1, "materials": 4, \
            "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": boundary,
            "bcdim_x": 1, "angular": angular}
    if boundary == [0, 0]:
        materials = [[0, "scattering", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"], \
             [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
    elif boundary == [0, 1]:
        materials = [[0, "scattering", "0-4"], [1, "vacuum", "4-5"], \
                     [2, "absorber", "5-6"], [3, "source", "6-8"]]
    elif boundary == [1, 0]:
        materials = [[0, "scattering", "4-8"], [1, "vacuum", "3-4"], \
                     [2, "absorber", "2-3"], [3, "source", "0-2"]]
    # Spatial
    length = 8. if np.sum(boundary) > 0.0 else 16.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    # Medium Map
    medium_map = ants._medium_map(materials, edges_x)
    # Angular
    angle_x, angle_w = ants._angle_x(info)
    # Cross Sections
    xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
    xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
    xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])
    # External and boundary sources
    external = ants.externals("reeds", (info["cells_x"], info["angles"], info["groups"]), \
                              edges_x=edges_x, bc=info["bc_x"]).flatten()
    boundary_x = np.zeros((2,))
    # Calculate Diamond Difference
    diamond = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    if info["angular"]:
        diamond = np.sum(diamond[:,:,0] * angle_w[None,:], axis=1)
    # Calculate Step Method
    info["spatial"] = 1
    step = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    if info["angular"]:
        step = np.sum(step[:,:,0] * angle_w[None,:], axis=1)
    # Compare discretization methods
    assert np.all(np.isclose(diamond, step, atol=1e-1)), "flux not accurate"
