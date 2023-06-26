########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Method of Manufactured Solutions for One Dimensional Slabs. Tests are
# for the time-independent solutions for scalar and angular flux, the
# diamond difference and step method, and for calculating at cell edges.
#
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed1d import source_iteration
from ants.utils import manufactured_1d as mms


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1)])
def test_manufactured_01(angular, spatial, edges):
    info = {"cells_x": 400, "angles": 4, "groups": 1, "materials": 1,
            "geometry": 1, "spatial": spatial, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": angular, "edges": edges}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d(1.0, (info["cells_x"] * info["angles"] * info["groups"],))
    boundary = ants.boundaries1d(1.0, (2, info["angles"], info["groups"]), [0])
    angle_x, angle_w = ants.angular_x(info)
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary.flatten(), medium_map, delta_x, \
                            angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_mms_01(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1)])
def test_manufactured_02(angular, spatial, edges):
    info = {"cells_x": 400, "angles": 4, "groups": 1, "materials": 1,
            "geometry": 1, "spatial": spatial, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": angular, "edges": edges}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d(0.5, (info["cells_x"] * info["angles"] * info["groups"],))
    boundary = ants.boundaries1d(1.0, (2, info["angles"], info["groups"]), [0])
    angle_x, angle_w = ants.angular_x(info)
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary.flatten(), medium_map, delta_x, \
                            angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_mms_02(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1)])
def test_manufactured_03(angular, spatial, edges):
    info = {"cells_x": 400, "angles": 4, "groups": 1, "materials": 1,
            "geometry": 1, "spatial": spatial, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": angular, "edges": edges}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.9]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d("mms-03", (info["cells_x"], info["angles"], \
                              info["groups"]), centers_x=centers_x, \
                              angle_x=angle_x).flatten()
    boundary = ants.boundaries1d("mms-03", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_mms_03(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1)])
def test_manufactured_04(angular, spatial, edges):
    info = {"cells_x": 400, "angles": 4, "groups": 1, "materials": 2,
            "geometry": 1, "spatial": spatial, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": angular, "edges": edges}
    length = 2.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals1d("mms-04", (info["cells_x"], info["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary = ants.boundaries1d("mms-04", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial_map(materials, edges_x)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_mms_04(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 1e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1)])
def test_manufactured_05(angular, spatial, edges):
    info = {"cells_x": 400, "angles": 4, "groups": 1, "materials": 2,
            "geometry": 1, "spatial": spatial, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": angular, "edges": edges}
    length = 2.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals1d("mms-05", (info["cells_x"], info["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary = ants.boundaries1d("mms-05", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial_map(materials, edges_x)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_mms_05(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 2e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"

if __name__ == "__main__":
    # test_manufactured_03(True, 1, 0)
    # test_manufactured_03(True, 1, 1)
    # test_manufactured_03(True, 2, 1)
    # test_manufactured_03(True, 2, 0)
    test_manufactured_04(False, 2, 1)
    # test_manufactured_04(False, 2, 1)
