########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Method of Manufactured Solutions for One Dimensional Slabs
#
########################################################################

import ants
from ants.fixed1d import source_iteration
from ants.utils import manufactured_solutions as mms

import pytest
import numpy as np

@pytest.mark.smoke
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_01(angular, spatial):
    params = {"cells": 400, "angles": 4, "groups": 1, "materials": 1,
              "geometry": 1, "spatial": spatial, "qdim": 1, "bc": [0, 0],
              "bcdim": 2, "angular": angular}
    length = 1.
    delta_x = np.repeat(length / params["cells"], params["cells"])
    edges_x = np.linspace(0, length, params["cells"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals(1.0, (params["cells"],))
    boundary = ants.boundaries(1.0, (2, params["angles"], params["groups"]), [0])
    angle_x, angle_w = ants._angle_x(params)
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary.flatten(), medium_map, delta_x, \
                            angle_x, angle_w, params)
    exact = mms.solution_mms_01(centers_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.all(np.isclose(flux[(..., 0)], exact, atol=atol)), \
        "Incorrect flux" 


@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_02(angular, spatial):
    params = {"cells": 400, "angles": 4, "groups": 1, "materials": 1,
              "geometry": 1, "spatial": spatial, "qdim": 1, "bc": [0, 0],
              "bcdim": 2, "angular": angular}
    length = 1.
    delta_x = np.repeat(length / params["cells"], params["cells"])
    edges_x = np.linspace(0, length, params["cells"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals(0.5, (params["cells"],))
    boundary = ants.boundaries(1.0, (2, params["angles"], params["groups"]), [0])
    angle_x, angle_w = ants._angle_x(params)
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary.flatten(), medium_map, delta_x, \
                            angle_x, angle_w, params)
    exact = mms.solution_mms_02(centers_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.all(np.isclose(flux[(..., 0)], exact, atol=atol)), \
        "Incorrect flux"


@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_03(angular, spatial):
    params = {"cells": 400, "angles": 4, "groups": 1, "materials": 1,
              "geometry": 1, "spatial": spatial, "qdim": 3, "bc": [0, 0],
              "bcdim": 2, "angular": angular}
    length = 1.
    delta_x = np.repeat(length / params["cells"], params["cells"])
    edges_x = np.linspace(0, length, params["cells"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.9]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals("mms-03", (params["cells"], params["angles"], \
                              params["groups"]), centers_x=centers_x, \
                              angle_x=angle_x).flatten()
    boundary = ants.boundaries("mms-03", (2, params["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, params)
    exact = mms.solution_mms_03(centers_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.all(np.isclose(flux[(..., 0)], exact, atol=atol)), \
        "Incorrect flux"


@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_04(angular, spatial):
    params = {"cells": 400, "angles": 4, "groups": 1, "materials": 2,
              "geometry": 1, "spatial": spatial, "qdim": 3, "bc": [0, 0],
              "bcdim": 2, "angular": angular}
    length = 2.
    delta_x = np.repeat(length / params["cells"], params["cells"])
    edges_x = np.linspace(0, length, params["cells"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals("mms-04", (params["cells"], params["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary = ants.boundaries("mms-04", (2, params["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants._medium_map(materials, edges_x)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, params)
    exact = mms.solution_mms_04(centers_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 1e-2
    assert np.all(np.isclose(flux[(..., 0)], exact, atol=atol)), \
        "Incorrect flux"


@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_05(angular, spatial):
    params = {"cells": 400, "angles": 4, "groups": 1, "materials": 2,
              "geometry": 1, "spatial": spatial, "qdim": 3, "bc": [0, 0],
              "bcdim": 2, "angular": angular}
    length = 2.
    delta_x = np.repeat(length / params["cells"], params["cells"])
    edges_x = np.linspace(0, length, params["cells"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals("mms-05", (params["cells"], params["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary = ants.boundaries("mms-05", (2, params["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants._medium_map(materials, edges_x)
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary, medium_map, delta_x, angle_x, \
                            angle_w, params)
    exact = mms.solution_mms_05(centers_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 2e-2
    assert np.all(np.isclose(flux[(..., 0)], exact, atol=atol)), \
        "Incorrect flux"
