########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test one-dimensional time-independent problems. Includes tests for 
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
from tests import problems1d


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), \
                         (True, 3, 0), (True, 3, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1), \
                         (False, 3, 0), (False, 3, 1)])
def test_manufactured_01(angular, spatial, edges):
    xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
        delta_x, angle_x, angle_w, info, edges_x, centers_x \
        = problems1d.manufactured_ss_01(400, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    info["edges"] = edges
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_01(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"



@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), \
                         (True, 3, 0), (True, 3, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1), \
                         (False, 3, 0), (False, 3, 1)])
def test_manufactured_02(angular, spatial, edges):
    xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
        delta_x, angle_x, angle_w, info, edges_x, centers_x \
        = problems1d.manufactured_ss_02(400, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    info["edges"] = edges
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_02(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), \
                         (True, 3, 0), (True, 3, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1), \
                         (False, 3, 0), (False, 3, 1)])
def test_manufactured_03(angular, spatial, edges):
    xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
        delta_x, angle_x, angle_w, info, edges_x, centers_x \
        = problems1d.manufactured_ss_03(400, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    info["edges"] = edges
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_03(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), \
                         (True, 3, 0), (True, 3, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1), \
                         (False, 3, 0), (False, 3, 1)])
def test_manufactured_04(angular, spatial, edges):
    xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
        delta_x, angle_x, angle_w, info, edges_x, centers_x \
        = problems1d.manufactured_ss_04(400, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    info["edges"] = edges
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_04(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 1e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial", "edges"), [(True, 1, 0), \
                         (True, 1, 1), (True, 2, 0), (True, 2, 1), \
                         (True, 3, 0), (True, 3, 1), (False, 1, 0), \
                         (False, 1, 1), (False, 2, 0), (False, 2, 1), \
                         (False, 3, 0), (False, 3, 1)])
def test_manufactured_05(angular, spatial, edges):
    xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
        delta_x, angle_x, angle_w, info, edges_x, centers_x \
        = problems1d.manufactured_ss_05(400, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    info["edges"] = edges
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_05(space_x, angle_x)
    if not angular:
        exact = np.sum(exact * angle_w[None,:], axis=1)
    atol = 1e-4 if spatial == 2 else 2e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.sphere1d
@pytest.mark.source_iteration
@pytest.mark.multigroup1d
def test_sphere_01_source_iteration():
    flux = source_iteration(*problems1d.sphere_01("fixed"))
    reference = np.load(problems1d.PATH + "uranium_sphere_source_iteration_flux.npy")
    assert np.isclose(flux, reference).all()