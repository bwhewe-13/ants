########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test two-dimensional time-independent problems. Includes method of 
# manufactured solutions for two-dimensional slabs.
#
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed2d import source_iteration
from ants.utils import manufactured_2d as mms
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_01(angular, spatial):
    xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
        medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
        centers_x, centers_y = problems2d.manufactured_01(200, 2)
    info["angular"] = angular
    info["spatial"] = spatial
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_01(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux" 


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_02(angular, spatial):
    xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
        medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
        centers_x, centers_y = problems2d.manufactured_02(200, 2)
    info["angular"] = angular
    info["spatial"] = spatial
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_02(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_03(angular, spatial):
    xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
        medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
        centers_x, centers_y = problems2d.manufactured_03(200, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_03(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_04(angular, spatial):
    xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
        medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
        centers_x, centers_y = problems2d.manufactured_04(200, 4)
    info["angular"] = angular
    info["spatial"] = spatial
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)

    exact = mms.solution_mms_04(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"