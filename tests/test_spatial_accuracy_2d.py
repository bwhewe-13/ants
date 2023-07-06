########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
# 
# Test Order of Accuracy for 1D Spatial Discretization Schemes. Uses
# Method of Manufactured Solutions for testing diamond difference and 
# step discretization method.
# 
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed2d import source_iteration
from ants.utils import manufactured_2d as mms
from tests import problems2d


def _error(approx, reference):
    assert approx.shape == reference.shape, "Not the same array shape"
    cells_x = approx.shape[0]
    cells_y = approx.shape[1]
    return (cells_x * cells_y)**(-0.5) * np.linalg.norm(approx - reference)


def _order_accuracy(error1, error2, ratio):
    # error2 is for refined spatial grid
    # ratio is h1 / h2
    return np.log(error1 / error2) / np.log(ratio)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_manufactured_01(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_01(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_diamond_manufactured_01(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_01(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_manufactured_02(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_02(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 5e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_diamond_manufactured_02(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_02(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_manufactured_04(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_04(ii, 4)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_04(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        print(_order_accuracy(errors[err], errors[err+1], ratio))
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_diamond_manufactured_04(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_04(ii, 6)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_mms_04(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(_error(flux, exact))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        print(_order_accuracy(errors[err], errors[err+1], ratio))
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol
