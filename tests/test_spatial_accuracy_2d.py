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
from ants.utils import pytools as tools
from tests import problems2d


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
            centers_x, centers_y = problems2d.manufactured_ss_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_01(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


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
            centers_x, centers_y = problems2d.manufactured_ss_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_01(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_manufactured_02(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_ss_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_02(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_diamond_manufactured_02(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_ss_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_02(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_manufactured_04(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_ss_04(ii, 4)
        info["angular"] = angular
        info["spatial"] = 1
        # Run Source Iteration
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_04(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_diamond_manufactured_04(angular):
    errors = []
    cells = np.array([200, 400, 800])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, boundary_y, \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info, \
            centers_x, centers_y = problems2d.manufactured_ss_04(ii, 6)
        info["angular"] = angular
        info["spatial"] = 2
        # Run Source Iteration 
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                                boundary_x, boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        exact = mms.solution_ss_04(centers_x, centers_y, angle_x, angle_y)
        # Rearrange dimensions
        if not angular:
            exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol
