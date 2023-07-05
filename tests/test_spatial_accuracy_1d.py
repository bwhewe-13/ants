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
from ants.fixed1d import source_iteration
from ants.utils import manufactured_1d as mms
from tests import problems1d


def _error(approx, reference):
    assert approx.shape == reference.shape, "Not the same array shape"
    cells_x = approx.shape[0]
    return cells_x**(-0.5) * np.linalg.norm(approx - reference)


def _order_accuracy(error1, error2, ratio):
    # error2 is for refined spatial grid
    # ratio is h1 / h2
    return np.log(error1 / error2) / np.log(ratio)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_step_manufactured_01(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_01(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_diamond_manufactured_01(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_01(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_step_manufactured_02(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_02(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_diamond_manufactured_02(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_02(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_step_manufactured_03(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_03(ii, 8)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_03(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_diamond_manufactured_03(angular, edges):
    errors = []
    cells = np.array([100, 1000, 10_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_03(ii, 8)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_03(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_step_manufactured_04(angular, edges):
    errors = []
    cells = np.array([200, 2000, 20_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_04(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_04(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_diamond_manufactured_04(angular, edges):
    errors = []
    cells = np.array([200, 2000, 20_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_04(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_04(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_step_manufactured_05(angular, edges):
    errors = []
    cells = np.array([200, 2000, 20_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_05(ii, 8)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_05(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_diamond_manufactured_05(angular, edges):
    errors = []
    cells = np.array([200, 2000, 20_000])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_05(ii, 8)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_mms_05(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(_order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol