########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
# 
# Test Order of Accuracy for 1D Spatial Discretization Schemes. Uses
# Method of Manufactured Solutions for testing the step, diamond 
# difference (dd), and step characteristic (sc) methods.
# 
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed1d import source_iteration
from ants.utils import manufactured_1d as mms
from ants.utils import pytools as tools
from tests import problems1d


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
            = problems1d.manufactured_ss_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_01(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_dd_manufactured_01(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_01(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_sc_manufactured_01(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_01(ii, 2)
        info["angular"] = angular
        info["spatial"] = 3
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_01(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


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
            = problems1d.manufactured_ss_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_02(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_dd_manufactured_02(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_02(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_sc_manufactured_02(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_02(ii, 2)
        info["angular"] = angular
        info["spatial"] = 3
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_02(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], ratio) - 2) < atol


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
            = problems1d.manufactured_ss_03(ii, 8)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_03(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_dd_manufactured_03(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_03(ii, 8)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_03(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_sc_manufactured_03(angular, edges):
    errors = []
    # cells = np.array([100, 1000, 10_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_03(ii, 8)
        info["angular"] = angular
        info["spatial"] = 3
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_03(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


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
            = problems1d.manufactured_ss_04(ii, 2)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_04(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_dd_manufactured_04(angular, edges):
    errors = []
    # cells = np.array([200, 2000, 20_000])
    cells = np.array([100, 200, 400])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_04(ii, 2)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_04(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_sc_manufactured_04(angular, edges):
    errors = []
    # cells = np.array([200, 2000, 20_000])
    cells = np.array([100, 200, 400])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_04(ii, 2)
        info["angular"] = angular
        info["spatial"] = 3
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_04(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


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
            = problems1d.manufactured_ss_05(ii, 8)
        info["angular"] = angular
        info["spatial"] = 1
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_05(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 1) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_dd_manufactured_05(angular, edges):
    errors = []
    # cells = np.array([200, 2000, 20_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_05(ii, 8)
        info["angular"] = angular
        info["spatial"] = 2
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_05(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges"), [(True, 0), (True, 1), \
                         (False, 0), (False, 1)])
def test_sc_manufactured_05(angular, edges):
    errors = []
    # cells = np.array([200, 2000, 20_000])
    cells = np.array([50, 100, 200])
    for ii in cells:
        xs_total, xs_scatter, xs_fission, external, boundary_x, medium_map, \
            delta_x, angle_x, angle_w, info, edges_x, centers_x \
            = problems1d.manufactured_ss_05(ii, 8)
        info["angular"] = angular
        info["spatial"] = 3
        info["edges"] = edges
        flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_05(space_x, angle_x)[:,:,None]
        if not angular:
            exact = np.sum(exact * angle_w[None,:,None], axis=1)
        errors.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err+1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err+1], \
                                        ratio) - 2) < atol