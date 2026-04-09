########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 2D Spatial Discretization Schemes. Uses
# Method of Manufactured Solutions for testing diamond difference and
# step discretization method.
#
########################################################################

import numpy as np
import pytest

from ants.fixed2d import fixed_source
from ants.utils import manufactured_2d as mms
from ants.utils import pytools as tools
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_center_01(angular):
    errors = []
    cells = np.array([200, 400, 600])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_01(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 1
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 2e-2
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 1) < atol


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_edge_01(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([200, 400, 600])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_01(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 1
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_01(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_01(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 2e-2
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 1)
            < atol
        )


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_center_01(angular):
    errors = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_01(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 2
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 2) < atol


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_edge_01(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([200, 400, 600])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_01(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 2
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_01(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_01(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 2e-2
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 2)
            < atol
        )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_center_02(angular):
    errors = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_02(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 1
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-2
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_edge_02(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_02(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 1
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_02(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_02(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 5e-2
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 1)
            < atol
        )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_center_02(angular):
    errors = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_02(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 2
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_edge_02(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_02(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = 2
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_02(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_02(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 5e-3
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 2)
            < atol
        )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_center_04(angular):
    errors = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_04(ii, 4)
        )
        solver.angular = angular
        geometry.space_disc = 1
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_04(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-2
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 1) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_step_mms_edge_04(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_04(ii, 4)
        )
        solver.angular = angular
        geometry.space_disc = 1
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_04(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_04(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_04(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 5e-2
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 1)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 1)
            < atol
        )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_center_04(angular):
    errors = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_04(ii, 6)
        )
        solver.angular = angular
        geometry.space_disc = 2
        flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_ss_04(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
        errors.append(tools.spatial_error(flux, exact, ndims=2))
    atol = 5e-3
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errors[err], errors[err + 1], ratio) - 2) < atol


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular"), [(True), (False)])
def test_dd_mms_edge_04(angular):
    errors_c = []
    errors_x = []
    errors_y = []
    cells = np.array([100, 200, 400])
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_04(ii, 6)
        )
        solver.angular = angular
        geometry.space_disc = 2
        solver.flux_at_edges = 1
        flux_x, flux_y = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_04(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_x = mms.solution_ss_04(
            edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        exact_y = mms.solution_ss_04(
            centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
        )
        if not angular:
            exact_c = np.sum(exact_c * quadrature.angle_w[None, None, :, None], axis=2)
            exact_x = np.sum(exact_x * quadrature.angle_w[None, None, :, None], axis=2)
            exact_y = np.sum(exact_y * quadrature.angle_w[None, None, :, None], axis=2)
        errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
        errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
        errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
    atol = 5e-3
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio) - 2)
            < atol
        )
        assert (
            abs(tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio) - 2)
            < atol
        )
