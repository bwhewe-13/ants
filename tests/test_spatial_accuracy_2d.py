########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 2D Spatial Discretization Schemes. Uses
# Method of Manufactured Solutions for testing the step, diamond
# difference (dd), and step characteristic (sc) methods.
#
########################################################################

import numpy as np
import pytest

from ants.fixed2d import fixed_source
from ants.utils import manufactured_2d as mms
from ants.utils import pytools as tools
from tests import problems2d

ANGULAR = [True, False]
SPATIAL = [1, 2, 3]
EDGES = [0, 1]
PARAMETERS = [
    (angular, edges, spatial)
    for angular in ANGULAR
    for edges in EDGES
    for spatial in SPATIAL
]


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_01(angular, edges, spatial):
    cells = np.array([200, 400, 600]) if spatial == 1 else np.array([100, 200, 400])
    order = 1 if spatial == 1 else 2
    atol_c = 5e-2 if spatial == 1 else 5e-3
    atol_e = 5e-2 if spatial == 1 else 2e-2
    errors_c, errors_x, errors_y = [], [], []
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_01(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = spatial
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if edges:
            solver.flux_at_edges = 1
            flux_x, flux_y = fixed_source(
                mat_data, sources, geometry, quadrature, solver
            )
            flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])
            exact_x = mms.solution_ss_01(
                edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
            )
            exact_y = mms.solution_ss_01(
                centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
            )
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_x = np.sum(
                    exact_x * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_y = np.sum(
                    exact_y * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
            errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
            errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
        else:
            flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux, exact_c, ndims=2))
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - order)
            < atol_c
        )
        if edges:
            assert (
                abs(
                    tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio)
                    - order
                )
                < atol_e
            )
            assert (
                abs(
                    tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio)
                    - order
                )
                < atol_e
            )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_02(angular, edges, spatial):
    cells = np.array([100, 200, 400])
    order = 1 if spatial == 1 else 2
    atol_c = 5e-2 if spatial == 1 else 5e-3
    atol_e = 5e-2 if spatial == 1 else 2e-2
    errors_c, errors_x, errors_y = [], [], []
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_02(ii, 2)
        )
        solver.angular = angular
        geometry.space_disc = spatial
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if edges:
            solver.flux_at_edges = 1
            flux_x, flux_y = fixed_source(
                mat_data, sources, geometry, quadrature, solver
            )
            flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])
            exact_x = mms.solution_ss_02(
                edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
            )
            exact_y = mms.solution_ss_02(
                centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
            )
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_x = np.sum(
                    exact_x * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_y = np.sum(
                    exact_y * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
            errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
            errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
        else:
            flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux, exact_c, ndims=2))
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - order)
            < atol_c
        )
        if edges:
            assert (
                abs(
                    tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio)
                    - order
                )
                < atol_e
            )
            assert (
                abs(
                    tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio)
                    - order
                )
                < atol_e
            )


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_04(angular, edges, spatial):
    cells = np.array([100, 200, 400])
    n_angles = 4 if spatial == 1 else 6
    order = 1 if spatial == 1 else 2
    atol_c = 5e-2 if spatial == 1 else 5e-3
    atol_e = 5e-2 if spatial == 1 else 2e-2
    errors_c, errors_x, errors_y = [], [], []
    for ii in cells:
        mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
            problems2d.manufactured_ss_04(ii, n_angles)
        )
        solver.angular = angular
        geometry.space_disc = spatial
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact_c = mms.solution_ss_04(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
        )
        if edges:
            solver.flux_at_edges = 1
            flux_x, flux_y = fixed_source(
                mat_data, sources, geometry, quadrature, solver
            )
            flux_c = 0.25 * (flux_x[:-1] + flux_x[1:] + flux_y[:, 1:] + flux_y[:, :-1])
            exact_x = mms.solution_ss_04(
                edges_x, centers_y, quadrature.angle_x, quadrature.angle_y
            )
            exact_y = mms.solution_ss_04(
                centers_x, edges_y, quadrature.angle_x, quadrature.angle_y
            )
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_x = np.sum(
                    exact_x * quadrature.angle_w[None, None, :, None], axis=2
                )
                exact_y = np.sum(
                    exact_y * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux_c, exact_c, ndims=2))
            errors_x.append(tools.spatial_error(flux_x, exact_x, ndims=2))
            errors_y.append(tools.spatial_error(flux_y, exact_y, ndims=2))
        else:
            flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
            if not angular:
                exact_c = np.sum(
                    exact_c * quadrature.angle_w[None, None, :, None], axis=2
                )
            errors_c.append(tools.spatial_error(flux, exact_c, ndims=2))
    for err in range(len(errors_c) - 1):
        ratio = cells[err + 1] / cells[err]
        assert (
            abs(tools.order_accuracy(errors_c[err], errors_c[err + 1], ratio) - order)
            < atol_c
        )
        if edges:
            assert (
                abs(
                    tools.order_accuracy(errors_x[err], errors_x[err + 1], ratio)
                    - order
                )
                < atol_e
            )
            assert (
                abs(
                    tools.order_accuracy(errors_y[err], errors_y[err + 1], ratio)
                    - order
                )
                < atol_e
            )
