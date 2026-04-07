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

import numpy as np
import pytest
from ants.fixed1d import fixed_source

from ants.utils import manufactured_1d as mms
from ants.utils import pytools as tools
from tests import problems1d as prob

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
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_01(angular, edges, spatial):
    errs = []
    cells = np.array([100, 1000, 10_000]) if spatial == 1 else np.array([50, 100, 200])
    order = 1 if spatial == 1 else 2
    for ii in cells:
        mat_data, sources, geo, quadrature, solver = prob.manufactured_ss_01(ii, 2)
        solver.angular = angular
        geo.space_disc = spatial
        solver.flux_at_edges = edges
        flux = fixed_source(mat_data, sources, geo, quadrature, solver)

        edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_01(space_x, quadrature.angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, :, None], axis=1)
        errs.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errs) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errs[err], errs[err + 1], ratio) - order) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_02(angular, edges, spatial):
    errs = []
    cells = np.array([100, 1000, 10_000]) if spatial == 1 else np.array([50, 100, 200])
    order = 1 if spatial == 1 else 2
    for ii in cells:
        mat_data, sources, geo, quadrature, solver = prob.manufactured_ss_02(ii, 2)
        solver.angular = angular
        geo.space_disc = spatial
        solver.flux_at_edges = edges
        flux = fixed_source(mat_data, sources, geo, quadrature, solver)

        edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_02(space_x, quadrature.angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, :, None], axis=1)
        errs.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errs) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errs[err], errs[err + 1], ratio) - order) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_03(angular, edges, spatial):
    errs = []
    cells = np.array([100, 1000, 10_000]) if spatial == 1 else np.array([50, 100, 200])
    order = 1 if spatial == 1 else 2
    for ii in cells:
        mat_data, sources, geo, quadrature, solver = prob.manufactured_ss_03(ii, 12)
        solver.angular = angular
        geo.space_disc = spatial
        solver.flux_at_edges = edges
        flux = fixed_source(mat_data, sources, geo, quadrature, solver)

        edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_03(space_x, quadrature.angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, :, None], axis=1)
        errs.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errs) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errs[err], errs[err + 1], ratio) - order) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_04(angular, edges, spatial):
    errs = []
    cells = np.array([200, 2000, 20_000]) if spatial == 1 else np.array([100, 200, 400])
    order = 1 if spatial == 1 else 2
    for ii in cells:
        mat_data, sources, geo, quadrature, solver = prob.manufactured_ss_04(ii, 2)
        solver.angular = angular
        geo.space_disc = spatial
        solver.flux_at_edges = edges
        flux = fixed_source(mat_data, sources, geo, quadrature, solver)

        edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_04(space_x, quadrature.angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, :, None], axis=1)
        errs.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errs) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errs[err], errs[err + 1], ratio) - order) < atol


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_05(angular, edges, spatial):
    errs = []
    cells = np.array([200, 2000, 20_000]) if spatial == 1 else np.array([50, 100, 200])
    order = 1 if spatial == 1 else 2
    for ii in cells:
        mat_data, sources, geo, quadrature, solver = prob.manufactured_ss_05(ii, 8)
        solver.angular = angular
        geo.space_disc = spatial
        solver.flux_at_edges = edges
        flux = fixed_source(mat_data, sources, geo, quadrature, solver)

        edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()
        exact = mms.solution_ss_05(space_x, quadrature.angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * quadrature.angle_w[None, :, None], axis=1)
        errs.append(tools.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for err in range(len(errs) - 1):
        ratio = cells[err + 1] / cells[err]
        assert abs(tools.order_accuracy(errs[err], errs[err + 1], ratio) - order) < atol
