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

import numpy as np
import pytest

from ants.fixed2d import fixed_source
from ants.utils import manufactured_2d as mms
from tests import problems2d

ANGULAR = [True, False]
SPATIAL = [1, 2]
PARAMETERS = [(angular, spatial) for angular in ANGULAR for spatial in SPATIAL]


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_01(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_01(200, 2)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_02(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_02(200, 2)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_03(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_03(200, 4)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_03(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_04(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_04(200, 4)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_04(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"
