########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for two-dimensional time-dependent problems. Includes tests for
# Backward Euler, Crank-Nicolson, BDF2, and TR-BDF2.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import hybrid2d
from ants.utils import hybrid as hytools
from ants.utils import manufactured_2d as mms
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid2d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()
