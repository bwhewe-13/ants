########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 2D Temporal Discretization Schemes. Uses
# Method of Manufactured Solutions for testing Backward Euler, Crank-
# Nicolson, BDF2, and TR-BDF2
#
########################################################################

import numpy as np
import pytest

from ants import timed2d
from ants.utils import manufactured_2d as mms
from ants.utils import pytools as tools
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_01():
    cells_x = 100
    angles = 4
    T = 100.0

    error_x = []
    error_y = []

    for steps in [25, 50, 100]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_02():
    cells_x = 50
    angles = 4
    T = 10.0

    error_x = []
    error_y = []

    for steps in [20, 40, 80]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_01():
    cells_x = 100
    angles = 4
    T = 200.0

    error_x = []
    error_y = []

    for steps in [20, 40, 60]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_02():
    cells_x = 100
    angles = 4
    T = 50.0

    error_x = []
    error_y = []

    for steps in [25, 50, 100]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_01():
    cells_x = 150
    angles = 4
    T = 175.0

    error_x = []
    error_y = []

    for steps in [60, 80, 100]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_02():
    cells_x = 100
    angles = 4
    T = 10.0

    error_x = []
    error_y = []

    for steps in [20, 40, 80]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    cells_x = 100
    angles = 4
    T = 200.0

    error_x = []
    error_y = []

    for steps in [15, 30, 45]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_01(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    cells_x = 100
    angles = 4
    T = 50.0

    error_x = []
    error_y = []

    for steps in [25, 50, 100]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)
        )
        approx = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
        centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
        exact = mms.solution_td_02(
            centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
def test_quadrature_not_ordered():
    # The angular flux of each step is found with one sweep of each
    # ordinate, so the angles need to be ordered for the reflecting faces
    import ants
    from ants.datatypes import (
        GeometryData,
        MaterialData,
        SolverData,
        SourceData,
        TimeDependentData,
    )

    cells = 4
    materials = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.zeros((1, 1, 1)),
        velocity=np.ones(1),
    )
    sources = SourceData(
        external=np.ones((1, cells, cells, 1, 1)),
        boundary_x=np.zeros((1, 2, 1, 1, 1)),
        boundary_y=np.zeros((1, 2, 1, 1, 1)),
        initial_flux=np.zeros((cells, cells, 16, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=np.full(cells, 0.5),
        delta_y=np.full(cells, 0.5),
        bc_x=[0, 0],
        bc_y=[1, 0],
        geometry=3,
    )
    time_data = TimeDependentData(steps=1, dt=0.1)
    with pytest.raises(ValueError, match="not ordered for bc_y"):
        timed2d.time_dependent(
            materials, sources, geometry, ants.angular_xy(4), SolverData(), time_data
        )
    quadrature = ants.angular_xy(4, bc_y=[1, 0])
    flux = timed2d.time_dependent(
        materials, sources, geometry, quadrature, SolverData(), time_data
    )
    assert np.all(flux > 0.0)
