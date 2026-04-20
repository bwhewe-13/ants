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
