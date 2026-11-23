########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 3D Temporal Discretization Schemes. Uses
# Method of Manufactured Solutions for testing Backward Euler, Crank-
# Nicolson, BDF2, and TR-BDF2
#
########################################################################

import numpy as np
import pytest

from ants import timed3d
from ants.utils import pytools as tools
from tests import problems3d


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.bdf1
def test_backward_euler_01():
    cells = 4
    angles = 4
    T = 1.0

    error_x = []
    error_y = []

    for steps in [10, 20, 40]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems3d.manufactured_td_01(cells, angles, edges_t, dt, temporal=1)
        )
        approx = timed3d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        exact = problems3d.solution_td_01(
            centers_x, centers_x, centers_x, quadrature.angle_x, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[:, None], axis=4)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.cn
def test_crank_nicolson_01():
    cells = 4
    angles = 4
    T = 1.0

    error_x = []
    error_y = []

    for steps in [10, 20, 40]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems3d.manufactured_td_01(cells, angles, edges_t, dt, temporal=2)
        )
        approx = timed3d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        exact = problems3d.solution_td_01(
            centers_x, centers_x, centers_x, quadrature.angle_x, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[:, None], axis=4)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.bdf2
def test_bdf2_01():
    cells = 4
    angles = 4
    T = 1.0

    error_x = []
    error_y = []

    for steps in [10, 20, 40]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems3d.manufactured_td_01(cells, angles, edges_t, dt, temporal=3)
        )
        approx = timed3d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        exact = problems3d.solution_td_01(
            centers_x, centers_x, centers_x, quadrature.angle_x, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[:, None], axis=4)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    cells = 4
    angles = 4
    T = 1.0

    error_x = []
    error_y = []

    for steps in [10, 20, 40]:
        dt = T / steps
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems3d.manufactured_td_01(cells, angles, edges_t, dt, temporal=4)
        )
        approx = timed3d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        exact = problems3d.solution_td_01(
            centers_x, centers_x, centers_x, quadrature.angle_x, edges_t[1:]
        )
        exact = np.sum(exact * quadrature.angle_w[:, None], axis=4)

        error_y.append(np.linalg.norm(approx - exact[-1]))
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)
