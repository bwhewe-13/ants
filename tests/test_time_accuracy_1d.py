########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 1D Temporal Discretization Schemes. Uses
# Method of Manufactured Solutions for testing Backward Euler, Crank-
# Nicolson, BDF2, and TR-BDF2
#
########################################################################

import numpy as np
import pytest

# import ants
from ants import timed1d

# from ants.datatypes import CrossSections, QuadratureData, SpatialGrid
from ants.utils import manufactured_1d as mms
from ants.utils import pytools as tools
from tests import problems1d


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.bdf1
def test_backward_euler_01():
    # General parameters
    cells_x = 200
    angles = 4
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 200.0

    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.bdf1
def test_backward_euler_02():
    # General parameters
    cells_x = 100
    angles = 4
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 20.0
    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.cn
def test_crank_nicolson_01():
    # General parameters
    cells_x = 200
    angles = 4
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 200.0
    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.cn
def test_crank_nicolson_02():
    # General parameters
    cells_x = 100
    angles = 4
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 20.0
    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )
        exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.bdf2
def test_bdf2_01():
    # General parameters
    cells_x = 200
    angles = 4
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 200.0
    for steps in [50, 100, 200]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.bdf2
def test_bdf2_02():
    # General parameters
    cells_x = 100
    angles = 4
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 20.0
    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # General parameters
    cells_x = 200
    angles = 4
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 200.0
    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )

        exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # General parameters
    cells_x = 100
    angles = 4
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    error_x = []
    error_y = []
    T = 20.0
    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        mat_data, sources, geometry, quadrature, solver, time_data = (
            problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)
        )
        approx = timed1d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data
        )
        exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx - exact[-1])
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)
