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

import pytest
import numpy as np

import ants
from ants import timed1d
from ants.utils import manufactured_1d as mms
from ants.utils import pytools as tools
from tests import problems1d


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.bdf1
def test_backward_euler_01():
    # Spatial
    cells_x = 200
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.backward_euler(*problems1d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=1))

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.bdf1
def test_backward_euler_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.backward_euler(*problems1d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=1))

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)
        
    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.cn
def test_crank_nicolson_01():
    # Spatial
    cells_x = 200
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.crank_nicolson(*problems1d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=2))

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.cn
def test_crank_nicolson_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.crank_nicolson(*problems1d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=2))

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.bdf2
def test_bdf2_01():
    # Spatial
    cells_x = 200
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [50, 100, 200]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.bdf2(*problems1d.manufactured_td_01(cells_x, \
                                        angles, edges_t, dt, temporal=3))

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.bdf2
def test_bdf2_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.bdf2(*problems1d.manufactured_td_02(cells_x, \
                                        angles, edges_t, dt, temporal=3))

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)
        
    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # Spatial
    cells_x = 200
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.tr_bdf2(*problems1d.manufactured_td_01(cells_x, \
                                        angles, edges_t, dt, temporal=4))

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab1d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = ants.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed1d.tr_bdf2(*problems1d.manufactured_td_02(cells_x, \
                                        angles, edges_t, dt, temporal=4))

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)
        
    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)