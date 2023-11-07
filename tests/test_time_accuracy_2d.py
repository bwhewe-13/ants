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

import pytest
import numpy as np

import ants
from ants import timed2d
from ants.utils import manufactured_2d as mms
from ants.utils import pytools as tools
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_01():
    # Spatial
    cells_x = 100
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = 2.
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 100.

    for steps in [25, 50, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.backward_euler(*problems2d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=1))

        exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_02():
    # Spatial
    cells_x = 50
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 50
    length_y = np.pi
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 10.

    for steps in [20, 40, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.backward_euler(*problems2d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=1))

        exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)
        
    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_01():
    # Spatial
    cells_x = 100
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = 2.
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [20, 40, 60]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.crank_nicolson(*problems2d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=2))

        exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = np.pi
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 50.

    for steps in [25, 50, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.crank_nicolson(*problems2d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=2))

        exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_01():
    # Spatial
    cells_x = 150
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 150
    length_y = 2.
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 175.

    for steps in [60, 80, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.bdf2(*problems2d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=3))

        exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = np.pi
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 10.

    for steps in [20, 40, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.bdf2(*problems2d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=3))

        exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)
        
    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # Spatial
    cells_x = 100
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = 2.
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 200.

    for steps in [15, 30, 45]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.tr_bdf2(*problems2d.manufactured_td_01(\
                                cells_x, angles, edges_t, dt, temporal=4))

        exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = 100
    length_y = np.pi
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angles = 4
    angle_x, angle_y, angle_w = ants.angular_xy(angles)

    error_x = []
    error_y = []
    T = 50.

    for steps in [25, 50, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        approx = timed2d.tr_bdf2(*problems2d.manufactured_td_02(\
                                cells_x, angles, edges_t, dt, temporal=4))

        exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
        exact = np.sum(exact * angle_w[None,None,None,:,None], axis=3)
        
        err = np.linalg.norm(approx[-1] - exact[-1])
        
        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii+1]
        accuracy = tools.order_accuracy(error_y[ii], error_y[ii+1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)