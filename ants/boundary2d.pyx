########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Built-in Boundary Sources for Two-Dimensional Problems
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np

from libc.math cimport ceil, erfc, pow


def manufactured_ss_01(x, y, angle_x, angle_y):
    boundary_x = 1.5 * np.ones((2, y.shape[0], angle_x.shape[0], 1))
    boundary_y = np.zeros((2, x.shape[0], angle_y.shape[0], 1))

    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0:
            boundary_y[...,nn,0] = 0.5 + np.exp(-x / mu)
        elif mu < 0.0:
            boundary_y[...,nn,0] = 0.5 + np.exp((1 - x) / mu)

    return boundary_x, boundary_y


def manufactured_ss_02(x, y, angle_x, angle_y):
    boundary_x = np.zeros((2, y.shape[0], angle_x.shape[0], 1))
    boundary_y = np.zeros((2, x.shape[0], angle_y.shape[0], 1))

    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0:
            boundary_y[...,nn,0] = 1.5 + 0.5 * np.exp(-x / mu)
        elif mu < 0.0:
            boundary_y[...,nn,0] = 1.5 + 0.5 * np.exp((1 - x) / mu)
        if eta > 0.0:
            boundary_x[...,nn,0] = 1.5 + 0.5 * np.exp(-y / eta)
        elif eta < 0.0:
            boundary_x[...,nn,0] = 1.5 + 0.5 * np.exp((1 - y) / eta)

    return boundary_x, boundary_y


def manufactured_ss_03(x, y, angle_x, angle_y):
    boundary_x = np.zeros((2, 1, angle_x.shape[0], 1))
    boundary_y = np.zeros((2, 1, angle_y.shape[0], 1))

    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        boundary_x[...,nn,0] = np.exp(mu) + np.exp(eta)
        boundary_y[...,nn,0] = np.exp(mu) + np.exp(eta)

    return boundary_x, boundary_y


def manufactured_ss_04(x, y, angle_x, angle_y):
    boundary_x = np.zeros((2, y.shape[0], angle_x.shape[0], 1))
    boundary_y = np.zeros((2, x.shape[0], angle_y.shape[0], 1))

    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0 and eta > 0.0:
            boundary_x[0,:,nn,0] = 1 + 0.1 * np.exp(eta) * y**2
            boundary_y[0,:,nn,0] = 1 + 0.1 * np.exp(mu) * x**2
        elif mu > 0.0 and eta < 0.0:
            boundary_x[0,:,nn,0] = 1 + 0.1 * np.exp(eta) * y**2
            boundary_y[1,:,nn,0] = 1 + 0.4 * np.exp(eta) + 0.1 * np.exp(mu) * x**2
        elif mu < 0.0 and eta > 0.0:
            boundary_x[1,:,nn,0] = 1 + 0.4 * np.exp(mu) + 0.1 * np.exp(eta) * y**2
            boundary_y[0,:,nn,0] = 1 + 0.1 * np.exp(mu) * x**2
        elif mu < 0.0 and eta < 0.0:
            boundary_x[1,:,nn,0] = 1 + 0.4 * np.exp(mu) + 0.1 * np.exp(eta) * y**2
            boundary_y[1,:,nn,0] = 1 + 0.4 * np.exp(eta) + 0.1 * np.exp(mu) * x**2

    return boundary_x, boundary_y


def manufactured_td_02(x, y, angle_x, angle_y, edges_t):
    # Returns half-angle boundary arrays:
    #   boundary_x shape (T, 2, cells_y, half_x, 1):
    #     [t, 0, :, ii, 0] = left  (x=0)  for ii-th incoming angle (angle_x > 0)
    #     [t, 1, :, ii, 0] = right (x=pi) for ii-th incoming angle (angle_x < 0)
    #   boundary_y shape (T, 2, cells_x, half_y, 1):
    #     [t, 0, :, jj, 0] = bottom (y=0)  for jj-th incoming angle (angle_y > 0)
    #     [t, 1, :, jj, 0] = top   (y=pi) for jj-th incoming angle (angle_y < 0)
    pos_x_idx = np.where(angle_x > 0)[0]  # incoming at left boundary
    neg_x_idx = np.where(angle_x < 0)[0]  # incoming at right boundary
    pos_y_idx = np.where(angle_y > 0)[0]  # incoming at bottom boundary
    neg_y_idx = np.where(angle_y < 0)[0]  # incoming at top boundary

    half_x = len(pos_x_idx)
    half_y = len(pos_y_idx)

    # Endpoints
    XX1 = 0.0; XX2 = np.pi
    YY1 = 0.0; YY2 = np.pi

    boundary_x = np.zeros((edges_t.shape[0], 2, y.shape[0], half_x, 1))
    boundary_y = np.zeros((edges_t.shape[0], 2, x.shape[0], half_y, 1))

    for cc, tt in enumerate(edges_t):
        # X boundaries: left (mu > 0) and right (mu < 0)
        for ii, nn in enumerate(pos_x_idx):
            mu, eta = angle_x[nn], angle_y[nn]
            boundary_x[cc, 0, :, ii, 0] = 1 + np.sin(XX1 - 0.5 * tt) + np.cos(mu) \
                                         + np.cos(y - 0.25 * tt) + np.sin(eta)
        for ii, nn in enumerate(neg_x_idx):
            mu, eta = angle_x[nn], angle_y[nn]
            boundary_x[cc, 1, :, ii, 0] = 1 + np.sin(XX2 - 0.5 * tt) + np.cos(mu) \
                                         + np.cos(y - 0.25 * tt) + np.sin(eta)
        # Y boundaries: bottom (eta > 0) and top (eta < 0)
        for jj, nn in enumerate(pos_y_idx):
            mu, eta = angle_x[nn], angle_y[nn]
            boundary_y[cc, 0, :, jj, 0] = 1 + np.sin(x - 0.5 * tt) + np.cos(mu) \
                                         + np.cos(YY1 - 0.25 * tt) + np.sin(eta)
        for jj, nn in enumerate(neg_y_idx):
            mu, eta = angle_x[nn], angle_y[nn]
            boundary_y[cc, 1, :, jj, 0] = 1 + np.sin(x - 0.5 * tt) + np.cos(mu) \
                                         + np.cos(YY2 - 0.25 * tt) + np.sin(eta)

    return boundary_x, boundary_y


def deuterium_deuterium(loc_x, loc_y, edges_g):
    # Source entering from 2.45 MeV
    group = np.argmin(abs(edges_g - 2.45E6))
    boundary_x = np.zeros((2, 1, 1, edges_g.shape[0] - 1))
    boundary_y = np.zeros((2, 1, 1, edges_g.shape[0] - 1))
    if loc_x >= 0:
        boundary_x[(loc_x, ..., group)] = 1.0
    if loc_y >= 0:
        boundary_y[(loc_y, ..., group)] = 1.0
    return boundary_x, boundary_y


def deuterium_tritium(loc_x, loc_y, edges_g):
    # Source entering from 14.1 MeV
    group = np.argmin(abs(edges_g - 14.1E6))
    boundary_x = np.zeros((2, 1, 1, edges_g.shape[0] - 1))
    boundary_y = np.zeros((2, 1, 1, edges_g.shape[0] - 1))
    if isinstance(loc_x, list) or loc_x >= 0:
        boundary_x[(loc_x, ..., group)] = 1.0
    if isinstance(loc_y, list) or loc_y >= 0:
        boundary_y[(loc_y, ..., group)] = 1.0
    return boundary_x, boundary_y


def time_dependence_decay_01(boundary, edges_t, off_time):
    # Turn off boundary at specific step
    steps = edges_t.shape[0] - 1
    boundary = np.repeat(boundary[None,...], steps, axis=0)
    loc = np.argwhere(edges_t[1:] > off_time).flatten()
    boundary[loc,...] *= 0.0
    return boundary


def time_dependence_decay_02(boundary, edges_t):
    # Turn off boundary by decay
    steps = edges_t.shape[0] - 1
    boundary = np.repeat(boundary[None,...], edges_t.shape[0] - 1, axis=0)
    for tt in range(steps):
        dt = edges_t[tt+1] - edges_t[tt]
        # Convert to microseconds
        t_us = edges_t[tt+1] * 1e6
        if t_us >= 0.2:
            k = ceil((t_us - 0.2) / 0.1)
            err_arg = (t_us - 0.1 * (1 + k)) / (0.01)
            boundary[tt] = pow(0.5, k) * (1 + 2 * erfc(err_arg))
    return boundary


def time_dependence_decay_03(boundary, edges_t):
    # Turn on and decay away linearly
    steps = edges_t.shape[0] - 1
    boundary = np.repeat(boundary[None,...], edges_t.shape[0] - 1, axis=0)
    for tt in range(steps):
        # Convert to microseconds
        t_us = edges_t[tt+1] * 1e6
        if (t_us < 10):
            boundary[tt][boundary[tt] != 0.0] = 0.1 * t_us
        elif (t_us >= 10) and (t_us < 20):
            boundary[tt][boundary[tt] != 0.0] = 1.0
        elif (t_us >= 20) and (t_us < 40):
            boundary[tt][boundary[tt] != 0.0] = -0.05 * t_us + 2
        elif (t_us >= 40):
            boundary[tt][boundary[tt] != 0.0] = 0.0
    return boundary
