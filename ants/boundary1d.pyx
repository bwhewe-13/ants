########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Built-in Boundary Conditions for One-Dimensional Problems
# 
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language = c++
# cython: profile=True

import numpy as np

from libc.math cimport pow, erfc, ceil

def manufactured_ss_03(angle_x):
    # One group, angle dependent boundary
    boundary_x = np.zeros((2, angle_x.shape[0], 1))
    boundary_x[0,:,0] = 0.5
    boundary_x[1,:,0] = 0.5 + 0.25 * np.exp(angle_x)
    return boundary_x


def manufactured_ss_04():
    # One group, angle independent boundary
    length_x = 2.
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[1] = 0.5 * length_x**2 + 0.125 * length_x
    return boundary_x


def manufactured_ss_05():
    # One group, angle independent boundary
    length_x = 2.
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[1] = length_x**3
    return boundary_x


def manufactured_td_02(angle_x, edges_t):
    # Time dependent, one group, angle dependent boundary
    length_x = np.pi
    boundary_x = np.zeros((edges_t.shape[0], 2, angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            boundary_x[cc,0,nn,0] = 1 + np.sin(0. - 0.5 * tt) + np.cos(mu)
            boundary_x[cc,1,nn,0] = 1 + np.sin(length_x - 0.5 * tt) + np.cos(mu)
    return boundary_x


def deuterium_deuterium(location, edges_g):
    # Source entering from 2.45 MeV
    group = np.argmin(abs(edges_g - 2.45E6))
    boundary_x = np.zeros((2, 1, edges_g.shape[0] - 1))
    boundary_x[(location, ..., group)] = 1.0
    return boundary_x


def deuterium_tritium(location, edges_g):
    # Source entering from 14.1 MeV
    group = np.argmin(abs(edges_g - 14.1E6))
    boundary_x = np.zeros((2, 1, edges_g.shape[0] - 1))
    boundary_x[(location, ..., group)] = 1.0
    return boundary_x


def time_dependence_constant(boundary_x):
    return boundary_x[None,...]


def time_dependence_decay_01(boundary_x, edges_t, off_time):
    # Turn off boundary at specific step
    steps = edges_t.shape[0] - 1
    boundary_x = np.repeat(boundary_x[None,...], steps, axis=0)
    loc = np.argwhere(edges_t[1:] > off_time).flatten()
    boundary_x[loc,...] *= 0.0
    return boundary_x


def time_dependence_decay_02(boundary_x, edges_t):
    # Turn off boundary by decay 
    steps = edges_t.shape[0] - 1
    # Find where boundary != 0
    idx = tuple(np.argwhere(boundary_x != 0.0).flatten())
    # Repeat over all groups
    boundary_x = np.repeat(boundary_x[None,...], edges_t.shape[0] - 1, axis=0)
    for tt in range(steps):
        dt = edges_t[tt+1] - edges_t[tt]
        # Convert to microseconds
        t_us = np.round(edges_t[tt+1] * 1e6, 12)
        if t_us >= 0.2:
            k = ceil((t_us - 0.2) / 0.1)
            err_arg = (t_us - 0.1 * (1 + k)) / (0.01)
            boundary_x[tt][idx] = pow(0.5, k) * (1 + 2 * erfc(err_arg))
    return boundary_x
