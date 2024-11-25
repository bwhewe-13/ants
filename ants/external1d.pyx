########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Built-in External Sources for One-Dimensional Problems
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
import pkg_resources

from ants.utils import pytools as tools

DATA_PATH = pkg_resources.resource_filename("ants","sources/")

def manufactured_ss_03(x, angle_x):
    # One group, angle dependent source
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    const1 = 0.5
    const2 = 0.25
    func = lambda mu: const2 * mu * np.exp(mu) * 2 * x + const1 + const2 \
                    * x**2 * np.exp(mu) - 0.9 / 2 * (2 * const1 + const2 \
                    * x**2 * (np.exp(1) - np.exp(-1)))

    for nn, mu in enumerate(angle_x):
        external[:,nn,0] = func(mu)
    
    return external


def manufactured_ss_04(x, angle_x):
    # One group, two material source
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    length_x = 2.
    
    def _quasi(x, mu):
        c = 0.3
        return 2 * length_x * mu - 4 * x * mu - 2 * x**2 \
               + 2 * length_x * x - c * (-2 * x**2 + 2 * length_x * x)
    
    def _scatter(x, mu):
        c = 0.9
        const = -0.125 * length_x + 0.5 * length_x**2
        return 0.25 * (mu + x) + const - c * ((0.25 * x + const))
    
    for nn, mu in enumerate(angle_x):
        idx = (x < (0.5 * length_x))
        external[idx,nn,0] = _quasi(x[idx], mu)
        idx = (x > (0.5 * length_x))
        external[idx,nn,0] = _scatter(x[idx], mu)

    return external


def manufactured_ss_05(x, angle_x):
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    length_x = 2.

    def _quasi(x, mu):
        c = 0.3
        return mu * (2 * length_x**2 - 4 * np.exp(mu) * x) - 2 * np.exp(mu) \
                * x**2 + 2 * length_x**2 * x - c / 2 * (-2 * x**2 \
                * (np.exp(1) - np.exp(-1)) + 4 * length_x**2 * x)
    
    def _scatter(x, mu):
        c = 0.9
        const = length_x**3 - length_x**2 * np.exp(mu)
        return length_x * mu * np.exp(mu) + length_x * x * np.exp(mu) + const \
                - c/2 * (2 * length_x**3 + (np.exp(1) - np.exp(-1)) \
                * (x * length_x - length_x**2))
    
    for nn, mu in enumerate(angle_x):
        idx = (x < (0.5 * length_x))
        external[idx,nn,0] = _quasi(x[idx], mu)
        idx = (x > (0.5 * length_x))
        external[idx,nn,0] = _scatter(x[idx], mu)
    
    return external


def manufactured_td_01(x, angle_x, edges_t):
    external = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            external[cc,:,nn,0] = 2 + x * (-0.2 - x * (-0.1 + mu) + 2 * mu) \
                                * np.cos(0.1 * tt - x) + ((-2 + x) * x + 2 \
                                * (x - 1) * mu) * np.sin(0.1 * tt - x)
    return external


def manufactured_td_02(x, angle_x, edges_t):
    external = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            external[cc,:,nn,0] = 1 + (mu - 0.5) * np.cos(0.5 * tt - x) \
                                    + np.cos(mu) - np.sin(0.5 * tt - x)
    return external


def reeds(edges_x, bc):
    # Spatial dependent source for Reeds Problem
    external = np.zeros((edges_x.shape[0] - 1, 1, 1))

    source = np.array([0.0, 1.0, 0.0, 50.0, 50.0, 0.0, 1.0, 0.0])
    bounds = np.array([0., 2., 4., 6., 8., 10., 12., 14., 16.])

    if np.sum(bc) > 0.0:
        idx = slice(0, 4) if bc == [0, 1] else slice(4, 8)
        source = source[idx].copy()
        bounds = bounds[0:5].copy()

    right_idx = lambda x: np.argwhere(edges_x >= x)[0,0]
    left_idx = lambda x: np.argwhere(edges_x <= x)[-1,0]

    for ii, (left, right) in enumerate(zip(bounds, bounds[1:])):
        idx = slice(left_idx(left), right_idx(right))
        external[idx] = source[ii]

    return external


def ambe(delta_x, loc_x, edges_g):
    # AmBe source in middle of material
    if isinstance(loc_x, int):
        loc_x = [loc_x]
    
    external = np.zeros((delta_x.shape[0], 1, edges_g.shape[0] - 1))

    data = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    # Convert to MeV
    if np.max(edges_g) > 20.0:
        edges_g *= 1E-6

    centers_g = tools.average_array(edges_g)
    loc_g = lambda x1, x2: np.argwhere((centers_g > x1) & (centers_g <= x2)).flatten()
    
    for ii in range(len(data["magnitude"])):
        idx = loc_g(data["edges"][ii], data["edges"][ii+1])
        for xx in loc_x:
            external[xx, 0, idx] = data["magnitude"][ii] / delta_x[xx]

    return external


def time_dependence_constant(boundary_x):
    return boundary_x[None,...]
