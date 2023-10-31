########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Built-in External Sources for Two-Dimensional Problems
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

import ants
from ants.utils import pytools as tools

DATA_PATH = pkg_resources.resource_filename("ants","sources/")


def manufactured_ss_03(x, y, angle_x, angle_y):
    # Angular dependent source
    external = np.zeros((x.shape[0], y.shape[0], angle_x.shape[0], 1))
    # Iterate over angles
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        external[...,nn,0] = 0.5 * (np.exp(-1) - np.exp(1)) + np.exp(mu) + np.exp(eta)

    return external


def manufactured_ss_04(x, y, angle_x, angle_y):
    # Spatial and angular dependent source
    external = np.zeros((x.shape[0], y.shape[0], angle_x.shape[0], 1))

    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        external[...,nn,0] = 1 + 0.1 * np.exp(mu) * mesh_x**2 \
                     + 0.1 * np.exp(eta) * mesh_y**2 + 0.025 * np.exp(-1) \
                     * (-20 * np.exp(1) + mesh_x**2 + mesh_y**2 - np.exp(2) \
                     * (mesh_x**2 + mesh_y**2)) + 0.2 * (mu * mesh_x * np.exp(mu) \
                     + eta * mesh_y * np.exp(eta))

    return external


def manufactured_td_01(x, y, angle_x, angle_y, edges_t):
    external = np.zeros((edges_t.shape[0], x.shape[0], y.shape[0], \
                         angle_x.shape[0], 1))

    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")

    for cc, tt in enumerate(edges_t):
        for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
            pp1 = 3 + (4 * mu - 2) * np.cos(0.5 * (tt - 2 * mesh_x))
            pp2 = 3 * np.cos(0.25 * (tt - 4 * mesh_y)) + 4 * np.cos(mu)
            pp3 = -np.sin(1) - 3 * np.sin(0.5 * (tt - 2 * mesh_x))
            pp4 = (4 * eta  - 1) * np.sin(0.25 * (tt - 4 * mesh_y))
            pp5 = 4 * np.sin(eta)
            external[cc,...,nn,0] = 0.25 * (pp1 + pp2 + pp3 + pp4 + pp5)

    return external


def ambe(edges_x, edges_y, coordinates, edges_g):
    external = np.zeros((edges_x.shape[0] - 1, edges_y.shape[0] - 1, \
                         1, edges_g.shape[0]))

    data = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    # Convert to MeV
    if np.max(edges_g) > 20.0:
        edges_g *= 1E-6
    # Get energy spectra of AmBe source
    value = tools.resize_array_1d(edges_g, data["edges"], data["magnitude"])
    # Put in location
    external = ants.spatial2d(external, value, coordinates, edges_x, edges_y)
    return external

