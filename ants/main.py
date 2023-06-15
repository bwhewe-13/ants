########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# The main driver for the ants package. This file is automatically 
# loaded with the __init__ file.
# 
########################################################################

from ants.constants import *
from ants.utils import dimensions

import numpy as np
import pkg_resources
import warnings

DATA_PATH = pkg_resources.resource_filename("ants","sources/energy/")

def _angle_x(params):
    angle_x, angle_w = np.polynomial.legendre.leggauss(params["angles"])
    angle_w /= np.sum(angle_w)
    # Ordering for reflective boundaries
    if np.sum(params["bc_x"]) > 0.0:
        if params["bc_x"] == [1, 0]:
            idx = angle_x.argsort()
        elif params["bc_x"] == [0, 1]:
            idx = angle_x.argsort()[::-1]
        angle_x = angle_x[idx].copy()
        angle_w = angle_w[idx].copy()
    return angle_x, angle_w

def _angle_xy(params, rewrite=True):
    angles = params["angles"]
    bc = [params["bc_x"], params["bc_y"]]
    # eta, xi, mu: direction cosines (x,y,z) 
    xx, wx = np.polynomial.legendre.leggauss(angles)
    yy, wy = np.polynomial.chebyshev.chebgauss(angles)
    idx = 0
    eta = np.zeros(2 * angles**2)
    xi = np.zeros(2 * angles**2)
    mu = np.zeros(2 * angles**2)
    w = np.zeros(2 * angles**2)
    for ii in range(angles):
        for jj in range(angles):
            mu[idx:idx+2] = xx[ii]
            eta[idx] = np.sqrt(1 - xx[ii]**2) * np.cos(np.arccos(yy[jj]))
            eta[idx+1] = np.sqrt(1 - xx[ii]**2) * np.cos(-np.arccos(yy[jj]))
            xi[idx] = np.sqrt(1 - xx[ii]**2) * np.sin(np.arccos(yy[jj]))
            xi[idx+1] = np.sqrt(1 - xx[ii]**2) * np.sin(-np.arccos(yy[jj]))
            w[idx:idx+2] = wx[ii] * wy[jj]
            idx += 2
    w, eta, xi = _ordering_xy_angles(w[mu > 0] / np.sum(w[mu > 0]), \
                                        eta[mu > 0], xi[mu > 0], bc)
    # Convert to naming convention
    angle_w = w.copy()
    angle_x = eta.copy()
    angle_y = xi.copy()
    if rewrite:
        params["angles"] = len(angle_x)
    return angle_x, angle_y, angle_w

def _ordering_xy_angles(w, nx, ny, bc):
    angles = np.vstack((w, nx, ny))
    if np.sum(bc) == 1:
        if bc[0] == [0, 1]:
            angles = angles[:,angles[1].argsort()[::-1]]
        elif bc[0] == [1, 0]:
            angles = angles[:,angles[1].argsort()]
        elif bc[1] == [0, 1]:
            angles = angles[:,angles[2].argsort()[::-1]]
        elif bc[1] == [1, 0]:
            angles = angles[:,angles[2].argsort()]
    elif np.sum(bc) == 2:
        if bc[0] == [0, 1] and bc[1] == [0, 1]:
            angles = angles[:,angles[1].argsort()]
            angles = angles[:,angles[2].argsort(kind="mergesort")[::-1]]
        elif bc[0] == [1, 0] and bc[1] == [0, 1]:
            angles = angles[:,angles[1].argsort()[::-1]]
            angles = angles[:,angles[2].argsort(kind="mergesort")[::-1]]
        elif bc[0] == [0, 1] and bc[1] == [1, 0]:
            angles = angles[:,angles[1].argsort()[::-1]]
            angles = angles[:,angles[2].argsort(kind="mergesort")]
        elif bc[0] == [1, 0] and bc[1] == [1, 0]:
            angles = angles[:,angles[1].argsort()]
            angles = angles[:,angles[2].argsort(kind="mergesort")]
    elif np.sum(bc) > 2:
        message = ("There must only be one reflected boundary "
                    "in each direction")
        warnings.warn(message)
    return angles

def _energy_grid(groups, grid):
    # Create energy grid
    if grid in [87, 361, 618]:
        energy_grid = np.load(DATA_PATH + "energy_bounds.npz")[str(grid)]
    else:
        energy_grid = np.arange(groups + 1)
    if groups == 361:
        label = str(self.groups).zfill(3)
        idx_edges = np.load(DATA_PATH + "G361_grid_index.npz")
        idx_edges = idx_edges[label]
    else:
        idx_edges = dimensions.index_generator(len(energy_grid)-1, groups)
    return energy_grid, idx_edges

def _medium_map(materials, edges_x, key=False):
    # materials is list of list with each element [idx, material, width]
    material_key = {}
    medium_map = np.ones((len(edges_x) - 1)) * -1
    for material in materials:
        material_key[material[0]] = material[1]
        for region in material[2].split(","):
            start, stop = region.split("-")
            idx1 = np.argmin(np.fabs(float(start) - edges_x))
            idx2 = np.argmin(np.fabs(float(stop) - edges_x))
            medium_map[idx1:idx2] = material[0]
    # Verify all cells are filled
    assert np.all(medium_map != -1)
    medium_map = medium_map.astype(np.int32)
    if key:
        return medium_map, material_key
    return medium_map

def _velocity(groups, edges_gg=None):
    """ Convert energy edges to speed at cell centers, Relative Physics
    Arguments:
        groups: Number of energy groups
        edges_gg: energy grid bounds
    Returns:
        speeds at cell centers (cm/s)   """
    if np.all(edges_gg == None):
        return np.ones((groups))
    centers_gg = 0.5 * (edges_gg[1:] + edges_gg[:-1])
    gamma = (EV_TO_JOULES * centers_gg) / (MASS_NEUTRON * LIGHT_SPEED**2) + 1
    velocity = LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    return velocity
