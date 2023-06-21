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

import numpy as np
import pkg_resources

from ants.constants import *
from ants.utils.hybrid import energy_coarse_index

DATA_PATH = pkg_resources.resource_filename("ants","sources/energy/")


def angular_x(info):
    angle_x, angle_w = np.polynomial.legendre.leggauss(info["angles"])
    angle_w /= np.sum(angle_w)
    # Ordering for reflective boundaries
    if np.sum(info["bc_x"]) > 0.0:
        if info["bc_x"] == [1, 0]:
            idx = angle_x.argsort()
        elif info["bc_x"] == [0, 1]:
            idx = angle_x.argsort()[::-1]
        angle_x = angle_x[idx].copy()
        angle_w = angle_w[idx].copy()
    return angle_x, angle_w


def angular_xy(info):
    angles = info["angles"]
    bc_x = info["bc_x"]
    bc_y = info["bc_y"]
    # eta, xi, mu: direction cosines (x,y,z) 
    xx, wx = np.polynomial.legendre.leggauss(angles)
    yy, wy = np.polynomial.chebyshev.chebgauss(angles)
    # Create arrays for each angle
    angle_x = np.zeros(2 * angles**2)
    angle_y = np.zeros(2 * angles**2)
    angle_z = np.zeros(2 * angles**2)
    angle_w = np.zeros(2 * angles**2)
    # Indexing
    idx = 0
    for ii in range(angles):
        for jj in range(angles):
            angle_z[idx:idx+2] = xx[ii]
            angle_x[idx] = np.sqrt(1 - xx[ii]**2) * np.cos(np.arccos(yy[jj]))
            angle_x[idx+1] = np.sqrt(1 - xx[ii]**2) * np.cos(-np.arccos(yy[jj]))
            angle_y[idx] = np.sqrt(1 - xx[ii]**2) * np.sin(np.arccos(yy[jj]))
            angle_y[idx+1] = np.sqrt(1 - xx[ii]**2) * np.sin(-np.arccos(yy[jj]))
            angle_w[idx:idx+2] = wx[ii] * wy[jj]
            idx += 2
    # Take only positive angle_z values
    angle_x = angle_x[angle_z > 0].copy()
    angle_y = angle_y[angle_z > 0].copy()
    angle_w = angle_w[angle_z > 0] / np.sum(angle_w[angle_z > 0])
    # Order for reflected surfaces and return
    return _ordering_angles_xy(angle_x, angle_y, angle_w, bc_x, bc_y)


def _ordering_angles_xy(angle_x, angle_y, angle_w, bc_x, bc_y):
    # Get number of discrete ordinates
    angles = int(np.sqrt(angle_x.shape[0]))
    # Get only positive angles
    matrix = np.fabs(np.round(np.vstack((angle_x, angle_y, angle_w)), 12))
    # Get unique combinations and convert to size N**2
    matrix = np.repeat(np.unique(matrix, axis=1), 4, axis=1)
    # signs for [angle_x, angle_y, angle_w]
    directions = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 1]])
    # Only one reflected surface
    if (bc_x == [0, 0] or bc_x == [0, 1]) and bc_y == [0, 0]:
        idx = [0, 1, 2, 3]
    elif bc_x == [0, 0] and bc_y == [1, 0]:
        idx = [2, 0, 3, 1]
    elif bc_x == [0, 0] and bc_y == [0, 1]:
        idx = [0, 2, 1, 3]
    elif bc_x == [1, 0] and bc_y == [0, 0]:
        idx = [1, 0, 3, 2]
    directions = np.tile(directions[:,idx], int(angles**2 / 4))
    return matrix * directions


# def _energy_grid(groups, grid):
def energy_grid(groups, grid):
    """
    Calculate energy grid bounds (MeV) and index for coarsening
    Arguments:
        groups (int): Number of energy groups for problem
        grid (int): specified energy grid to use (87, 361, 618)
    Returns:
        edges_g (float [grid + 1]): MeV energy group bounds
        edges_gidx (int [groups + 1]): Location of grid index for problem
    """
    # Create energy grid
    if grid in [87, 361, 618]:
        edges_g = np.load(DATA_PATH + "energy_bounds.npz")[str(grid)]
    else:
        edges_g = np.arange(groups + 1, dtype=float)
    # Calculate the indicies for the specific grid
    if groups == 361:
        label = str(self.groups).zfill(3)
        edges_gidx = np.load(DATA_PATH + "G361_grid_index.npz")
        edges_gidx = edges_gidx[label]
    else:
        edges_gidx = energy_coarse_index(len(edges_g)-1, groups)
    return edges_g, edges_gidx


# def _medium_map(materials, edges_x, key=False):
def spatial_map(materials, edges_x, key=False):
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


# def _velocity(groups, edges_g=None):
def energy_velocity(groups, edges_g=None):
    """ Convert energy edges to speed at cell centers, Relative Physics
    Arguments:
        groups: Number of energy groups
        edges_g: energy grid bounds
    Returns:
        speeds at cell centers (cm/s)   """
    if np.all(edges_g == None):
        return np.ones((groups,))
    centers_gg = 0.5 * (edges_g[1:] + edges_g[:-1])
    gamma = (EV_TO_JOULES * centers_gg) / (MASS_NEUTRON * LIGHT_SPEED**2) + 1
    velocity = LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    return velocity
