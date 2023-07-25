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


def spatial1d(materials, edges_x, key=False):
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


def cylinder2d(radii, xs_total, xs_scatter, xs_fission, delta_x, delta_y, \
        bc_x, bc_y, weight_map=None):
    """ Convert cartesian squares into an approximate cylindrical shape
    """
    # Calculate outer radius 
    radius = max(radii)[1]
    # Set the center of the medium
    center = (radius, radius)
    # Calculate the percentage of each material in each cell
    if weight_map is None:
        weight_map = _mc_weight_matrix(delta_x, delta_y, center, radii)
    # Convert the weights into distinct materials
    cells_x = delta_x.shape[0]
    cells_y = delta_y.shape[0]
    data = _weight_to_material(weight_map, xs_total, xs_scatter, \
                               xs_fission, cells_x, cells_y)
    quarter, xs_total, xs_scatter, xs_fission = data
    # Calculate the correct size of the cylinder (originally in +,+ quadrant)
    medium_map = _cylinder_medium_map(quarter, bc_x, bc_y)
    return medium_map, xs_total, xs_scatter, xs_fission, weight_map


def _mc_weight_matrix(delta_x, delta_y, center, radii, samples=100000):
    # Calculate the fraction of each material inside and outside radii
    weight_map = []
    # Global spatial grid edges
    np.random.seed(42)
    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    # Set local grid points
    grid_x = edges_x[(edges_x >= center[0]) & (edges_x <= center[0] \
                        + max(radii)[1])] - center[0]
    grid_y = edges_y[(edges_y >= center[1]) & (edges_y <= center[1] \
                        + max(radii)[1])] - center[1]
    # Calculate all samples
    samples_x = np.random.uniform(0, max(radii)[1], samples)
    samples_y = np.random.uniform(0, max(radii)[1], samples)
    # Iterate over spatial grid
    for yy in range(len(grid_y) - 1):
        for xx in range(len(grid_x) - 1):
            weight_map.append(_weight_grid_cell(samples_x, samples_y, radii, \
                        grid_x[xx], grid_x[xx+1], grid_y[yy], grid_y[yy+1]))
            # print(grid_x[xx], grid_x[xx+1])
    return weight_map / np.sum(weight_map, axis=1)[:,None]


def _weight_grid_cell(x, y, radii, x1, x2, y1, y2):
    ppr = []
    for iir, oor in radii:
        # Collect particles in circle
        idx = np.argwhere(((x**2 + y**2) > iir**2) & ((x**2 + y**2) <= oor**2))
        temp_x = x[idx].copy()
        ppr.append(len(temp_x[np.where((temp_x >= x1) & (temp_x < x2) \
                                    & (y[idx] >= y1) & (y[idx] < y2))]))
    # Collect particles outside circle
    temp_x = x[(x**2 + y**2) > max(radii)[1]**2]
    temp_y = y[(x**2 + y**2) > max(radii)[1]**2]
    ppr.append(len(temp_x[np.where((temp_x >= x1) & (temp_x < x2) \
                                    & (temp_y >= y1) & (temp_y < y2))]))
    return ppr


def _weight_to_material(weight_map, xs_total, xs_scatter, xs_fission, \
        cells_x, cells_y):
    # Convert cross sections to weight map
    cy_xs_total = []
    cy_xs_scatter = []
    cy_xs_fission = []
    if weight_map.shape[0] != (cells_x * cells_y):
        cells_x = int(0.5 * cells_x)
        cells_y = int(0.5 * cells_y)
    medium_map = np.zeros((cells_x * cells_y), dtype=np.int32)
    for mat, weight in enumerate(np.unique(weight_map, axis=0)):
        cy_xs_total.append(np.sum(xs_total * weight[:,None], axis=0))
        cy_xs_scatter.append(np.sum(xs_scatter * weight[:,None,None], axis=0))
        cy_xs_fission.append(np.sum(xs_fission * weight[:,None,None], axis=0))
        medium_map[np.where(np.all(weight_map == weight, axis=1))] = mat
    medium_map = medium_map.reshape(cells_x, cells_y)
    cy_xs_total = np.array(cy_xs_total)
    cy_xs_scatter = np.array(cy_xs_scatter)
    cy_xs_fission = np.array(cy_xs_fission)
    return medium_map, cy_xs_total, cy_xs_scatter, cy_xs_fission


def _cylinder_medium_map(quad1, bc_x, bc_y):
    # quadrants=[1,2,3,4]):
    """ Gets correct shape of the medium map to account for all quadrants
    (Default quarant is I)
          +
      II  |  I   
    ------------- +
      III |  IV
    """
    quad2 = np.flip(quad1, axis=1).copy()
    quad3 = np.flip(quad1, axis=(1,0)).copy()
    quad4 = np.flip(quad1, axis=0).copy()
    if bc_y == [0, 0]:
        # Full circle
        if bc_x == [0, 0]:
            medium_map = np.block([[quad3, quad4], [quad2, quad1]])
        elif bc_x == [0, 1]:
            medium_map = np.block([[quad3], [quad2]])
        elif bc_x == [1, 0]:
            medium_map = np.block([[quad4], [quad1]])
    elif bc_y == [1, 0]:
        if bc_x == [0, 0]:
            medium_map = np.block([quad2, quad1])
        elif bc_x == [0, 1]:
            medium_map = np.block([quad2])
        elif bc_x == [1, 0]:
            medium_map = np.block([quad1])
    elif bc_y == [0, 1]:
        if bc_x == [0, 0]:
            medium_map = np.block([quad3, quad4])
        elif bc_x == [0, 1]:
            medium_map = np.block([quad3])
        elif bc_x == [1, 0]:
            medium_map = np.block([quad4])
    return medium_map


def location2d(matrix, value, coordinates, edges_x, edges_y):
    """ Populating areas of 2D matrices easily (medium_map, sources)
    Arguments:
        matrix: (I x J x ...): At least a 2D array, where the value 
        value: (int) or array depending on additional dimensions of matrix
        coordinates: list of [(starting index), x_length, y_length] or
                list of triangle coordinates [(x1, y1), (x2, y2), (x3, y3)]
        edges_x: (array of size I + 1)
        edges_y: (array of size J + 1)
    Returns:
        matrix populated with value
    """
    # This is for triangular grid cells (not ready yet)
    if isinstance(coordinates[0][1], tuple):
        pass
    # For rectangular grids
    for index, span_x, span_y in coordinates:
        # Get starting locations
        idx_x1 = np.argwhere(edges_x == index[0])[0, 0]
        idx_y1 = np.argwhere(edges_y == index[1])[0, 0]
        # Get widths of rectangular cells
        idx_x2 = np.argwhere(edges_x == index[0] + span_x)[0, 0]
        idx_y2 = np.argwhere(edges_y == index[1] + span_y)[0, 0]
        # Populate with value
        matrix[idx_x1:idx_x2, idx_y1:idx_y2] = value
    return matrix