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
    if isinstance(info, int):
        angles = info
        bc_x = [0, 0]
    else:
        angles = info["angles"]
        bc_x = info["bc_x"]
    angle_x, angle_w = np.polynomial.legendre.leggauss(angles)
    angle_w /= np.sum(angle_w)
    # Ordering for reflective boundaries
    if np.sum(bc_x) > 0.0:
        if bc_x == [1, 0]:
            idx = angle_x.argsort()
        elif bc_x == [0, 1]:
            idx = angle_x.argsort()[::-1]
        angle_x = angle_x[idx].copy()
        angle_w = angle_w[idx].copy()
    return angle_x, angle_w


def angular_xy(info):
    if isinstance(info, int):
        angles = info
        bc_x = [0, 0]
        bc_y = [0, 0]
    else:
        angles = info["angles"]
        bc_x = info["bc_x"]
        bc_y = info["bc_y"]
    # Get angles and weights from product quadrature
    angle_x, angle_y, angle_z, angle_w = _product_quadrature(angles)
    # Take only positive angle_z values
    angle_x = angle_x[angle_z > 0].copy()
    angle_y = angle_y[angle_z > 0].copy()
    angle_w = angle_w[angle_z > 0] / np.sum(angle_w[angle_z > 0])
    # Order the angles for boundary conditions and return angle_x, angle_y, angle_w
    return _ordering_angles_xy(angle_x, angle_y, angle_w, bc_x, bc_y)
    # return angle_x, angle_y, angle_w


def _product_quadrature(angles):
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
    # Round for reflecting angles
    angle_x = np.round(angle_x, 12)
    angle_y = np.round(angle_y, 12)
    angle_z = np.round(angle_z, 12)
    angle_w = np.round(angle_w, 12)
    # Return all angles
    return angle_x, angle_y, angle_z, angle_w


def _ordering_angles_xy(angle_x, angle_y, angle_w, bc_x, bc_y):
    # Get number of discrete ordinates
    angles = int(np.sqrt(angle_x.shape[0]))
    # Get only positive angles
    matrix = np.fabs(np.vstack((angle_x, angle_y, angle_w)))
    # Get unique combinations and convert to size N**2
    matrix = np.repeat(np.unique(matrix, axis=1), 4, axis=1)
    
    # signs for [angle_x, angle_y, angle_w]
    directions = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 1]])

    if bc_x == [0, 0]:
        
        if bc_y == [0, 0]: 
            idx = [0, 1, 2, 3]
        
        elif bc_y == [1, 0]:
            # idx = [2, 0, 3, 1]
            idx = [3, 1, 2, 0]

        elif bc_y == [0, 1]:
            # idx = [0, 2, 1, 3]
            idx = [0, 1, 2, 3]

    elif bc_x == [1, 0]:

        if bc_y == [0, 0]:
            # idx = [1, 0, 3, 2]
            idx = [1, 3, 2, 0]

        elif bc_y == [1, 0]:
            # idx = [3, 2, 1, 0]
            idx = [3, 1, 2, 0]

        elif bc_y == [0, 1]:
            # idx = [1, 0, 3, 2]
            idx = [1, 3, 0, 2]

    elif bc_x == [0, 1]:

        if bc_y == [0, 0]:
            # idx = [0, 1, 2, 3]
            idx = [0, 2, 1, 3]

        elif bc_y == [1, 0]:
            # idx = [2, 3, 0, 1]
            idx = [2, 0, 3, 1]

        elif bc_y == [0, 1]:
            # idx = [0, 1, 2, 3]
            idx = [0, 2, 1, 3]

    directions = np.tile(directions[:,idx], int(angles**2 / 4))
    return matrix * directions


def energy_grid(grid, groups_fine, groups_coarse=None, optimize=True):
    """
    Calculate energy grid bounds (MeV) and index for coarsening
    Arguments:
        grid (int): specified energy grid to use (87, 361, 618)
        groups_fine (int): Number of fine energy groups for problem
        groups_coarse (int): Number of coarse energy groups for problem
        optimize (bool): Whether to use predetermined group edges (Default True)
    Returns:
        edges_g (float [grid + 1]): MeV energy group bounds
        edges_gidx_fine (int [groups + 1]): Location of fine grid index for problem
        edges_gidx_coarse (int [groups + 1]): Location of coarse grid index for problem
    """
    # Create energy grid
    if grid in [87, 361, 618]:
        edges_g = np.load(DATA_PATH + "energy_grids.npz")[str(grid)]
        
        # Collect grid boundary indices
        fgrid = str(grid).zfill(3)
        edges_data = np.load(DATA_PATH + f"G{fgrid}_grid_index.npz")

    else:
        edges_g = np.arange(groups_fine + 1, dtype=float)
        edges_data = None
    
    # Calculate the indices for the specific fine grid
    edges_gidx_fine = _group_indexing(grid, len(edges_g)-1, groups_fine, \
                                edges_data=edges_data, optimize=optimize)

    if groups_coarse is None:
        return edges_g, edges_gidx_fine
    
    # Calculate the indices for the specific coarse grid
    edges_gidx_coarse = _group_indexing(grid, groups_fine, groups_coarse, \
                                edges_data=edges_data, optimize=optimize)

    # Check for both reduced (hybrid splitting)
    if (groups_fine == groups_coarse) and (groups_fine not in [87, 361, 618]):
        edges_gidx_coarse = np.arange(groups_coarse + 1, dtype=np.int32)
    
    return edges_g, edges_gidx_fine, edges_gidx_coarse


def _group_indexing(grid, groups_fine, groups_coarse, edges_data=None, optimize=True):
    # Calculate the indices for the specific grid
    if (grid in [87, 361, 618]) and (optimize):
        # Predefined coarse grid index
        try:
            label = str(groups_coarse).zfill(3)
            edges_gidx = edges_data[label].copy()
        
        except KeyError:
            edges_gidx = energy_coarse_index(groups_fine, groups_coarse)

    else:
        edges_gidx = energy_coarse_index(groups_fine, groups_coarse)
    
    # Convert to correct type
    edges_gidx = edges_gidx.astype(np.int32)

    return edges_gidx


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


def gamma_time_steps(edges_t, gamma=0.5, half_step=True):
    """ Add gamma half time steps to original time steps with initial step
    where gamma = 0.5 or 2 - sqrt(2). For external source with TR-BDF2 problems.

    Arguments:
        edges_t: array of length (steps + 1)
        half_step: if True, first half step is 0.5, else gamma
    Returns:
        array of length (steps * 2 + 1)
    """

    if gamma == 0.5:
        half_steps = 0.5 * (edges_t[1:] + edges_t[:-1])
    else:
        half_steps = edges_t[:-1] + np.diff(edges_t) * gamma
    # Combine half steps
    combined_steps = np.sort(np.concatenate((edges_t, half_steps)))
    if half_step:
        combined_steps[1] = 0.5 * (combined_steps[0] + combined_steps[2])
    return combined_steps


def spatial1d(layers, edges_x):
    """ Creating one-dimensional medium map
    
    :param layers: list of lists where each layer is a new material. A 
        layer is comprised of an index (int), material name (str), and 
        the width (str) in the form [index, material, width]. The width 
        is the starting and ending points of the material (in cm) 
        separated by a dash. If there are multiple regions, a comma can 
        separate them. I.E. layer = [0, "plutonium", "0 - 2, 3 - 4"].
    :param edges_x: Array of length I + 1 with the location of the cell edges
    :return: One-dimensional array of length I, identifying the locations 
        of the materials
    """
    # Initialize medium_map
    medium_map = np.ones((len(edges_x) - 1), dtype=np.int32) * -1
    # Iterate over all layers
    for layer in layers:
        for region in layer[2].split(","):
            start, stop = region.split("-")
            idx1 = np.argmin(np.fabs(float(start) - edges_x))
            idx2 = np.argmin(np.fabs(float(stop) - edges_x))
            medium_map[idx1:idx2] = layer[0]
    # Verify all cells are filled
    assert np.all(medium_map != -1)
    return medium_map


def spatial2d(medium_map, value, coordinates, edges_x, edges_y):
    """ Populating 2D medium_map with rectangular geometries

    :param medium_map: 2D medium_map array of size (I x J) to input value, 
        must be of type np.int32
    :param value: Integer to be populating the medium_map. It will correspond
        to the ordering of the cross section materials
    :type value: int
    :param coordinates: list of locations, where each location is composed
        of the starting index tuple, the length in the x direction and 
        the length in the y direction, with all values being in centimeters.
        I.E. location = [(x1, y1), dx, dy].
    :param edges_x: Array of length I + 1 with the location of the cell 
        edges in the x direction
    :param edges_y: Array of length J + 1 with the location of the cell 
        edges in the y direction
    :return: medium_map populated with value at specific coordinates
    """
    # Iterate over coordinates
    for [(x1, y1), span_x, span_y] in coordinates:
        # Get starting locations
        idx_x1 = np.argwhere(edges_x == np.round(x1, 12))[0, 0]
        idx_y1 = np.argwhere(edges_y == np.round(y1, 12))[0, 0]
        # Get widths of rectangular cells
        idx_x2 = np.argwhere(edges_x == np.round(x1 + span_x, 12))[0, 0]
        idx_y2 = np.argwhere(edges_y == np.round(y1 + span_y, 12))[0, 0]
        # Populate with value
        medium_map[idx_x1:idx_x2, idx_y1:idx_y2] = value
    return medium_map


def weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission):
    """ Convert weight matrices to medium maps and create appropriate 
    cross sections. materials are the original materials used while 
    materials* are the reweighted values.

    Arguments:
        weight_matrix (float [cells_x, cells_y, materials]): weight matrix
            needed for estimating the percentage of each material in each
            spatial cell
        xs_total (float [materials, groups]): list of total cross sections, 
            must be ordered correctly
        xs_scatter (float [materials, groups, groups]): list of scatter 
            cross sections, must be ordered correctly
        xs_fission (float [materials, groups, groups]): list of fission 
            cross sections, must be ordered correctly
    Returns:
        medium_map (int [cells_x, cells_y]): identifier of material layout
        new_xs_total (float [materials*, groups]): list of total cross 
            sections, where the materials* index corresponds to a specific 
            location on the medium_map
        new_xs_scatter (float [materials*, groups, groups]): list of 
            scatter cross sections, where the materials* index corresponds 
            to a specific location on the medium_map
        new_xs_fission (float [materials*, groups, groups]): list of 
            fission cross sections, where the materials* index corresponds 
            to a specific location on the medium_map
    """
    # Convert cross sections to weight map
    new_xs_total = []
    new_xs_scatter = []
    new_xs_fission = []
    cells_x, cells_y = weight_matrix.shape[:2]
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    # Get the unique number of material weights    
    weights = np.unique(weight_matrix.reshape(-1, weight_matrix.shape[2]), axis=0)
    # Iterate over all weights
    for mat, weight in enumerate(weights):
        # print(weight.shape)
        new_xs_total.append(np.sum(xs_total * weight[:,None], axis=0))
        new_xs_scatter.append(np.sum(xs_scatter * weight[:,None,None], axis=0))
        new_xs_fission.append(np.sum(xs_fission * weight[:,None,None], axis=0))
        # Identify location on map
        medium_map[np.where(np.all(weight_matrix == weight, axis=2))] = mat
    # Return adjusted cross sections and arrays
    return medium_map, np.array(new_xs_total), np.array(new_xs_scatter), \
        np.array(new_xs_fission)


def weight_matrix2d(edges_x, edges_y, materials, N_particles=100_000, **kwargs):
    """ Creating weight matrix for a triangle in cartesian coordinates
    Arguments:
        edges_x (float [cells_x + 1]): Spatial cell edge values in x direction
        edges_y (float [cells_y + 1]): Spatial cell edge values in y direction
        N (int): Optional, number of MC samples, default = 100_000
    Keyword Arguments
        rectangles (list [(x1, y1), dx, dy]): list of all rectangles 
            staring vertices and widths (in cm), put None if shape not needed
        triangles (list [(x1, y1), (x2, y2), (x3, y3)]): list of all triangle
            vertices (in cm), put None if shape not needed
        circles (list [(x1, y1), (r1, r2)]): list of all circle centers and
            inside and outside radii
    Returns:
        weight_matrix (float [cells_x, cells_y]): Normalized weight
            matrix for percent inside material shapes
    """
    # Assert at least one shape
    assert ("triangles" in kwargs.keys()) or ("rectangles" in kwargs.keys()) \
        or ("circles" in kwargs.keys()), "Need at least one shape"
    # Create uniform samples
    samples = np.random.uniform(size=(N_particles, 2), low=[0, 0], \
                                high=[np.max(edges_x), np.max(edges_y)])
    # Create weight matrix (ii x jj x materials)
    weight_matrix = np.zeros((edges_x.shape[0] - 1, edges_y.shape[0] - 1, \
                              materials))
    tally = np.zeros((materials))
    # Iterate over samples
    for x, y in zip(samples[:,0], samples[:,1]):
        idx_x = np.digitize(x, edges_x) - 1
        idx_y = np.digitize(y, edges_y) - 1
        tally *= 0.0
        # Check different shapes
        if "triangles" in kwargs.keys() and np.sum(tally) == 0.0:
            _inside_triangle(kwargs["triangles"], x, y, tally, \
                             kwargs["triangle_index"])
        if "rectangles" in kwargs.keys() and np.sum(tally) == 0.0:
            _inside_rectangle(kwargs["rectangles"], x, y, tally, \
                              kwargs["rectangle_index"])
        if "circles" in kwargs.keys() and np.sum(tally) == 0.0:
            _inside_circle(kwargs["circles"], x, y, tally, \
                           kwargs["circle_index"])
        tally[tally > 0.0] = 1.0
        if np.sum(tally) == 0:
            tally[-1] = 1
        weight_matrix[idx_x, idx_y] += tally
    # Make sure symmetric circles
    if "circles" in kwargs.keys():
        weight_matrix = _quarter_symmetry(weight_matrix, kwargs["circles"], \
                                          edges_x, edges_y)
    # Normalize and return percentage inside
    return weight_matrix / np.sum(weight_matrix, axis=2)[...,None]


def _inside_triangle(triangles, x, y, tally, index):
    """ Using Finite Element theory to see if point is inside triangle
    Based off of http://www.alternatievewiskunde.nl/sunall/suna57.htm
    Arguments:
        triangles (list [(x1, y1), (x2, y2), (x3, y3)]): list of all triangle
            vertices (in cm), put None if shape not needed
        x, y (float): x, y coordinates to test
        tally (int [length materials]): List of which circle to tally
    Returns:
        Updated tally array
    """
    # Iterate through triangles
    for ii, triangle in enumerate(triangles):
        # Unpack vertices
        (x1, y1), (x2, y2), (x3, y3) = triangle
        # Perform transformation
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        xi = ((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1)) / determinant
        eta = ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / determinant
        # tally[ii] = np.min([xi, eta, 1 - xi - eta])
        in_out = np.min([xi, eta, 1 - xi - eta])
        in_out = 1.0 if in_out >= 0.0 else 0.0
        tally[index[ii]] = in_out
        if in_out == 1.0:
            break


def _inside_rectangle(rectangles, x, y, tally, index):
    """ Checking if point is inside rectangle
    Arguments:
        rectangles (list [(x1, y1), dx, dy]): list of all rectangles 
            staring vertices and widths (in cm), put None if shape not needed
        x, y (float): x, y coordinates to test
        tally (int [length materials]): List of which circle to tally
    Returns:
        Updated tally array
    """
    # Iterate through rectangles
    for ii, [(x1, y1), dx, dy] in enumerate(rectangles):
        tally[index[ii]] = (x1 <= x <= (x1 + dx)) and (y1 <= y < (y1 + dy))
        if (x1 <= x <= (x1 + dx)) and (y1 <= y < (y1 + dy)):
            break


def _inside_circle(circles, x, y, tally, index):
    """ Checking if point is inside circle
    Arguments:
        circles (list [(x1, y1), (r1, r2)]): list of all circle centers and
            inside and outside radii
        x, y (float): x, y coordinates to test
        tally (int [length materials]): List of which circle to tally
    Returns:
        Updated tally array
    """
    # Create checker for each rectangle
    for ii, [(x1, y1), (r1, r2)] in enumerate(circles):
        radius = np.sqrt((x - x1)**2 + (y - y1)**2)
        tally[index[ii]] = (r1 <= radius) and (radius <= r2)
        if (r1 <= radius) and (radius <= r2):
            break


def _quarter_symmetry(weight_matrix, circles, edges_x, edges_y):
    # Ensure symmetrical
    symmetrical_weight_matrix = np.zeros(weight_matrix.shape)
    # Get center (x1, y1) and radius (r2) of last circle
    [(x1, y1), (r1, r2)] = circles[-1]

    # (+x, +y)
    idx_x1 = np.argwhere((edges_x >= x1) & (edges_x < x1 + r2)).flatten()
    idx_y1 = np.argwhere((edges_y >= y1) & (edges_y < y1 + r2)).flatten()
    quarter = 0.25 * weight_matrix[idx_x1][:,idx_y1].copy()
    idx_x1, idx_y1 = np.meshgrid(idx_x1, idx_y1, indexing="ij")

    # (-x, +y)
    idx_x2 = np.argwhere((edges_x < x1) & (edges_x >= x1 - r2)).flatten()
    idx_y2 = np.argwhere((edges_y >= y1) & (edges_y < y1 + r2)).flatten()

    quarter += 0.25 * weight_matrix[idx_x2][:,idx_y2][::-1].copy()
    idx_x2, idx_y2 = np.meshgrid(idx_x2, idx_y2, indexing="ij")

    # (-x, -y)
    idx_x3 = np.argwhere((edges_x < x1) & (edges_x >= x1 - r2)).flatten()
    idx_y3 = np.argwhere((edges_y < y1) & (edges_y >= y1 - r2)).flatten()
    
    quarter += 0.25 * weight_matrix[idx_x3][:,idx_y3][::-1,::-1].copy()
    idx_x3, idx_y3 = np.meshgrid(idx_x3, idx_y3, indexing="ij")

    # (+x, -y)
    idx_x4 = np.argwhere((edges_x >= x1) & (edges_x < x1 + r2)).flatten()
    idx_y4 = np.argwhere((edges_y < y1) & (edges_y >= y1 - r2)).flatten()
    
    quarter += 0.25 * weight_matrix[idx_x4][:,idx_y4][:,::-1].copy()
    idx_x4, idx_y4 = np.meshgrid(idx_x4, idx_y4, indexing="ij")
    
    # Populate new matrix
    symmetrical_weight_matrix[idx_x1, idx_y1] = quarter.copy()
    symmetrical_weight_matrix[idx_x2, idx_y2] = quarter[::-1].copy()
    symmetrical_weight_matrix[idx_x3, idx_y3] = quarter[::-1,::-1].copy()
    symmetrical_weight_matrix[idx_x4, idx_y4] = quarter[:,::-1].copy()

    return symmetrical_weight_matrix
