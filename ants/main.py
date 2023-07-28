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

np.random.seed(42)

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


def spatial2d(matrix, value, coordinates, edges_x, edges_y):
    """ Populating areas of 2D matrices easily (medium_map, sources) with
    rectangular geometries
    Arguments:
        matrix: (I x J x ...): At least a 2D array, where the value 
        value: (int) or array depending on additional dimensions of matrix
        coordinates: list of [(starting index), x_length, y_length]
        edges_x: (array of size I + 1)
        edges_y: (array of size J + 1)
    Returns:
        matrix populated with value
    """
    # For rectangular grids
    for [(x1, y1), span_x, span_y] in coordinates:
        # Get starting locations
        idx_x1 = np.argwhere(edges_x == x1)[0, 0]
        idx_y1 = np.argwhere(edges_y == y1)[0, 0]
        # Get widths of rectangular cells
        idx_x2 = np.argwhere(edges_x == x1 + span_x)[0, 0]
        idx_y2 = np.argwhere(edges_y == y1 + span_y)[0, 0]
        # Populate with value
        matrix[idx_x1:idx_x2, idx_y1:idx_y2] = value
    return matrix


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


def weight_matrix2d(rectangles, triangles, circles, edges_x, edges_y, N=100_000):
    """ Creating weight matrix for a triangle in cartesian coordinates
    Arguments:
        rectangles (list [(x1, y1), dx, dy]): list of all rectangles 
            staring vertices and widths (in cm), put None if shape not needed
        triangles (list [(x1, y1), (x2, y2), (x3, y3)]): list of all triangle
            vertices (in cm), put None if shape not needed
        circles (list [(x1, y1), (r1, r2)]): list of all circle centers and
            inside and outside radii
        edges_x (float [cells_x + 1]): Spatial cell edge values in x direction
        edges_y (float [cells_y + 1]): Spatial cell edge values in y direction
        N (int): Optional, number of MC samples, default = 100_000
    Returns:
        weight_matrix (float [cells_x, cells_y]): Normalized weight
            matrix for percent inside material shapes
    """
    # Create uniform samples
    samples = np.random.uniform(size=(N, 2), low=[0, 0], \
                                high=[np.max(edges_x), np.max(edges_y)])
    # Create weight matrix (ii x jj x outside, inside)
    weight_matrix = np.zeros((edges_x.shape[0] - 1, edges_y.shape[0] - 1, 2))
    # Iterate over samples
    for x, y in zip(samples[:,0], samples[:,1]):
        idx_x = np.digitize(x, edges_x) - 1
        idx_y = np.digitize(y, edges_y) - 1
        # Check different shapes
        where1 = _inside_triangle(x, y, triangles)
        where2 = _inside_rectangle(x, y, rectangles)
        where3 = _inside_circle(x, y, circles)
        weight_matrix[idx_x, idx_y, where1 + where2 + where3] += 1

    # Normalize and return percentage inside
    return weight_matrix[:,:,1] / np.sum(weight_matrix, axis=2)


def _inside_triangle(x, y, triangles):
    """ Using Finite Element theory to see if point is inside triangle
    Based off of http://www.alternatievewiskunde.nl/sunall/suna57.htm
    Arguments:
        x, y (float): x, y coordinates to test
        triangles (list [(x1, y1), (x2, y2), (x3, y3)]): list of all triangle
            vertices (in cm), put None if shape not needed
    Returns:
        True/False integer if point is inside triangle
    """
    # Check if exist
    if triangles is None:
        return 0
    # Create checker for each triangle
    all_triangles = np.zeros((len(triangles)))
    # Iterate through triangles
    for ii, triangle in enumerate(triangles):
        # Unpack vertices
        (x1, y1), (x2, y2), (x3, y3) = triangle
        # Perform transformation
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        xi = ((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1)) / determinant
        eta = ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / determinant
        all_triangles[ii] = np.min([xi, eta, 1 - xi - eta])
    return int(np.any(all_triangles >= 0))


def _inside_rectangle(x, y, rectangles):
    """ Checking if point is inside rectangle
    Arguments:
        x, y (float): x, y coordinates to test
        rectangles (list [(x1, y1), dx, dy]): list of all rectangles 
            staring vertices and widths (in cm), put None if shape not needed
    Returns:
        True/False integer if point is inside rectangle
    """
    # Check if exist
    if rectangles is None:
        return 0
    # Create checker for each rectangle
    all_rectangles = np.zeros((len(rectangles)))
    # Iterate through rectangles
    for ii, [(x1, y1), dx, dy] in enumerate(rectangles):
        all_rectangles[ii] = (x1 <= x <= (x1 + dx)) and (y1 <= y < (y1 + dy))
    return int(np.any(all_rectangles))


def _inside_circle(x, y, circles):
    """ Checking if point is inside circle
    Arguments:
        x, y (float): x, y coordinates to test
        circles (list [(x1, y1), (r1, r2)]): list of all circle centers and
            inside and outside radii
    Returns:
        True/False integer if point is inside circle
    """
    # Check if exist
    if circles is None:
        return 0
    # Create checker for each rectangle
    all_circles = np.zeros((len(circles)))
    for ii, [(x1, y1), (r1, r2)] in enumerate(circles):
        radius = np.sqrt((x - x1)**2 + (y - y1)**2)
        all_circles[ii] = (r1 <= radius <= r2)
    return int(np.any(all_circles))


########################################################################
# To Remove later
########################################################################

def _cylinder_symmetric(weight_matrix):
    half_x = int(0.5 * weight_matrix.shape[0])
    half_y = int(0.5 * weight_matrix.shape[1])
    # Divide into quarters
    quarters = np.stack((weight_matrix[:half_x, :half_y].copy(), 
                         weight_matrix[half_x:, :half_y][::-1].copy(), 
                         weight_matrix[:half_x, half_y:][:,::-1].copy(),
                         weight_matrix[half_x:, half_y:][::-1, ::-1].copy()))
    quarters = np.mean(quarters, axis=0)
    # Repopulate matrix
    weight_matrix[:half_x, :half_y] = quarters.copy()
    weight_matrix[half_x:, :half_y][::-1] = quarters.copy()
    weight_matrix[:half_x, half_y:][:,::-1] = quarters.copy()
    weight_matrix[half_x:, half_y:][::-1,::-1] = quarters.copy()
    assert np.round(np.sum(weight_matrix, axis=2), 10).all() == 1.
    return weight_matrix


def weight_cylinder2d(coordinates, edges_x, edges_y, N=100_000):
    """ Creating weight matrix for circles in cartesian coordinates
    Arguments:
        coordinates (list [tuple, list]): tuple is vertices (x, y) of circle 
            center and the list is comprised of all the radii
        edges_x (float [cells_x + 1]): Spatial cell edge values in x direction
        edges_y (float [cells_y + 1]): Spatial cell edge values in y direction
        N (int): Optional, number of MC samples, default = 100_000
    Returns:
        weight_matrix (float [cells)x, cells_y, len(radii) + 1]): Normalized 
            weight matrix for percent inside and outside each circle
    """
    # Unpack coordinates
    center, radii = coordinates
    # Create uniform samples
    samples = np.random.uniform(size=(N, 2), low=[0, 0], \
                                high=[np.max(edges_x), np.max(edges_y)])
    # Create weight matrix (ii x jj x inside/outside bins)
    weight_matrix = np.zeros((edges_x.shape[0] - 1, edges_y.shape[0] - 1, len(radii) + 1))
    # Iterate over samples
    for x, y in zip(samples[:,0], samples[:,1]):
        idx_x = np.digitize(x, edges_x) - 1
        idx_y = np.digitize(y, edges_y) - 1
        where = np.digitize(np.sqrt((x - center[0])**2 + (y - center[1])**2), radii)
        weight_matrix[idx_x, idx_y, where] += 1
    # Normalize
    weight_matrix /= np.sum(weight_matrix, axis=2)[:,:,None]
    # Make symmetric circle
    weight_matrix = _cylinder_symmetric(weight_matrix)
    return weight_matrix


def _triangle_transform(x, y, v1, v2, v3):
    """ Using Finite Element theory to see if point is inside triangle
    Based off of http://www.alternatievewiskunde.nl/sunall/suna57.htm
    Arguments:
        x, y (float): x, y coordinates to test
        v1, v2, v3 (tuple [float, float]): vertices (x, y) of triangle 
    Returns:
        Minimum of transformation, where min >= 0 is inside the triangle
    """
    # Unpack vertices
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    # Perform transformation
    determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    xi = ((y3 - y1) * (x - x1) - (x3 - x1) * (y - y1)) / determinant
    eta = ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / determinant
    # Return minimum
    return np.min([xi, eta, 1 - xi - eta])


def weight_triangle2d(v1, v2, v3, edges_x, edges_y, N=100_000):
    """ Creating weight matrix for a triangle in cartesian coordinates
    Arguments:
        v1, v2, v3 (tuple [float, float]): vertices (x, y) of triangle
        edges_x (float [cells_x + 1]): Spatial cell edge values in x direction
        edges_y (float [cells_y + 1]): Spatial cell edge values in y direction
        N (int): Optional, number of MC samples, default = 100_000
    Returns:
        weight_matrix (float [cells)x, cells_y, 2]): Normalized weight
            matrix for percent inside and outside the triangle
    """
    # Create uniform samples
    samples = np.random.uniform(size=(N, 2), low=[0, 0], \
                                high=[np.max(edges_x), np.max(edges_y)])
    # Create weight matrix (ii x jj x inside/outside)
    weight_matrix = np.zeros((edges_x.shape[0] - 1, edges_y.shape[0] - 1, 2))
    # Iterate over samples
    for x, y in zip(samples[:,0], samples[:,1]):
        idx_x = np.digitize(x, edges_x) - 1
        idx_y = np.digitize(y, edges_y) - 1
        where = 0 if _triangle_transform(x, y, v1, v2, v3) >= 0 else 1
        weight_matrix[idx_x, idx_y, where] += 1
    # Normalize
    weight_matrix /= np.sum(weight_matrix, axis=2)[:,:,None]
    return weight_matrix