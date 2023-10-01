########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Writing python functions for block interpolation
# 
########################################################################

import numpy as np


########################################################################
# Manufactured Solutions and Accuracy
########################################################################
def _spatial_error(approx, reference, ndims=1):
    """ Calculating the spatial error between an approximation and the
    reference solution
    Arguments:
        approx (array double): approximate flux
        reference (array double): Reference flux
        ndims (int): Number of spatial dimensions for flux (1 or 2)
    Returns:
        L2 error normalized for the number of spatial cells
    """
    assert approx.shape == reference.shape, "Not the same array shape"
    if ndims == 1:
        normalized = (approx.shape[0])**(-0.5)
    elif ndims == 2:
        normalized = (approx.shape[0] * approx.shape[1])**(-0.5)
    return normalized * np.linalg.norm(approx - reference)


def _spatial_accuracy(error1, error2, ratio):
    """ Finding the order of accuracy between errors on different spatial
    grids, where error2 is the refined grid
    Arguments:
        error1 (double): Error between an approximate solution and the
            reference solution on the same grid
        error2 (double): Error between an approximate solution and the
            reference solution on a more refined grid
        ratio (double): Ratio between the spatial cell width of the error1
            grid and the error2 grid (delta x1 / delta x2)
    Returns:
        Order of accuracy
    """
    return np.log(error1 / error2) / np.log(ratio)


def _wynn_epsilon(lst, rank):
    """ Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        lst: list of values for convergence
        rank: rank of system
    Returns:
        2D Array where diagonal is convergence
    """
    N = 2 * rank + 1
    error = np.zeros((N + 1, N + 1))
    for ii in range(1, N + 1):
        error[ii, 1] = lst[ii - 1]
    for ii in range(3, N + 2):
        for jj in range(3, ii + 1):
            if (error[ii-1,jj-2] - error[ii-2,jj-2]) == 0.0:
                error[ii-1,jj-1] = error[ii-2,jj-3]
            else:
                error[ii-1,jj-1] = error[ii-2,jj-3] \
                            + 1 / (error[ii-1,jj-2] - error[ii-2,jj-2])
    return abs(error[-1,-1])


def _flux_coarsen_2d(fine_flux, fine_edges_x, fine_edges_y, coarse_edges_x, \
        coarse_edges_y, ratio):
    # Set coarse grid cells
    cells_x = coarse_edges_x.shape[0] - 1
    cells_y = coarse_edges_y.shape[0] - 1
    # Initialize coarse flux
    coarse_flux = np.zeros(((cells_x, cells_y,) + fine_flux.shape[2:]))
    # Iterate over x cells
    count_x = 0
    for ii in range(cells_x):
        # Keep track of x bounds
        idx_x = np.argwhere((fine_edges_x < coarse_edges_x[ii+1]) \
                        & (fine_edges_x >= coarse_edges_x[ii]))
        count_x += len(idx_x)
        count_y = 0
        # Iterate over y cells
        for jj in range(cells_y):
            # Keep track of y bounds
            idx_y = np.argwhere((fine_edges_y < coarse_edges_y[jj+1]) \
                            & (fine_edges_y >= coarse_edges_y[jj]))
            count_y += len(idx_y)
            coarse_flux[ii,jj] = np.sum(fine_flux[idx_x,idx_y], axis=(0,1)) * ratio
            # coarse_flux[ii,jj] = np.mean(fine_flux[idx_x,idx_y], axis=(0,1))
        # Make sure we got all columns
        assert count_y == fine_flux.shape[1], "Not including all y cells"
    # Make sure we got all rows
    assert count_x == fine_flux.shape[0], "Not including all x cells"
    return coarse_flux


########################################################################
# Two Dimensional Nearby Problems
########################################################################
def _material_index(medium_map):
    """ Finding index of one-dimensional medium map
    Arguments:
        medium_map (int [cells_x]): one-dimensional array of materials, 
            where each different number represents a new material
    Returns:
        array of indexes between material interfaces
    """
    # Get middle indices
    splits = list(np.where(medium_map[:-1] != medium_map[1:])[0] + 1)
    # Add endpoints
    splits = np.array([0] + splits + [len(medium_map)], dtype=np.int32)
    return splits

def _to_block(medium_map):
    """ Separating medium map into chunks for MNP for each chunk
    Arguments:
        medium_map (int [cells_x, cells_y]): two-dimensional array of 
            materials where each different number represents a new material
    Returns:
        x_splits (int []): array of index for x direction
        y_splits (int []): array of index for y direction
    """
    # X direction - for each y cell
    x_splits = []
    for jj in range(medium_map.shape[1]):
        x_splits.append(list(_material_index(medium_map[:,jj])))
    x_splits = np.unique([ii for lst in x_splits for ii in lst])
    # Y direction - for each x cell
    y_splits = []
    for ii in range(medium_map.shape[0]):
        y_splits.append(list(_material_index(medium_map[ii])))
    y_splits = np.unique([ii for lst in y_splits for ii in lst])
    # Return two arrays (might be different lengths)
    return x_splits, y_splits
