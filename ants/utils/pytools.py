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
def spatial_error(approx, reference, ndims=1):
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


def order_accuracy(error1, error2, ratio):
    """ Finding the order of accuracy between errors on different 
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


def wynn_epsilon(lst, rank):
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
# Meshing Different Energy Grids
########################################################################

def _concatenate_edges_1d(fine, coarse, value):
    # Combine the edges for both the coarse and fine grid
    new_edges = np.sort(np.unique(np.concatenate((coarse, fine))))
    # Create new array for values
    new_value = np.zeros((new_edges.shape[0] - 1))
    # Iterate over fine edges
    for cc, (gg1, gg2) in enumerate(zip(fine[:-1], fine[1:])):
        idx1 = np.argmin(np.fabs(gg1 - new_edges))
        idx2 = np.argmin(np.fabs(gg2 - new_edges))
        for gg in range(idx1, idx2):
            new_value[gg] = value[cc]
    return new_edges, new_value


def resize_array_1d(fine, coarse, value):
    """ Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G] or [G hat]): values of grid values
    Returns:
        resized (float [G hat] or [G]): values of resized grid
    """
    # Combine edges
    if (value.shape[0] + 1 == coarse.shape[0]):
        fine, coarse = coarse.copy(), fine.copy()
    fine, value = _concatenate_edges_1d(fine, coarse, value)
    # Create coarse array
    shrink = np.zeros((coarse.shape[0] - 1))
    # Iterate over all coarse bins
    for cc, (gg1, gg2) in enumerate(zip(coarse[:-1], coarse[1:])):
        # Find indices for edge locations
        idx1 = np.argmin(np.fabs(gg1 - fine))
        idx2 = np.argmin(np.fabs(gg2 - fine))
        # Estimate magnitude
        magnitude = np.sum(value[idx1:idx2] * np.diff(fine[idx1:idx2+1]))
        magnitude /= (gg2 - gg1)
        # Populate coarsened array
        shrink[cc] = magnitude
    return shrink


def _concatenate_edges_2d(fine, coarse, value):
    # Combine the edges for both the coarse and fine grid
    new_edges = np.sort(np.unique(np.concatenate((coarse, fine))))
    # Create new array for values
    new_value = np.zeros((new_edges.shape[0] - 1, new_edges.shape[0] - 1))
    # Iterate over fine edges
    for cc1, (gg1, gg2) in enumerate(zip(fine[:-1], fine[1:])):
        idx1 = np.argmin(np.fabs(gg1 - new_edges))
        idx2 = np.argmin(np.fabs(gg2 - new_edges))
        for cc2, (gg3, gg4) in enumerate(zip(fine[:-1], fine[1:])):
            idx3 = np.argmin(np.fabs(gg3 - new_edges))
            idx4 = np.argmin(np.fabs(gg4 - new_edges))
            for gg1 in range(idx1, idx2):
                for gg2 in range(idx3, idx4):
                    new_value[gg1,gg2] = value[cc1,cc2]
    return new_edges, new_value


def resize_array_2d(fine, coarse, value):
    """ Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G x G] or [G hat x G hat]): values of grid values
    Returns:
        resized (float [G hat x G hat] or [G x G]): values of resized grid
    """
    # Combine edges
    if (value.shape[0] + 1 == coarse.shape[0]):
        fine, coarse = coarse.copy(), fine.copy()
    fine, value = _concatenate_edges_2d(fine, coarse, value)
    # return value
    # Create coarse array
    shrink = np.zeros((coarse.shape[0] - 1, coarse.shape[0] - 1))
    # Iterate over all coarse bins
    for cc1, (gg1, gg2) in enumerate(zip(coarse[:-1], coarse[1:])):
        # Find indices for edge locations
        idx1 = np.argmin(np.fabs(gg1 - fine))
        idx2 = np.argmin(np.fabs(gg2 - fine))
        for cc2, (gg3, gg4) in enumerate(zip(coarse[:-1], coarse[1:])):
            # Find indices for edge locations
            idx3 = np.argmin(np.fabs(gg3 - fine))
            idx4 = np.argmin(np.fabs(gg4 - fine))
            # Estimate magnitude
            magnitude = np.sum(value[idx1:idx2,idx3:idx4] \
                                * (np.diff(fine[idx1:idx2+1])[:,None] \
                                @ np.diff(fine[idx3:idx4+1])[None,:]))
            magnitude /= ((gg2 - gg1) * (gg4 - gg3))
            # Populate coarsened array
            shrink[cc1, cc2] = magnitude
    return shrink


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


########################################################################
# One Dimensional DMD
########################################################################

def dmd_1d(flux_old, y_minus, y_plus, K):
    # Convert from memoryview
    flux_old = np.asarray(flux_old)
    y_minus = np.asarray(y_minus)
    y_plus = np.asarray(y_plus)

    # Collect dimensions
    cells_x, groups = flux_old.shape

    # Flatten y_minus, y_plus
    y_minus = y_minus.reshape(cells_x * groups, K - 1)
    y_plus = y_plus.reshape(cells_x * groups, K - 1)

    # Call SVD
    U, S, V = _svd_dmd(y_minus, K)

    # Calculate Atilde
    Atilde = U.T @ y_plus @ V.T @ S.T
    
    # Calculate delta_y
    I = np.identity(Atilde.shape[0])
    delta_y = np.linalg.solve(I - Atilde, (U.T @ y_plus[:,-1]).T)
    
    # Estimate new flux
    flux = (flux_old.flatten() - y_plus[:,K-2]) + (U @ delta_y).T
    flux = flux.reshape(cells_x, groups)

    return flux


def dmd_2d(flux_old, y_minus, y_plus, K):
    # Convert from memoryview
    flux_old = np.asarray(flux_old)
    y_minus = np.asarray(y_minus)
    y_plus = np.asarray(y_plus)

    # Collect dimensions
    cells_x, cells_y, groups = flux_old.shape

    # Flatten y_minus, y_plus
    y_minus = y_minus.reshape(cells_x * cells_y * groups, K - 1)
    y_plus = y_plus.reshape(cells_x * cells_y * groups, K - 1)

    # Call SVD
    U, S, V = _svd_dmd(y_minus, K)

    # Calculate Atilde
    Atilde = U.T @ y_plus @ V.T @ S.T
    
    # Calculate delta_y
    I = np.identity(Atilde.shape[0])
    delta_y = np.linalg.solve(I - Atilde, (U.T @ y_plus[:,-1]).T)
    
    # Estimate new flux
    flux = (flux_old.flatten() - y_plus[:,K-2]) + (U @ delta_y).T
    flux = flux.reshape(cells_x, cells_y, groups)

    return flux


def _svd_dmd(A, K):
    residual = 1e-09

    # Compute SVD
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # Find the non-zero singular values
    if (S[(1-np.cumsum(S)/np.sum(S)) > residual].size >= 1):
        spos = S[(1 - np.cumsum(S) / np.sum(S)) > residual].copy()
    else:
        spos = S[S > 0].copy()
    
    # Create diagonal matrix
    mat_size = np.min([K, len(spos)])
    S = np.zeros((mat_size, mat_size))
    
    # Select the u and v that correspond with the nonzero singular values
    U = U[:, :mat_size].copy()
    V = V[:mat_size, :].copy()
    
    # S will be the inverse of the singular value diagonal matrix 
    S[np.diag_indices(mat_size)] = 1 / spos

    return U, S, V