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

import logging

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import svd as scipy_svd

logger = logging.getLogger(__name__)


########################################################################
# Evaluating Fluxes
########################################################################
def reaction_rates(flux, xs_matrix, medium_map):
    if len(flux.shape) == 2:
        return _reaction_rate_1d(flux, xs_matrix, medium_map)
    elif len(flux.shape) == 3:
        return _reaction_rate_2d(flux, xs_matrix, medium_map)
    else:
        logger.warning("Unable to calculate reaction rate")


def _reaction_rate_1d(flux, xs_matrix, medium_map):
    # Flux parameters
    cells_x, groups = flux.shape
    # Initialize reaction rate data
    rate = np.zeros((cells_x, groups))
    # Iterate over spatial cells
    for ii, mat in enumerate(medium_map):
        rate[ii] = flux[ii] @ xs_matrix[mat].T
    # return reaction rate
    return rate


def _reaction_rate_2d(flux, xs_matrix, medium_map):
    # Flux parameters
    cells_x, cells_y, groups = flux.shape
    # Initialize reaction rate data
    rate = np.zeros((cells_x, cells_y, groups))
    # Iterate over spatial cells
    for ii in range(cells_x):
        for jj in range(cells_y):
            mat = medium_map[ii, jj]
            rate[ii, jj] = flux[ii, jj] @ xs_matrix[mat].T
    # return reaction rate
    return rate


def average_array(arr):
    return 0.5 * (arr[1:] + arr[:-1])


########################################################################
# Manufactured Solutions and Accuracy
########################################################################
def spatial_error(approx, reference, ndims=1):
    """Calculating the spatial error between an approximation and the
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
        normalized = (approx.shape[0]) ** (-0.5)
    elif ndims == 2:
        normalized = (approx.shape[0] * approx.shape[1]) ** (-0.5)
    return normalized * np.linalg.norm(approx - reference)


def order_accuracy(error1, error2, ratio):
    """Finding the order of accuracy between errors on different
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
    """Perform Wynn Epsilon Convergence Algorithm
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
            if (error[ii - 1, jj - 2] - error[ii - 2, jj - 2]) == 0.0:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3]
            else:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3] + 1 / (
                    error[ii - 1, jj - 2] - error[ii - 2, jj - 2]
                )
    return abs(error[-1, -1])


def flux_coarsen_1d(fine_flux, fine_x, coarse_x):
    # Set coarse grid cells
    cells_x = coarse_x.shape[0] - 1
    # Initialize coarse flux
    coarse_flux = np.zeros(((cells_x,) + fine_flux.shape[1:]))
    # Iterate over x cells
    count_x = 0
    for ii in range(cells_x):
        # Keep track of x bounds
        idx_x = np.argwhere((fine_x < coarse_x[ii + 1]) & (fine_x >= coarse_x[ii]))
        count_x += len(idx_x)
        coarse_flux[ii] = np.mean(fine_flux[idx_x], axis=0)
    # Make sure we got all rows
    assert count_x == fine_flux.shape[0], "Not including all x cells"
    return coarse_flux


def flux_propagate_1d(coarse_flux, mult):
    cells_x = coarse_flux.shape[0]
    # Initialize fine flux
    fine_flux = np.zeros(((mult * cells_x,) + coarse_flux.shape[2:]))
    for ii in range(cells_x):
        fine_flux[ii * mult : (ii + 1) * mult] = coarse_flux[ii]
    return fine_flux


def flux_coarsen_2d(fine_flux, fine_x, fine_y, coarse_x, coarse_y):
    # Set coarse grid cells
    cells_x = coarse_x.shape[0] - 1
    cells_y = coarse_y.shape[0] - 1
    # Initialize coarse flux
    coarse_flux = np.zeros(
        (
            (
                cells_x,
                cells_y,
            )
            + fine_flux.shape[2:]
        )
    )
    # Iterate over x cells
    count_x = 0
    for ii in range(cells_x):
        # Keep track of x bounds
        idx_x = np.argwhere((fine_x < coarse_x[ii + 1]) & (fine_x >= coarse_x[ii]))
        count_x += len(idx_x)
        count_y = 0
        # Iterate over y cells
        for jj in range(cells_y):
            # Keep track of y bounds
            idx_y = np.argwhere((fine_y < coarse_y[jj + 1]) & (fine_y >= coarse_y[jj]))
            count_y += len(idx_y)
            coarse_flux[ii, jj] = np.mean(fine_flux[idx_x, idx_y], axis=(0, 1))
        # Make sure we got all columns
        assert count_y == fine_flux.shape[1], "Not including all y cells"
    # Make sure we got all rows
    assert count_x == fine_flux.shape[0], "Not including all x cells"
    return coarse_flux


def flux_propagate_2d(coarse_flux, mult):
    cells_x, cells_y = coarse_flux.shape[:2]
    # Initialize fine flux
    fine_flux = np.zeros(
        (
            (
                mult * cells_x,
                mult * cells_y,
            )
            + coarse_flux.shape[2:]
        )
    )
    for ii in range(cells_x):
        for jj in range(cells_y):
            fine_flux[ii * mult : (ii + 1) * mult, jj * mult : (jj + 1) * mult] = (
                coarse_flux[ii, jj]
            )
    return fine_flux


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
    """Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G] or [G hat]): values of grid values
    Returns:
        resized (float [G hat] or [G]): values of resized grid
    """
    # Combine edges
    if value.shape[0] + 1 == coarse.shape[0]:
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
        magnitude = np.sum(value[idx1:idx2] * np.diff(fine[idx1 : idx2 + 1]))
        magnitude /= gg2 - gg1
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
                    new_value[gg1, gg2] = value[cc1, cc2]
    return new_edges, new_value


def resize_array_2d(fine, coarse, value):
    """Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G x G] or [G hat x G hat]): values of grid values
    Returns:
        resized (float [G hat x G hat] or [G x G]): values of resized grid
    """
    # Combine edges
    if value.shape[0] + 1 == coarse.shape[0]:
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
            magnitude = np.sum(
                value[idx1:idx2, idx3:idx4]
                * (
                    np.diff(fine[idx1 : idx2 + 1])[:, None]
                    @ np.diff(fine[idx3 : idx4 + 1])[None, :]
                )
            )
            magnitude /= (gg2 - gg1) * (gg4 - gg3)
            # Populate coarsened array
            shrink[cc1, cc2] = magnitude
    return shrink


########################################################################
# Nearby Problems
########################################################################
def _material_index(medium_map):
    """Finding index of one-dimensional medium map
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
    """Separating medium map into chunks for MNP for each chunk
    Arguments:
        medium_map (int [cells_x, cells_y]): two-dimensional array of
            materials where each different number represents a new material
    Returns:
        x_splits (int []): array of index for x direction
        y_splits (int []): array of index for y direction
    """
    # Return one array for one-dimensional
    if len(medium_map.shape) == 1:
        return _material_index(medium_map)
    # X direction - for each y cell
    x_splits = []
    for jj in range(medium_map.shape[1]):
        x_splits.append(list(_material_index(medium_map[:, jj])))
    x_splits = np.unique([ii for lst in x_splits for ii in lst])
    # Y direction - for each x cell
    y_splits = []
    for ii in range(medium_map.shape[0]):
        y_splits.append(list(_material_index(medium_map[ii])))
    y_splits = np.unique([ii for lst in y_splits for ii in lst])
    # Return two arrays (might be different lengths)
    return x_splits, y_splits


def inscribed_circle(edges_x, edges_y, radii):
    """Separating medium map into inscribed chunks for MNP for each chunk
    Arguments:
        edges_x (float [cells_x + 1]): locations of spatial cell edges (x)
        edges_y (float [cells_y + 1]): locations of spatial cell edges (y)
        radii (float [Number of circles]): list of inscribed circle radii
    Returns:
        x_splits (int []): array of index for x direction
        y_splits (int []): array of index for y direction
    """
    x_splits = []
    y_splits = []
    center = np.max(radii)

    for radius in radii:
        # Vertices
        left = center - radius / np.sqrt(2)
        right = center + radius / np.sqrt(2)

        # X direction
        xx = list(np.where((edges_x >= left) & (edges_x < right))[0][[0, -1]])
        xx = np.array([0] + xx + [len(edges_x) - 1], dtype=np.int32)
        x_splits.append(list(xx))

        # Y direction
        yy = list(np.where((edges_y >= left) & (edges_y < right))[0][[0, -1]])
        yy = np.array([0] + yy + [len(edges_y) - 1], dtype=np.int32)
        y_splits.append(list(yy))

    # Collect unique values
    x_splits = np.unique([ii for lst in x_splits for ii in lst])
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

    # Call SVD; s_inv is a 1-D vector of reciprocal singular values
    U, s_inv, V = _svd_dmd(y_minus, K)

    # Cache U.T @ y_plus once (shape r x K-1); reused for Atilde and RHS
    UtYp = U.T @ y_plus

    # Atilde = U.T @ y_plus @ V.T @ diag(s_inv)
    # V.T * s_inv broadcasts diag(s_inv) without a full matrix multiply
    Atilde = UtYp @ (V.T * s_inv)

    # Solve (I - Atilde) delta_y = U.T @ y_plus[:, -1]
    delta_y = np.linalg.solve(np.eye(Atilde.shape[0]) - Atilde, UtYp[:, -1])

    # Estimate new flux
    flux = flux_old.ravel() - y_plus[:, K - 2] + U @ delta_y
    return flux.reshape(cells_x, groups)


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

    # Call SVD; s_inv is a 1-D vector of reciprocal singular values
    U, s_inv, V = _svd_dmd(y_minus, K)

    # Cache U.T @ y_plus once (shape r x K-1); reused for Atilde and RHS
    UtYp = U.T @ y_plus

    # Atilde = U.T @ y_plus @ V.T @ diag(s_inv)
    # V.T * s_inv broadcasts diag(s_inv) without a full matrix multiply
    Atilde = UtYp @ (V.T * s_inv)

    # Solve (I - Atilde) delta_y = U.T @ y_plus[:, -1]
    delta_y = np.linalg.solve(np.eye(Atilde.shape[0]) - Atilde, UtYp[:, -1])

    # Estimate new flux
    flux = flux_old.ravel() - y_plus[:, K - 2] + U @ delta_y
    return flux.reshape(cells_x, cells_y, groups)


# Row count above which the randomized range-finder replaces exact SVD.
# Speedup vs exact SVD grows when the energy threshold truncates the rank
# to r << K-1 (few dominant DMD modes), because then l = r + _RAND_OVERSAMPLING
# << m = K-1 and both the sketch and back-projection operate on smaller matrices.
_N_RAND_THRESHOLD = 1000
_RAND_OVERSAMPLING = 10  # extra sketch columns; trade accuracy for speed by lowering
_RAND_N_ITER = 2  # power iterations; 0=fastest, 2=balanced, 4+=most accurate
_rng = np.random.default_rng(0)


def _rand_svd(A, k):
    """Randomized SVD: k leading singular triplets of tall matrix A (n, m).

    Implements the randomized range-finder with power iterations from
    Halko, Martinsson & Tropp (2011).

    Complexity vs exact SVD (both O(n * m^2)):
      - Equal cost when k == m (full rank target, sketch fills all columns).
      - Speedup ≈ m / (k + oversampling) when k << m, i.e. when the energy
        threshold truncates most of the K-1 snapshot columns.
    """
    n, m = A.shape
    ll = min(k + _RAND_OVERSAMPLING, m)  # sketch size, capped at m

    # Random projection: sketch A's column space down to l dimensions
    Omega = _rng.standard_normal((m, ll))
    Y = A @ Omega  # (n, l)

    # Power iterations sharpen the range approximation at cost of two extra
    # passes over A each; stabilized by intermediate QR orthonormalization
    for _ in range(_RAND_N_ITER):
        Q, _ = scipy_qr(Y, mode="economic", check_finite=False)
        Z, _ = scipy_qr(A.T @ Q, mode="economic", check_finite=False)
        Y = A @ Z  # (n, l)

    Q, _ = scipy_qr(Y, mode="economic", check_finite=False)  # (n, l)

    # Project A into the l-dimensional sketch, then exact SVD on (l, m)
    B = Q.T @ A  # (l, m)
    Uh, s, V = scipy_svd(B, full_matrices=False, check_finite=False)

    return (Q @ Uh)[:, :k], s[:k], V[:k, :]


def _svd_dmd(A, K):
    residual = 1e-09
    n, m = A.shape

    if n < _N_RAND_THRESHOLD:
        # Exact economy SVD for smaller matrices
        U, s, V = scipy_svd(A, full_matrices=False, check_finite=False)
    else:
        # Randomized SVD for large spatial grids (n = cells * groups >> K-1).
        # Targets all m = K-1 columns; the gain materialises after the energy
        # truncation below reduces the working rank to r << K-1.
        U, s, V = _rand_svd(A, m)

    # Truncate to components capturing > (1 - residual) of total energy
    s_sum = s.sum()
    if s_sum > 0:
        tail = 1.0 - np.cumsum(s) / s_sum
        r = int(np.count_nonzero(tail > residual))
        if r == 0:
            r = int(np.count_nonzero(s > 0))
    else:
        r = int(np.count_nonzero(s > 0))
    r = max(1, min(r, K))

    # Return truncated bases and reciprocal singular values as a 1-D vector
    return U[:, :r].copy(), 1.0 / s[:r], V[:r, :].copy()
    return U[:, :r].copy(), 1.0 / s[:r], V[:r, :].copy()
