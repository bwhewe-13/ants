################################################################################
#                            ___    _   _____________
#                           /   |  / | / /_  __/ ___/
#                          / /| | /  |/ / / /  \__ \
#                         / ___ |/ /|  / / /  ___/ /
#                        /_/  |_/_/ |_/ /_/  /____/
#
# Angular quadrature for the ants package. Provides discrete-ordinates
# direction cosines and weights for 1D, 2D, and 3D problems.
#
# Two quadrature families are available for 2D/3D (selected with the
# ``method`` keyword):
#   * "product" (default): Gauss-Legendre (polar) x Gauss-Chebyshev
#     (azimuthal) product set.
#   * "ldfe": Linear Discontinuous Finite Element sets (LDFE-SA variant)
#     from Jarrell and Adams, M&C 2011. Tabulated for refinement levels
#     1-3, corresponding to angles = 4, 8, 16.
#
################################################################################

import itertools
from importlib.resources import files

import numpy as np
from scipy.special import erf

from ants.datatypes import QuadratureData

# Refinement levels available in the tabulated LDFE-SA data, keyed by the
# per-dimension ``angles`` count they correspond to (2D: 4 * 4**n ordinates
# is a perfect square 2**(n+1); 3D: 8 * 4**n resolves identically).
_LDFE_ANGLE_TO_LEVEL = {4: 1, 8: 2, 16: 3}

# Lazily loaded LDFE-SA first-octant tables (columns: mu, eta, xi, weight).
_LDFE_SA_CACHE = None


def angular_x(angles, bc_x=[0, 0], datatype=True):
    """Compute 1D Gauss-Legendre quadrature angles and weights.

    Parameters
    ----------
    angles : int
        Number of angles (int) or problem info dict with keys ``"angles"``
        and ``"bc_x"`` (list of two ints, 1 = reflected boundary).

    Returns
    -------
    angle_x : ndarray, shape (angles,)
        Quadrature direction cosines, ordered for boundary conditions.
    angle_w : ndarray, shape (angles,)
        Normalized quadrature weights summing to 1.
    """

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
    if datatype:
        return QuadratureData(angle_x=angle_x, angle_w=angle_w)
    return angle_x, angle_w


# Called from cython
def _angular_x(angles, bc_x):
    return angular_x(angles, bc_x, False)


def angular_xy(angles, bc_x=[0, 0], bc_y=[0, 0], datatype=True, method="product"):
    """Compute 2D quadrature angles and weights for slab geometry.

    Builds a quadrature set, keeps only the directions with positive polar
    cosine (upper hemisphere), and reorders them to satisfy reflective
    boundary conditions.

    Parameters
    ----------
    angles : int
        Number of angles per dimension. For ``method="ldfe"`` only
        ``angles`` in {4, 8, 16} are available (LDFE-SA levels 1-3).
    bc_x : list of two ints, optional
        Boundary conditions in the x-direction (default is [0, 0]).
    bc_y : list of two ints, optional
        Boundary conditions in the y-direction (default is [0, 0]).
    datatype : bool, optional
        If True, return a QuadratureData object. If False, return raw arrays.
    method : str, optional
        Quadrature family: "product" (Legendre x Chebyshev, default) or
        "ldfe" (LDFE-SA from Jarrell and Adams, M&C 2011).

    Returns
    -------
    angle_x : ndarray, shape (angles**2,)
        x-direction cosines, ordered for boundary conditions.
    angle_y : ndarray, shape (angles**2,)
        y-direction cosines, ordered for boundary conditions.
    angle_w : ndarray, shape (angles**2,)
        Normalized quadrature weights summing to 1.

    Notes
    -----
    The ``method`` keyword only affects this Python entry point. The cython
    hybrid solvers regenerate coarse quadratures internally via the default
    product set; LDFE selection applies to user-built QuadratureData.
    """
    # Get angles and weights from the requested quadrature family
    if method == "ldfe":
        angle_x, angle_y, angle_z, angle_w = _ldfe_quadrature(angles)
    else:
        angle_x, angle_y, angle_z, angle_w = _product_quadrature(angles)
    # Take only positive angle_z values
    angle_x = angle_x[angle_z > 0].copy()
    angle_y = angle_y[angle_z > 0].copy()
    angle_w = angle_w[angle_z > 0] / np.sum(angle_w[angle_z > 0])
    # Order the angles for boundary conditions and return angle_x, angle_y, angle_w
    angle_x, angle_y, angle_w = _ordering_angles_xy(
        angle_x, angle_y, angle_w, bc_x, bc_y
    )
    if datatype:
        return QuadratureData(angle_x=angle_x, angle_y=angle_y, angle_w=angle_w)
    return angle_x, angle_y, angle_w


# Called from cython
def _angular_xy(angles, bc_x, bc_y):
    return angular_xy(angles, bc_x, bc_y, False)


def artificial_scatter_matrix(angle_x, angle_y, angle_w, sigma_as, beta):
    """Compute the normalized artificial scattering matrix M_as[N, N].

    Implements the forward-peaked scattering kernel from Frank et al. (2020),
    "Ray Effect Mitigation for the Discrete Ordinates Method Using Artificial
    Scattering", Nuclear Science and Engineering.

    The kernel is s_eps(mu) = (2 / (sqrt(pi) * eps * Erf(2/eps)))
                                * exp(-(1 - mu)**2 / eps**2)
    with eps = beta / N_q, where N_q is the number of ordinates.

    Parameters
    ----------
    angle_x : ndarray, shape (N,)
        x-direction cosines for 2D.
    angle_y : ndarray, shape (N,)
        y-direction cosines for 2D.
    angle_w : ndarray, shape (N,)
        Normalized quadrature weights (sum to 1).
    sigma_as : float
        Artificial scattering strength parameter. Set to 0 to disable.
    beta : float
        Kernel width parameter. Typical values: 4.5 (explicit), 4.0 (implicit).

    Returns
    -------
    M_as : ndarray, shape (N, N)
        Normalized artificial scattering matrix.
    """
    N = len(angle_x)
    eps = beta / N if N > 0 else 1.0

    # Compute pairwise dot products
    dots = np.outer(angle_x, angle_x) + np.outer(angle_y, angle_y)  # (N, N)

    # Kernel Eq. (6) from Frank et al.
    if eps > 1e-15:
        erf_val = erf(2.0 / eps)
    else:
        erf_val = 1.0
    prefactor = 2.0 / (np.sqrt(np.pi) * eps * erf_val)
    S = prefactor * np.exp(-((1.0 - dots) ** 2) / eps**2)  # (N, N)

    # Weight by quadrature weights: M_as[n,m]
    WS = S * angle_w[np.newaxis, :]  # (N, N)

    # Per-ordinate normalization factor c_n
    c_n = WS.sum(axis=1, keepdims=True)  # (N, 1)
    c_n = np.where(c_n > 0, c_n, 1.0)  # avoid division by zero

    # Final normalized matrix
    M_as = sigma_as * WS / c_n  # (N, N)
    return M_as.astype(np.float64)


def _product_quadrature(angles):
    """Build a Legendre * Chebyshev product quadrature set over the full sphere.

    Uses Gauss-Legendre points for the polar cosine (z-direction, mu) and
    Chebyshev points for the azimuthal angle (mapped to x/y direction cosines
    eta/xi). Each (mu, phi) pair gives +/-phi, producing 2*angles**2 directions
    covering the full sphere before filtering to the upper hemisphere.

    Parameters
    ----------
    angles : int
        Number of quadrature points per dimension.

    Returns
    -------
    angle_x, angle_y, angle_z : ndarray, shape (2*angles**2,)
        Direction cosines for x, y, z axes over the full sphere.
    angle_w : ndarray, shape (2*angles**2,)
        Quadrature weights (un-normalized product of Legendre * Chebyshev weights).
    """
    # Polar cosine (mu) via Gauss-Legendre; azimuthal (phi) via Chebyshev
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
            angle_z[idx : idx + 2] = xx[ii]
            angle_x[idx] = np.sqrt(1 - xx[ii] ** 2) * np.cos(np.arccos(yy[jj]))
            angle_x[idx + 1] = np.sqrt(1 - xx[ii] ** 2) * np.cos(-np.arccos(yy[jj]))
            angle_y[idx] = np.sqrt(1 - xx[ii] ** 2) * np.sin(np.arccos(yy[jj]))
            angle_y[idx + 1] = np.sqrt(1 - xx[ii] ** 2) * np.sin(-np.arccos(yy[jj]))
            angle_w[idx : idx + 2] = wx[ii] * wy[jj]
            idx += 2
    # Round for reflecting angles
    angle_x = np.round(angle_x, 12)
    angle_y = np.round(angle_y, 12)
    angle_z = np.round(angle_z, 12)
    angle_w = np.round(angle_w, 12)
    # Return all angles
    return angle_x, angle_y, angle_z, angle_w


def _ldfe_first_octant(level):
    """Load the tabulated LDFE-SA first-octant directions for a refinement level.

    Returns an array of shape (M, 4) with columns [mu, eta, xi, weight], all
    strictly positive (first octant), with weights summing to pi/2.
    """
    global _LDFE_SA_CACHE
    if _LDFE_SA_CACHE is None:
        path = files("ants").joinpath("sources/quadrature/ldfe_sa.npz")
        with np.load(path) as data:
            _LDFE_SA_CACHE = {key: data[key].copy() for key in data.files}
    return _LDFE_SA_CACHE[str(level)]


def _ldfe_quadrature(angles):
    """Build an LDFE-SA quadrature set over the full sphere.

    Drop-in analogue of ``_product_quadrature``: replicates the tabulated
    first-octant directions to all eight sign-octants. Because every
    first-octant component is strictly positive, each row produces eight
    distinct directions with no duplicates.

    Parameters
    ----------
    angles : int
        Per-dimension angle count; must be in {4, 8, 16} (LDFE-SA levels 1-3).

    Returns
    -------
    angle_x, angle_y, angle_z, angle_w : ndarray, shape (8*M,)
        Direction cosines and (un-normalized) weights over the full sphere,
        where M is the number of first-octant directions for the level.
    """
    if angles not in _LDFE_ANGLE_TO_LEVEL:
        raise ValueError(
            "LDFE quadrature supports angles in {4, 8, 16} "
            "(LDFE-SA refinement levels 1-3); got " + repr(angles)
        )
    octant = _ldfe_first_octant(_LDFE_ANGLE_TO_LEVEL[angles])  # (M, 4)
    # All eight sign combinations of (mu, eta, xi)
    signs = np.array(list(itertools.product([1, -1], repeat=3)))  # (8, 3)
    xyz = (octant[:, None, :3] * signs[None, :, :]).reshape(-1, 3)  # (8*M, 3)
    angle_w = np.repeat(octant[:, 3], signs.shape[0])  # (8*M,)
    # Round for reflecting angles (matches _product_quadrature convention)
    angle_x = np.round(xyz[:, 0], 12)
    angle_y = np.round(xyz[:, 1], 12)
    angle_z = np.round(xyz[:, 2], 12)
    angle_w = np.round(angle_w, 12)
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

    directions = np.tile(directions[:, idx], int(angles**2 / 4))
    return matrix * directions


def _ordering_angles_xyz(angle_x, angle_y, angle_z, angle_w, bc_x, bc_y, bc_z):
    # Get unique magnitude combinations, repeat for all 8 octants
    matrix = np.fabs(np.vstack((angle_x, angle_y, angle_z, angle_w)))
    unique_matrix = np.unique(matrix, axis=1)
    n_unique = unique_matrix.shape[1]
    matrix = np.repeat(unique_matrix, 8, axis=1)

    # Columns represent 8 octants: (+x,+y,+z), (-x,+y,+z), (+x,-y,+z), (-x,-y,+z),
    #                               (+x,+y,-z), (-x,+y,-z), (+x,-y,-z), (-x,-y,-z)
    directions = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    # For bc=[1,0] (reflect at origin), the negative direction must sweep first (f=1).
    # For bc=[0,1] or bc=[0,0], positive direction sweeps first (f=0).
    fx = 1 if bc_x == [1, 0] else 0
    fy = 1 if bc_y == [1, 0] else 0
    fz = 1 if bc_z == [1, 0] else 0

    # col_signs[axis, col]: 0 = positive, 1 = negative for that axis in that column
    col_signs = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],  # x
            [0, 0, 1, 1, 0, 0, 1, 1],  # y
            [0, 0, 0, 0, 1, 1, 1, 1],  # z
        ]
    )

    # XOR with f: when f=1, the negative column (sign=1) gets key=0 and sorts first.
    # Primary sort key = x, secondary = y, tertiary = z.
    sort_keys = (col_signs ^ np.array([[fx], [fy], [fz]])).T  # (8, 3)
    idx = sorted(range(8), key=lambda k: tuple(sort_keys[k]))

    directions = np.tile(directions[:, idx], n_unique)
    return matrix * directions


def angular_xyz(
    angles, bc_x=[0, 0], bc_y=[0, 0], bc_z=[0, 0], datatype=True, method="product"
):
    """Compute 3D quadrature angles and weights.

    Builds a quadrature set over the full sphere and reorders directions to
    satisfy reflective boundary conditions.

    Parameters
    ----------
    angles : int
        Number of angles per dimension. For ``method="ldfe"`` only
        ``angles`` in {4, 8, 16} are available (LDFE-SA levels 1-3).
    bc_x : list of two ints, optional
        Boundary conditions in the x-direction (default is [0, 0]).
    bc_y : list of two ints, optional
        Boundary conditions in the y-direction (default is [0, 0]).
    bc_z : list of two ints, optional
        Boundary conditions in the z-direction (default is [0, 0]).
    datatype : bool, optional
        If True, return a QuadratureData object. If False, return raw arrays.
    method : str, optional
        Quadrature family: "product" (Legendre x Chebyshev, default) or
        "ldfe" (LDFE-SA from Jarrell and Adams, M&C 2011).

    Returns
    -------
    angle_x : ndarray, shape (2*angles**2,)
        x-direction cosines, ordered for boundary conditions.
    angle_y : ndarray, shape (2*angles**2,)
        y-direction cosines, ordered for boundary conditions.
    angle_z : ndarray, shape (2*angles**2,)
        z-direction cosines, ordered for boundary conditions.
    angle_w : ndarray, shape (2*angles**2,)
        Normalized quadrature weights summing to 1.
    """
    if method == "ldfe":
        angle_x, angle_y, angle_z, angle_w = _ldfe_quadrature(angles)
    else:
        angle_x, angle_y, angle_z, angle_w = _product_quadrature(angles)
    angle_w = angle_w / np.sum(angle_w)
    angle_x, angle_y, angle_z, angle_w = _ordering_angles_xyz(
        angle_x, angle_y, angle_z, angle_w, bc_x, bc_y, bc_z
    )
    if datatype:
        return QuadratureData(
            angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, angle_w=angle_w
        )
    return angle_x, angle_y, angle_z, angle_w


# Called from cython
def _angular_xyz(angles, bc_x, bc_y, bc_z):
    return angular_xyz(angles, bc_x, bc_y, bc_z, False)
