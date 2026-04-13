########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One-Dimensional Interpolation Utilities
#
# Provides derivative approximations and Hermite spline interpolation
# used by the Method of Nearby Problems.
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np

cimport numpy as np

from ants.utils import pytools

########################################################################
# Basis Matrices
########################################################################

CUBIC_BASIS = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]],
    dtype=np.float64,
)

QUINTIC_BASIS = np.array(
    [
        [1,   0,  0,   0,    0,    0],
        [0,   0,  1,   0,    0,    0],
        [0,   0,  0,   0,  0.5,    0],
        [-10, 10, -6,  -4, -1.5,  0.5],
        [15, -15,  8,   7,  1.5,   -1],
        [-6,   6, -3,  -3, -0.5,  0.5],
    ],
    dtype=np.float64,
)


########################################################################
# Derivative Approximations
########################################################################

cpdef np.ndarray first_derivative(double[:] psi, double[:] x):
    """Estimate the first derivative dpsi/dx at discrete knots.

    Uses second-order finite-difference stencils at interior points
    and second-order endpoint formulas on non-uniform grids.

    Parameters
    ----------
    psi : double[:]
        Function values at knot positions.
    x : double[:]
        Knot coordinates (same length as psi).

    Returns
    -------
    numpy.ndarray of shape (N,)
    """
    cdef int N = psi.shape[0]
    cdef int ii
    assert N == x.shape[0], "Need to be same length"
    assert N > 2, "Need to be at least three knots"
    out = np.zeros(N)
    cdef double[:] dpsi = out
    dpsi[0] = (
        (psi[0] - psi[1]) / (x[0] - x[1])
        + (psi[0] - psi[2]) / (x[0] - x[2])
        + (-psi[1] + psi[2]) / (x[1] - x[2])
    )
    dpsi[N - 1] = (
        (psi[N-1] - psi[N-2]) / (x[N-1] - x[N-2])
        + (psi[N-1] - psi[N-3]) / (x[N-1] - x[N-3])
        + (-psi[N-2] + psi[N-3]) / (x[N-2] - x[N-3])
    )
    for ii in range(1, N - 1):
        dpsi[ii] = (
            (psi[ii] - psi[ii-1]) / (x[ii] - x[ii-1])
            + (psi[ii] - psi[ii+1]) / (x[ii] - x[ii+1])
            + (-psi[ii-1] + psi[ii+1]) / (x[ii-1] - x[ii+1])
        )
    return out


cpdef np.ndarray second_derivative(double[:] psi, double[:] x):
    """Estimate the second derivative d2psi/dx2 at discrete knots.

    Uses second-order finite-difference stencils at interior points
    and second-order endpoint formulas on non-uniform grids.

    Parameters
    ----------
    psi : double[:]
        Function values at knot positions.
    x : double[:]
        Knot coordinates (same length as psi).

    Returns
    -------
    numpy.ndarray of shape (N,)
    """
    cdef int N = psi.shape[0]
    cdef int ii
    assert N == x.shape[0], "Need to be same length"
    assert N > 2, "Need to be at least three points"
    out = np.zeros(N)
    cdef double[:] dpsi = out
    dpsi[0] = (
        2 * psi[0] / ((x[1] - x[0]) * (x[2] - x[0]))
        + 2 * psi[1] / ((x[1] - x[0]) * (x[1] - x[2]))
        + 2 * psi[2] / ((x[2] - x[0]) * (x[2] - x[1]))
    )
    dpsi[N - 1] = (
        2 * psi[N-1] / ((x[N-1] - x[N-2]) * (x[N-1] - x[N-3]))
        + 2 * psi[N-2] / ((x[N-3] - x[N-2]) * (x[N-1] - x[N-2]))
        + 2 * psi[N-3] / ((x[N-1] - x[N-3]) * (x[N-2] - x[N-3]))
    )
    for ii in range(1, N - 1):
        dpsi[ii] = (
            2 * psi[ii-1] / ((x[ii+1] - x[ii-1]) * (x[ii] - x[ii-1]))
            + 2 * psi[ii] / ((x[ii] - x[ii-1]) * (x[ii] - x[ii+1]))
            + 2 * psi[ii+1] / ((x[ii+1] - x[ii-1]) * (x[ii+1] - x[ii]))
        )
    return out


########################################################################
# Integral Weight Helpers
########################################################################

cdef _cubic_integrals(double a, double b, double x0, double x1):
    """Integral weight vectors for a cubic Hermite spline on [a, b].

    Returns (t, dt) such that t @ coefs gives the integral of the spline
    from a to b, and dt @ coefs gives the integral of its derivative.
    """
    cdef double t2, t3, t4, dt2, dt3, dt4
    t2 = (b - a) * (a + b - 2 * x0) / (2 * (x1 - x0))
    t3 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
        / (3 * (x1 - x0) ** 2)
    )
    t4 = ((a - x0) ** 4 - (b - x0) ** 4) / (4 * (x0 - x1) ** 3)
    dt2 = (b - a) / (x1 - x0)
    dt3 = (b - a) * (a + b - 2 * x0) / ((x1 - x0) ** 2)
    dt4 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
        / ((x1 - x0) ** 3)
    )
    return np.array([b - a, t2, t3, t4]), np.array([0.0, dt2, dt3, dt4])


cdef _quintic_integrals(double a, double b, double x0, double x1):
    """Integral weight vectors for a quintic Hermite spline on [a, b].

    Returns (t, dt) such that t @ coefs gives the integral of the spline
    from a to b, and dt @ coefs gives the integral of its derivative.
    """
    cdef double t2, t3, t4, t5, t6, dt2, dt3, dt4, dt5, dt6
    t2 = (b - a) * (a + b - 2 * x0) / (2 * (x1 - x0))
    t3 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
        / (3 * (x1 - x0) ** 2)
    )
    t4 = ((a - x0) ** 4 - (b - x0) ** 4) / (4 * (x0 - x1) ** 3)
    t5 = (-((a - x0) ** 5) + (b - x0) ** 5) / (5 * (x0 - x1) ** 4)
    t6 = ((a - x0) ** 6 - (b - x0) ** 6) / (6 * (x0 - x1) ** 5)
    dt2 = (b - a) / (x1 - x0)
    dt3 = (b - a) * (a + b - 2 * x0) / ((x1 - x0) ** 2)
    dt4 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
        / ((x1 - x0) ** 3)
    )
    dt5 = (-((a - x0) ** 4) + (b - x0) ** 4) / ((x0 - x1) ** 4)
    dt6 = (-((a - x0) ** 5) + (b - x0) ** 5) / ((x1 - x0) ** 5)
    return (
        np.array([b - a, t2, t3, t4, t5, t6]),
        np.array([0.0, dt2, dt3, dt4, dt5, dt6]),
    )


########################################################################
# Spline Extension Types
########################################################################

cdef class CubicHermite:
    """Cubic Hermite spline over a set of knots.

    Parameters
    ----------
    psi : array_like
        Function values at the knot points.
    knots_x : array_like
        Knot coordinates (monotonic sequence).
    """

    cdef double[:] psi
    cdef double[:] knots_x
    cdef double[:, :] coefs     # shape (4, N_intervals)

    def __init__(self, psi, knots_x):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int N
        dpsi_dx = first_derivative(self.psi, self.knots_x)
        knots = np.asarray(self.knots_x)
        psi_arr = np.asarray(self.psi)
        N = knots.shape[0]
        delta_x = knots[1:N] - knots[0:N-1]
        control = np.array(
            [
                psi_arr[0:N-1],
                psi_arr[1:N],
                dpsi_dx[0:N-1] * delta_x,
                dpsi_dx[1:N] * delta_x,
            ]
        )
        self.coefs = np.ascontiguousarray(CUBIC_BASIS @ control, dtype=np.float64)

    def _find_zone(self, n):
        cdef int Nk
        knots = np.asarray(self.knots_x)
        Nk = knots.shape[0]
        idx = np.digitize(n, bins=knots) - 1
        idx[idx == Nk - 1] = Nk - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, n):
        """Interpolate values at positions n."""
        cdef int ii
        if isinstance(n, float):
            n = np.array([n])
        n = np.asarray(n)
        idx = self._find_zone(n)
        knots = np.asarray(self.knots_x)
        t_norm = (n - knots[idx]) / (knots[idx + 1] - knots[idx])
        t = np.array([[1] * len(n), t_norm, t_norm**2, t_norm**3])
        splines_psi = np.zeros(n.shape[0])
        for _ii in np.unique(np.sort(idx)):
            ii = _ii
            inside = np.argwhere(idx == ii).flatten()
            splines_psi[inside] = t[:, inside].T @ self.coefs[:, ii]
        return splines_psi

    def integrate_edges(self):
        """Return (int_psi, int_dpsi) integrated over each knot-defined interval."""
        cdef int N, ii
        knots = np.asarray(self.knots_x)
        N = knots.shape[0] - 1
        delta_x = knots[1:N+1] - knots[0:N]
        t = np.array([delta_x, 0.5 * delta_x, delta_x / 3.0, 0.25 * delta_x])
        dt = np.ones((4, N))
        dt[0] = 0.0
        int_psi = np.zeros(N)
        int_dpsi = np.zeros(N)
        for ii in range(N):
            int_psi[ii] = np.dot(t[:, ii], self.coefs[:, ii])
            int_dpsi[ii] = np.dot(dt[:, ii], self.coefs[:, ii])
        return int_psi, int_dpsi

    def integrate_centers(self, limits_x):
        """Integrate spline over cell-centered limits.

        limits_x should be the cell boundaries (length N+1 for N knots).
        """
        cdef int N, ii
        limits_x = np.asarray(limits_x, dtype=np.float64)
        knots = np.asarray(self.knots_x)
        N = knots.shape[0]
        int_psi = np.zeros(N)
        int_dpsi = np.zeros(N)
        t, dt = _cubic_integrals(
            limits_x[0], limits_x[1], knots[0], knots[1]
        )
        int_psi[0] = np.dot(t, self.coefs[:, 0])
        int_dpsi[0] = np.dot(dt, self.coefs[:, 0])
        t, dt = _cubic_integrals(
            limits_x[N - 1], limits_x[N], knots[N - 2], knots[N - 1]
        )
        int_psi[N - 1] = np.dot(t, self.coefs[:, N - 2])
        int_dpsi[N - 1] = np.dot(dt, self.coefs[:, N - 2])
        for ii in range(1, N - 1):
            t1, dt1 = _cubic_integrals(
                limits_x[ii], knots[ii], knots[ii - 1], knots[ii]
            )
            t2, dt2 = _cubic_integrals(
                knots[ii], limits_x[ii + 1], knots[ii], knots[ii + 1]
            )
            int_psi[ii] = (
                np.dot(t1, self.coefs[:, ii - 1]) + np.dot(t2, self.coefs[:, ii])
            )
            int_dpsi[ii] = (
                np.dot(dt1, self.coefs[:, ii - 1]) + np.dot(dt2, self.coefs[:, ii])
            )
        return int_psi, int_dpsi


cdef class QuinticHermite:
    """Quintic Hermite spline over a set of knots.

    Uses function values, first, and second derivatives at knot points
    to build a degree-5 polynomial on each cell.

    Parameters
    ----------
    psi : array_like
        Function values at the knot points.
    knots_x : array_like
        Knot coordinates (monotonic sequence).
    """

    cdef double[:] psi
    cdef double[:] knots_x
    cdef double[:, :] coefs     # shape (6, N_intervals)

    def __init__(self, psi, knots_x):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int N
        dpsi_dx = first_derivative(self.psi, self.knots_x)
        d2psi_dx2 = second_derivative(self.psi, self.knots_x)
        knots = np.asarray(self.knots_x)
        psi_arr = np.asarray(self.psi)
        N = knots.shape[0]
        delta_x = knots[1:N] - knots[0:N-1]
        control = np.array(
            [
                psi_arr[0:N-1],
                psi_arr[1:N],
                dpsi_dx[0:N-1] * delta_x,
                dpsi_dx[1:N] * delta_x,
                d2psi_dx2[0:N-1] * delta_x**2,
                d2psi_dx2[1:N] * delta_x**2,
            ]
        )
        self.coefs = np.ascontiguousarray(QUINTIC_BASIS @ control, dtype=np.float64)

    def _find_zone(self, n):
        cdef int Nk
        if isinstance(n, float):
            n = np.array([n])
        knots = np.asarray(self.knots_x)
        Nk = knots.shape[0]
        idx = np.digitize(n, bins=knots) - 1
        idx[idx == Nk - 1] = Nk - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, n):
        """Evaluate the quintic spline at positions n."""
        cdef int ii
        if isinstance(n, float):
            n = np.array([n])
        n = np.asarray(n)
        idx = self._find_zone(n)
        knots = np.asarray(self.knots_x)
        t_norm = (n - knots[idx]) / (knots[idx + 1] - knots[idx])
        t = np.array(
            [[1] * len(n), t_norm, t_norm**2, t_norm**3, t_norm**4, t_norm**5]
        )
        splines_psi = np.zeros(n.shape[0])
        for _ii in np.unique(np.sort(idx)):
            ii = _ii
            inside = np.argwhere(idx == ii).flatten()
            splines_psi[inside] = t[:, inside].T @ self.coefs[:, ii]
        return splines_psi

    def integrate_edge(self, x0, x1):
        """Integrate the spline over a single cell [x0, x1]."""
        cdef double delta_x = x1 - x0
        cdef int zone
        idx = self._find_zone(0.5 * (x1 + x0))
        zone = idx[0]
        t = np.array(
            [
                delta_x,
                0.5 * delta_x,
                delta_x / 3.0,
                0.25 * delta_x,
                0.2 * delta_x,
                delta_x / 6.0,
            ]
        )
        dt = np.ones(6)
        dt[0] = 0.0
        int_psi = np.dot(t, self.coefs[:, zone])
        int_dpsi = np.dot(dt, self.coefs[:, zone])
        return int_psi, int_dpsi

    def integrate_edges(self):
        """Return (int_psi, int_dpsi) integrated over each knot-defined interval."""
        cdef int N, ii
        knots = np.asarray(self.knots_x)
        N = knots.shape[0] - 1
        delta_x = knots[1:N+1] - knots[0:N]
        t = np.array(
            [
                delta_x,
                0.5 * delta_x,
                delta_x / 3.0,
                0.25 * delta_x,
                0.2 * delta_x,
                delta_x / 6.0,
            ]
        )
        dt = np.ones((6, N))
        dt[0] = 0.0
        int_psi = np.zeros(N)
        int_dpsi = np.zeros(N)
        for ii in range(N):
            int_psi[ii] = np.dot(t[:, ii], self.coefs[:, ii])
            int_dpsi[ii] = np.dot(dt[:, ii], self.coefs[:, ii])
        return int_psi, int_dpsi

    def integrate_centers(self, limits_x):
        """Integrate quintic spline over cell-centered limits.

        limits_x should be the cell boundaries (length N+1 for N knots).
        """
        cdef int N, ii
        limits_x = np.asarray(limits_x, dtype=np.float64)
        knots = np.asarray(self.knots_x)
        N = knots.shape[0]
        int_psi = np.zeros(N)
        int_dpsi = np.zeros(N)
        t, dt = _quintic_integrals(
            limits_x[0], limits_x[1], knots[0], knots[1]
        )
        int_psi[0] = np.dot(t, self.coefs[:, 0])
        int_dpsi[0] = np.dot(dt, self.coefs[:, 0])
        t, dt = _quintic_integrals(
            limits_x[N - 1], limits_x[N], knots[N - 2], knots[N - 1]
        )
        int_psi[N - 1] = np.dot(t, self.coefs[:, N - 2])
        int_dpsi[N - 1] = np.dot(dt, self.coefs[:, N - 2])
        for ii in range(1, N - 1):
            t1, dt1 = _quintic_integrals(
                limits_x[ii], knots[ii], knots[ii - 1], knots[ii]
            )
            t2, dt2 = _quintic_integrals(
                knots[ii], limits_x[ii + 1], knots[ii], knots[ii + 1]
            )
            int_psi[ii] = (
                np.dot(t1, self.coefs[:, ii - 1]) + np.dot(t2, self.coefs[:, ii])
            )
            int_dpsi[ii] = (
                np.dot(dt1, self.coefs[:, ii - 1]) + np.dot(dt2, self.coefs[:, ii])
            )
        return int_psi, int_dpsi


########################################################################
# Block and Convenience Wrappers
########################################################################

class BlockInterpolation:
    """Apply a spline class on disjoint x-blocks.

    Parameters
    ----------
    Splines : class
        Either CubicHermite or QuinticHermite.
    psi, knots_x : array_like
        Global arrays of function values and knot coordinates.
    medium_map : array_like
        Per-cell medium mapping used to infer block boundaries when
        x_splits is empty.
    x_splits : array_like
        Explicit block boundary indices. If empty, derived from medium_map.
    """

    def __init__(self, Splines, psi, knots_x, medium_map, x_splits):
        self.Splines = Splines
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        if np.asarray(x_splits).shape[0] > 0:
            self.x_splits = np.asarray(x_splits).copy()
        else:
            self.x_splits = pytools._to_block(np.asarray(medium_map))
        self._generate_coefs()

    def _generate_coefs(self):
        cdef int ii, n_blocks
        n_blocks = self.x_splits.shape[0] - 1
        self.splines = []
        for ii in range(n_blocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            self.splines.append(self.Splines(self.psi[x1:x2], self.knots_x[x1:x2]))

    def interpolate(self, nx):
        """Interpolate values using the block-wise splines."""
        cdef int ii, n_blocks
        if isinstance(nx, float):
            nx = np.array([nx])
        nx = np.asarray(nx)
        splines_psi = np.zeros(nx.shape[0])
        n_blocks = self.x_splits.shape[0] - 1
        for ii in range(n_blocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            if ii == 0:
                idx_x = np.argwhere(nx < self.knots_x[x2]).flatten()
            elif ii == n_blocks - 1:
                idx_x = np.argwhere(nx >= self.knots_x[x1]).flatten()
            else:
                idx_x = np.argwhere(
                    (nx >= self.knots_x[x1]) & (nx < self.knots_x[x2])
                ).flatten()
            splines_psi[idx_x] = self.splines[ii].interpolate(nx[idx_x])
        return splines_psi

    def integrate_edges(self):
        """Compute (int_psi, int_dpsi) for all edges across blocks."""
        cdef int ii, n_blocks
        Nx = self.knots_x.shape[0] - 1
        int_psi = np.zeros(Nx)
        int_dpsi = np.zeros(Nx)
        n_blocks = self.x_splits.shape[0] - 1
        for ii in range(n_blocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            approx = self.Splines(self.psi[x1:x2 + 1], self.knots_x[x1:x2 + 1])
            b_int_psi, b_int_dpsi = approx.integrate_edges()
            int_psi[x1:x2] = b_int_psi
            int_dpsi[x1:x2] = b_int_dpsi
        return int_psi, int_dpsi

    def integrate_centers(self, limits_x):
        """Compute (int_psi, int_dpsi) for cell-centered limits across blocks."""
        cdef int ii, n_blocks
        limits_x = np.asarray(limits_x)
        Nx = self.knots_x.shape[0]
        int_psi = np.zeros(Nx)
        int_dpsi = np.zeros(Nx)
        n_blocks = self.x_splits.shape[0] - 1
        for ii in range(n_blocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            approx = self.Splines(self.psi[x1:x2], self.knots_x[x1:x2])
            b_int_psi, b_int_dpsi = approx.integrate_centers(limits_x[x1:x2 + 1])
            int_psi[x1:x2] = b_int_psi
            int_dpsi[x1:x2] = b_int_dpsi
        return int_psi, int_dpsi


class Interpolation:
    """Convenience wrapper that selects the appropriate spline/block combination.

    Parameters
    ----------
    psi, knots_x : array_like
        Global function values and knot coordinates.
    medium_map : array_like
        Passed through to BlockInterpolation when block is True.
    x_splits : array_like
        Explicit block boundaries (used when block is True).
    block : bool
        If True, build a block-wise interpolator.
    quintic : bool
        If True, use the quintic Hermite basis; otherwise use cubic.
    """

    def __init__(self, psi, knots_x, medium_map, x_splits, block=True, quintic=True):
        if block and quintic:
            self.instance = BlockInterpolation(
                QuinticHermite, psi, knots_x, medium_map, x_splits
            )
        elif block:
            self.instance = BlockInterpolation(
                CubicHermite, psi, knots_x, medium_map, x_splits
            )
        elif quintic:
            self.instance = QuinticHermite(psi, knots_x)
        else:
            self.instance = CubicHermite(psi, knots_x)

    def interpolate(self, n):
        return self.instance.interpolate(n)

    def integrate_edges(self):
        return self.instance.integrate_edges()

    def integrate_centers(self, limits_x):
        return self.instance.integrate_centers(limits_x)
