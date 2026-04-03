########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Used for the curve fit approximation with the Method of Nearby
# Problems. Includes the first and second derivative approximation
# (second order), Cubic Hermite splines, and Quintic Hermite splines.
#
########################################################################

"""1D interpolation utilities used by ANTS.

This module provides helpers for constructing Hermite-style spline
approximations over 1D data. It is used by the Method of Nearby Problems
implementation and exposes:

- finite-difference based first and second derivative approximations
- CubicHermite: cubic Hermite spline interpolation with integrals
- QuinticHermite: quintic Hermite spline interpolation with integrals
- BlockInterpolation: apply a spline on a per-block basis
- Interpolation: convenience wrapper that selects the appropriate
    spline/block combination

Notes
-----
The spline classes expect ``psi`` and ``knots_x`` to have the same
length. Many of the methods accept either a scalar or a 1D numpy array
as input coordinates for interpolation and return a numpy array of
interpolated values. Integrator methods return tuple of (integral of
psi over cell, integral of dpsi/dx over cell) depending on method.

The implementations intentionally use small dense linear algebra
operations (matrix multiplies with stored basis matrices) for clarity
and reasonable performance on the small vectors used by the library.

Examples
--------
>>> import numpy as np
>>> from ants.utils import interp1d
>>> x = np.linspace(0, 1, 5)
>>> psi = x**3 - 2*x**2 + 2
>>> # Cubic interpolation
>>> ch = interp1d.CubicHermite(psi, x)
>>> ch.interpolate(x)  # should reproduce values at knots
array([2.    , 1.878 , 1.5   , 1.   , 1.    ])
>>> # Quintic interpolation
>>> qh = interp1d.QuinticHermite(psi, x)
>>> qh.interpolate(x)  # also reproduces knot values
array([2.    , 1.878 , 1.5   , 1.   , 1.    ])
"""

import numpy as np

from ants.utils import pytools


def first_derivative(psi, x):
    """Estimate the first derivative dpsi/dx at discrete knots.

    Uses a second-order accurate finite-difference stencil at interior
    points and a second-order accurate endpoint formula. The returned
    array has the same length as ``psi``.

    Parameters
    ----------
    psi : array_like
        Function values at knot positions.
    x : array_like
        Knot coordinates (must be same length as ``psi``).

    Returns
    -------
    numpy.ndarray
        Array of first derivative estimates of shape (N,).

    Raises
    ------
    AssertionError
        If input lengths differ or there are fewer than 3 knots.
    """
    # Ensure same length
    assert len(psi) == len(x), "Need to be same length"
    # Get size of array
    N = len(psi)
    assert N > 2, "Need to be at least three knots"
    # Initialize derivative array
    dpsi = np.zeros((N,))
    # Second order accurate endpoints
    dpsi[0] = (
        (psi[0] - psi[1]) / (x[0] - x[1])
        + (psi[0] - psi[2]) / (x[0] - x[2])
        + (-psi[1] + psi[2]) / (x[1] - x[2])
    )
    dpsi[N - 1] = (
        (psi[N - 1] - psi[N - 2]) / (x[N - 1] - x[N - 2])
        + (psi[N - 1] - psi[N - 3]) / (x[N - 1] - x[N - 3])
        + (-psi[N - 2] + psi[N - 3]) / (x[N - 2] - x[N - 3])
    )
    # Iterate over middle points
    for ii in range(1, N - 1):
        dpsi[ii] = (
            (psi[ii] - psi[ii - 1]) / (x[ii] - x[ii - 1])
            + (psi[ii] - psi[ii + 1]) / (x[ii] - x[ii + 1])
            + (-psi[ii - 1] + psi[ii + 1]) / (x[ii - 1] - x[ii + 1])
        )
    return dpsi


def second_derivative(psi, x):
    """Estimate the second derivative d2psi/dx2 at discrete knots.

    The implementation uses a compact second-order accurate finite
    difference approximation at interior points and special endpoint
    formulas that remain second-order accurate on non-uniform grids.

    Parameters
    ----------
    psi : array_like
        Function values at knot positions.
    x : array_like
        Knot coordinates (must be same length as ``psi``).

    Returns
    -------
    numpy.ndarray
        Array of second derivative estimates of shape (N,).

    Raises
    ------
    AssertionError
        If input lengths differ or there are fewer than 3 knots.
    """
    # Ensure same length
    assert len(psi) == len(x), "Need to be same length"
    # Get size of array
    N = len(psi)
    assert N > 2, "Need to be at least three points"
    dpsi = np.zeros((N,))
    # Second order accurate endpoints
    dpsi[0] = (
        2 * psi[0] / ((x[1] - x[0]) * (x[2] - x[0]))
        + 2 * psi[1] / ((x[1] - x[0]) * (x[1] - x[2]))
        + 2 * psi[2] / ((x[2] - x[0]) * (x[2] - x[1]))
    )
    dpsi[N - 1] = (
        2 * psi[N - 1] / ((x[N - 1] - x[N - 2]) * (x[N - 1] - x[N - 3]))
        + 2 * psi[N - 2] / ((x[N - 3] - x[N - 2]) * (x[N - 1] - x[N - 2]))
        + 2 * psi[N - 3] / ((x[N - 1] - x[N - 3]) * (x[N - 2] - x[N - 3]))
    )
    # Iterate over midpoints
    for ii in range(1, N - 1):
        dpsi[ii] = (
            2 * psi[ii - 1] / ((x[ii + 1] - x[ii - 1]) * (x[ii] - x[ii - 1]))
            + 2 * psi[ii] / ((x[ii] - x[ii - 1]) * (x[ii] - x[ii + 1]))
            + 2 * psi[ii + 1] / ((x[ii + 1] - x[ii - 1]) * (x[ii + 1] - x[ii]))
        )
    return dpsi


class CubicHermite:
    basis = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])

    def __init__(self, psi, knots_x):
        """Construct a cubic Hermite spline over given knots.

        Parameters
        ----------
        psi : array_like
            Function values at the knot points.
        knots_x : array_like
            Knot coordinates (monotonic sequence).
        """
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self._generate_coefs()

    def _generate_coefs(self):
        """Compute the cubic polynomial coefficients for each interval.

        This populates ``self.coefs`` with shape (4, N_intervals) where each
        column contains the polynomial coefficients for the local
        representation in the normalized coordinate t in [0,1].
        """
        dpsi_dx = first_derivative(self.psi, self.knots_x)
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        control = np.array(
            [self.psi[:-1], self.psi[1:], dpsi_dx[:-1] * delta_x, dpsi_dx[1:] * delta_x]
        )
        self.coefs = CubicHermite.basis @ control

    def _find_zone(self, n):
        idx = np.digitize(n, bins=self.knots_x) - 1
        idx[idx == len(self.knots_x) - 1] = len(self.knots_x) - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, n):
        """Interpolate values at positions ``n``.

        Parameters
        ----------
        n : float or array_like
            Coordinates to evaluate. A scalar or 1D array are accepted.

        Returns
        -------
        numpy.ndarray
            Interpolated values with the same shape as the input points
            flattened to a 1D array.
        """
        if isinstance(n, float):
            n = np.array([n])
        n = np.asarray(n)
        idx = self._find_zone(n)
        # Normalize input
        t = (n - self.knots_x[idx]) / (self.knots_x[idx + 1] - self.knots_x[idx])
        t = np.array([[1] * len(n), t, t**2, t**3])
        # Iterate over each zone
        splines_psi = np.zeros((n.shape[0]))
        for ii in np.unique(np.sort(idx)):
            inside = np.argwhere(idx == ii).flatten()
            splines_psi[inside] = t[:, inside].T @ self.coefs[:, ii]
        return splines_psi

    # Integral of X - edges
    def integrate_edges(self):
        # Take integral integral of derivative
        """Return integrals of the spline over each cell defined by knots.

        Returns
        -------
        (int_psi, int_dpsi)
        int_psi : numpy.ndarray
            Cellwise integrals of the spline (integral of psi over cell).
        int_dpsi : numpy.ndarray
            Cellwise integrals of the derivative basis (useful when
            computing cell-averaged derivative contributions).
        """
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        N = delta_x.shape[0]
        t = np.array([delta_x, 0.5 * delta_x, 1 / 3.0 * delta_x, 0.25 * delta_x])
        dt = np.ones((4, N))
        dt[0] = 0.0
        # Calculate splines
        int_psi = np.array([t[:, ii].T @ self.coefs[:, ii] for ii in range(N)])
        int_dpsi = np.array([dt[:, ii].T @ self.coefs[:, ii] for ii in range(N)])
        return int_psi, int_dpsi

    def _integrals(a, b, x0, x1):
        """Helper: compute integral weight vectors for an interval.

        Returns two arrays (t, dt) such that t.T @ coefs gives the
        integral of psi from a to b and dt.T @ coefs gives the integral
        of the derivative basis over the same limits. This is a
        low-level method used by integrate_centers.
        """
        t2 = (b - a) * (a + b - 2 * x0) / (2 * (x1 - x0))
        t3 = (
            (b - a)
            * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
            / (3 * (x1 - x0) ** 2)
        )
        t4 = ((a - x0) ** 4 - (b - x0) ** 4) / (4 * (x0 - x1) ** 3)
        # Integral of dpsi between a and b with knots x0 and x1
        dt2 = (b - a) / (x1 - x0)
        dt3 = (b - a) * (a + b - 2 * x0) / ((x1 - x0) ** 2)
        dt4 = (
            (b - a)
            * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
            / ((x1 - x0) ** 3)
        )
        return np.array([b - a, t2, t3, t4]), np.array([0, dt2, dt3, dt4])

    # Integral of X - centers
    def integrate_centers(self, limits_x):
        """Integrate spline over cell-centered limits.

        ``limits_x`` should be the cell boundaries (length == N+1 for N
        knots). The method returns arrays of length N containing the
        integral of psi and the integral of dpsi/dx for each cell.
        """
        N = self.knots_x.shape[0]
        int_psi = np.zeros((N,))
        int_dpsi = np.zeros((N,))
        # First cell
        t, dt = CubicHermite._integrals(
            limits_x[0], limits_x[1], self.knots_x[0], self.knots_x[1]
        )
        int_psi[0] = t.T @ self.coefs[:, 0]
        int_dpsi[0] = dt.T @ self.coefs[:, 0]
        # Last Cell
        t, dt = CubicHermite._integrals(
            limits_x[-2], limits_x[-1], self.knots_x[-2], self.knots_x[-1]
        )
        int_psi[-1] = t.T @ self.coefs[:, -1]
        int_dpsi[-1] = dt.T @ self.coefs[:, -1]
        # Interate over spatial dimension
        for ii, (a, b) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            if (ii == 0) or ii == (N - 1):
                continue
            t1, dt1 = CubicHermite._integrals(
                a, self.knots_x[ii], self.knots_x[ii - 1], self.knots_x[ii]
            )
            t2, dt2 = CubicHermite._integrals(
                self.knots_x[ii], b, self.knots_x[ii], self.knots_x[ii + 1]
            )
            int_psi[ii] = (t1.T @ self.coefs[:, ii - 1]) + (t2.T @ self.coefs[:, ii])
            int_dpsi[ii] = (dt1.T @ self.coefs[:, ii - 1]) + (dt2.T @ self.coefs[:, ii])
        return int_psi, int_dpsi


class QuinticHermite:
    basis = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0],
            [-10, 10, -6, -4, -1.5, 0.5],
            [15, -15, 8, 7, 1.5, -1],
            [-6, 6, -3, -3, -0.5, 0.5],
        ]
    )

    def __init__(self, psi, knots_x):
        """Construct a quintic Hermite spline over given knots.

        The quintic spline uses function values, first and second
        derivatives at knot points to build a degree-5 polynomial on
        each cell. Inputs are similar to :class:`CubicHermite`.
        """
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self._generate_coefs()

    def _generate_coefs(self):
        """Compute quintic polynomial coefficients for each interval.

        The control matrix contains values and scaled derivatives used to
        compute local polynomial coefficients via the stored basis
        matrix.
        """
        dpsi_dx = first_derivative(self.psi, self.knots_x)
        d2psi_dx2 = second_derivative(self.psi, self.knots_x)
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        control = np.array(
            [
                self.psi[:-1],
                self.psi[1:],
                dpsi_dx[:-1] * delta_x,
                dpsi_dx[1:] * delta_x,
                d2psi_dx2[:-1] * delta_x**2,
                d2psi_dx2[1:] * delta_x**2,
            ]
        )
        self.coefs = QuinticHermite.basis @ control

    def _find_zone(self, n):
        if isinstance(n, float):
            n = np.array([n])
        idx = np.digitize(n, bins=self.knots_x) - 1
        idx[idx == len(self.knots_x) - 1] = len(self.knots_x) - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, n):
        """Evaluate the quintic spline at positions ``n``.

        See :meth:`CubicHermite.interpolate` for accepted input types and
        return shape.
        """
        if isinstance(n, float):
            n = np.array([n])
        n = np.asarray(n)
        idx = self._find_zone(n)
        # Normalize input
        t = (n - self.knots_x[idx]) / (self.knots_x[idx + 1] - self.knots_x[idx])
        t = np.array([[1] * len(n), t, t**2, t**3, t**4, t**5])
        # Iterate over each zone
        splines_psi = np.zeros((n.shape[0]))
        for ii in np.unique(np.sort(idx)):
            inside = np.argwhere(idx == ii).flatten()
            splines_psi[inside] = t[:, inside].T @ self.coefs[:, ii]
        return splines_psi

    # Integral of X - single cell
    def integrate_edge(self, x0, x1):
        # Take integral integral of derivative
        """Integrate the spline over a single cell [x0, x1].

        Returns the pair (integral of psi over the cell, integral of the
        derivative basis over the cell).
        """
        delta_x = x1 - x0
        idx = self._find_zone(0.5 * (x1 + x0))
        t = np.array(
            [
                delta_x,
                0.5 * delta_x,
                1 / 3.0 * delta_x,
                0.25 * delta_x,
                0.2 * delta_x,
                1 / 6.0 * delta_x,
            ]
        )
        dt = np.ones((6,))
        dt[0] = 0.0
        # Calculate splines
        int_psi = t.T @ self.coefs[:, idx]
        int_dpsi = dt.T @ self.coefs[:, idx]
        return int_psi, int_dpsi

    # Integral of X - edges as knots
    def integrate_edges(self):
        # Take integral integral of derivative
        """Return integrals over all knot-defined edges (cellwise)."""
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        N = delta_x.shape[0]
        t = np.array(
            [
                delta_x,
                0.5 * delta_x,
                1 / 3.0 * delta_x,
                0.25 * delta_x,
                0.2 * delta_x,
                1 / 6.0 * delta_x,
            ]
        )
        dt = np.ones((6, N))
        dt[0] = 0.0
        # Calculate splines
        int_psi = np.array([t[:, ii].T @ self.coefs[:, ii] for ii in range(N)])
        int_dpsi = np.array([dt[:, ii].T @ self.coefs[:, ii] for ii in range(N)])
        return int_psi, int_dpsi

    def _integrals(a, b, x0, x1):
        """Compute weight vectors for integrals over [a,b] for quintic.

        See :meth:`CubicHermite._integrals` for the purpose of the return
        values (t, dt).
        """
        t2 = (b - a) * (a + b - 2 * x0) / (2 * (x1 - x0))
        t3 = (
            (b - a)
            * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
            / (3 * (x1 - x0) ** 2)
        )
        t4 = ((a - x0) ** 4 - (b - x0) ** 4) / (4 * (x0 - x1) ** 3)
        t5 = (-((a - x0) ** 5) + (b - x0) ** 5) / (5 * (x0 - x1) ** 4)
        t6 = ((a - x0) ** 6 - (b - x0) ** 6) / (6 * (x0 - x1) ** 5)
        t = np.array([b - a, t2, t3, t4, t5, t6])
        # Integral of dpsi between a and b with knots x0 and x1
        dt2 = (b - a) / (x1 - x0)
        dt3 = (b - a) * (a + b - 2 * x0) / ((x1 - x0) ** 2)
        dt4 = (
            (b - a)
            * (a**2 + a * b + b**2 - 3 * (a + b) * x0 + 3 * x0**2)
            / ((x1 - x0) ** 3)
        )
        dt5 = (-((a - x0) ** 4) + (b - x0) ** 4) / ((x0 - x1) ** 4)
        dt6 = (-((a - x0) ** 5) + (b - x0) ** 5) / ((x1 - x0) ** 5)
        dt = np.array([0, dt2, dt3, dt4, dt5, dt6])
        return t, dt

    # Integral of X - centers as knots
    def integrate_centers(self, limits_x):
        """Integrate quintic spline over cell-centered limits.

        Similar contract to :meth:`CubicHermite.integrate_centers` but
        using the quintic polynomial basis.
        """
        N = self.knots_x.shape[0]
        int_psi = np.zeros((N,))
        int_dpsi = np.zeros((N,))
        # First cell
        t, dt = QuinticHermite._integrals(
            limits_x[0], limits_x[1], self.knots_x[0], self.knots_x[1]
        )
        int_psi[0] = t.T @ self.coefs[:, 0]
        int_dpsi[0] = dt.T @ self.coefs[:, 0]
        # Last Cell
        t, dt = QuinticHermite._integrals(
            limits_x[-2], limits_x[-1], self.knots_x[-2], self.knots_x[-1]
        )
        int_psi[-1] = t.T @ self.coefs[:, -1]
        int_dpsi[-1] = dt.T @ self.coefs[:, -1]
        # Interate over spatial dimension
        for ii, (a, b) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            if (ii == 0) or ii == (N - 1):
                continue
            t1, dt1 = QuinticHermite._integrals(
                a, self.knots_x[ii], self.knots_x[ii - 1], self.knots_x[ii]
            )
            t2, dt2 = QuinticHermite._integrals(
                self.knots_x[ii], b, self.knots_x[ii], self.knots_x[ii + 1]
            )
            int_psi[ii] = (t1.T @ self.coefs[:, ii - 1]) + (t2.T @ self.coefs[:, ii])
            int_dpsi[ii] = (dt1.T @ self.coefs[:, ii - 1]) + (dt2.T @ self.coefs[:, ii])
        return int_psi, int_dpsi


class BlockInterpolation:

    def __init__(self, Splines, psi, knots_x, medium_map, x_splits):
        """Apply a spline class on disjoint x-blocks.

        Parameters
        ----------
        Splines : class
            Either :class:`CubicHermite` or :class:`QuinticHermite` (or any
            compatible class providing the same interface).
        psi, knots_x : array_like
            Global arrays of function values and knot coordinates.
        medium_map : array_like
            Per-cell medium mapping used to infer block boundaries if
            ``x_splits`` is empty.
        x_splits : array_like
            Optional explicit block boundary indices. If empty, the
            ``medium_map`` is converted to blocks with
            ``pytools._to_block``.
        """
        self.Splines = Splines
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        # Determine knot splits
        if x_splits.shape[0] > 0:
            self.x_splits = np.asarray(x_splits).copy()
        else:
            medium_map = np.asarray(medium_map)
            self.x_splits = pytools._to_block(medium_map)
        self._generate_coefs()

    def _generate_coefs(self):
        """Create one spline instance per block and store them in a list.

        Each block receives the subset of ``psi`` and ``knots_x`` that
        belong to that block.
        """
        # Create 2D list of Interpolations
        self.splines = []
        for x1, x2 in zip(self.x_splits[:-1], self.x_splits[1:]):
            # Initialize new spline section
            self.splines.append(self.Splines(self.psi[x1:x2], self.knots_x[x1:x2]))

    def interpolate(self, nx):
        """Interpolate values using the block-wise splines.

        ``nx`` may be a scalar or 1D array. The method dispatches each
        query point to the spline instance for its block and returns a
        flattened 1D array of values.
        """
        if isinstance(nx, float):
            nx = np.array([nx])
        nx = np.asarray(nx)
        # Initialize splines
        splines_psi = np.zeros((nx.shape[0],))
        # Iterate over x blocks
        for ii, (x1, x2) in enumerate(zip(self.x_splits[:-1], self.x_splits[1:])):
            # Find correct x block
            if ii == 0:
                idx_x = np.argwhere(nx < self.knots_x[x2]).flatten()
            elif ii == (self.x_splits.shape[0] - 2):
                idx_x = np.argwhere(nx >= self.knots_x[x1]).flatten()
            else:
                idx_x = np.argwhere(
                    (nx >= self.knots_x[x1]) & (nx < self.knots_x[x2])
                ).flatten()
            # Interpolate on block
            splines_psi[idx_x] = self.splines[ii].interpolate(nx[idx_x])
        return splines_psi

    # Integral of X - edges
    def integrate_edges(self):
        """Compute cellwise integrals for all edges across blocks.

        Returns arrays aligned with the global knot indexing.
        """
        # Initialize full matrices
        Nx = self.knots_x.shape[0] - 1
        int_psi = np.zeros((Nx,))
        int_dx = np.zeros((Nx,))
        # Iterate over each block
        for x1, x2 in zip(self.x_splits[:-1], self.x_splits[1:]):
            # Initialize new spline section
            approx = self.Splines(self.psi[x1 : x2 + 1], self.knots_x[x1 : x2 + 1])
            # Interpolate on block
            b_int_psi, b_int_dx = approx.integrate_edges()
            int_psi[x1:x2] = b_int_psi.copy()
            int_dx[x1:x2] = b_int_dx.copy()
        return int_psi, int_dx

    # Integral of X - centers
    def integrate_centers(self, limits_x):
        """Compute cell-centered integrals for all blocks.

        ``limits_x`` should be the global cell boundary array.
        """
        limits_x = np.asarray(limits_x)
        Nx = self.knots_x.shape[0]
        int_psi = np.zeros((Nx,))
        int_dx = np.zeros((Nx,))
        # Iterate over each block
        for x1, x2 in zip(self.x_splits[:-1], self.x_splits[1:]):
            # Initialize new spline section
            approx = self.Splines(self.psi[x1:x2], self.knots_x[x1:x2])
            # Interpolate on block
            b_int_psi, b_int_dx = approx.integrate_centers(limits_x[x1 : x2 + 1])
            int_psi[x1:x2] = b_int_psi.copy()
            int_dx[x1:x2] = b_int_dx.copy()
        return int_psi, int_dx


class Interpolation:

    def __init__(self, psi, knots_x, medium_map, x_splits, block=True, quintic=True):
        """Convenience wrapper selecting spline/blocking strategy.

        Parameters
        ----------
        psi, knots_x : array_like
            Global function values and knot coordinates.
        medium_map : array_like
            Passed through to :class:`BlockInterpolation` when ``block`` is
            True.
        x_splits : array_like
            If ``block`` is True, the explicit block boundaries. Otherwise
            ignored.
        block : bool
            If True, build a block-wise interpolator.
        quintic : bool
            If True, use the quintic Hermite basis; otherwise use cubic.
        """

        self.block = block
        self.quintic = quintic
        # Block Quintic
        if (block) and (quintic):
            self.instance = BlockInterpolation(
                QuinticHermite, psi, knots_x, medium_map, x_splits
            )

        # Quintic
        elif (not block) and (quintic):
            self.instance = QuinticHermite(psi, knots_x)

        # Block Cubic
        elif (block) and (not quintic):
            self.instance = BlockInterpolation(
                CubicHermite, psi, knots_x, medium_map, x_splits
            )

        # Cubic
        elif (not block) and (not quintic):
            self.instance = CubicHermite(psi, knots_x)

    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)
