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

import numpy as np

from ants.utils import pytools


def first_derivative(psi, x):
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
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self._generate_coefs()

    def _generate_coefs(self):
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
        # Integral of psi between a and b with knots x0 and x1
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
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self._generate_coefs()

    def _generate_coefs(self):
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
        # Integral of psi between a and b with knots x0 and x1
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
        # Create 2D list of Interpolations
        self.splines = []
        for x1, x2 in zip(self.x_splits[:-1], self.x_splits[1:]):
            # Initialize new spline section
            self.splines.append(self.Splines(self.psi[x1:x2], self.knots_x[x1:x2]))

    def interpolate(self, nx):
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
        # Initialize full matrices
        Nx = self.knots_x.shape[0] - 1
        int_psi = np.zeros((Nx,))
        int_dx = np.zeros((Nx,))
        int_dy = np.zeros((Nx,))
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
