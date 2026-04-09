########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two-Dimensional Interpolation Utilities
#
# Provides derivative approximations and bi-Hermite spline interpolation
# used by the Method of Nearby Problems.
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

import numpy as np

cimport numpy as np

from ants.utils import interp1d, pytools

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

cpdef first_derivative(double[:, :] psi, double[:] x, double[:] y):
    """Estimate dpsi/dx and dpsi/dy at each knot on a 2D grid.

    Parameters
    ----------
    psi : double[:,:]
        Function values; shape (Nx, Ny).
    x : double[:]
        Knot coordinates in x; length Nx.
    y : double[:]
        Knot coordinates in y; length Ny.

    Returns
    -------
    (d_dx, d_dy) : tuple of numpy.ndarray, each shape (Nx, Ny)
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    Nx = x_np.shape[0]
    Ny = y_np.shape[0]
    assert (Nx, Ny) == psi_np.shape, "Shape mismatch"
    assert Nx > 2 and Ny > 2, "Need at least 3 knots in each direction"
    d_dx = np.zeros((Nx, Ny))
    d_dy = np.zeros((Nx, Ny))
    for jj in range(Ny):
        d_dx[:, jj] = interp1d.first_derivative(psi_np[:, jj], x_np)
    for ii in range(Nx):
        d_dy[ii] = interp1d.first_derivative(psi_np[ii], y_np)
    return d_dx, d_dy


cpdef second_derivative(double[:, :] psi, double[:] x, double[:] y):
    """Estimate d2psi/dx2, d2psi/dxdy, and d2psi/dy2 at each knot.

    Parameters
    ----------
    psi : double[:,:]
        Function values; shape (Nx, Ny).
    x : double[:]
        Knot coordinates in x.
    y : double[:]
        Knot coordinates in y.

    Returns
    -------
    (d2_dx2, d2_dxdy, d2_dy2) : tuple of numpy.ndarray, each shape (Nx, Ny)
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    Nx = x_np.shape[0]
    Ny = y_np.shape[0]
    d2_dx2 = np.zeros((Nx, Ny))
    d2_dy2 = np.zeros((Nx, Ny))
    d2_dxdy = np.zeros((Nx, Ny))
    for jj in range(Ny):
        d2_dx2[:, jj] = interp1d.second_derivative(psi_np[:, jj], x_np)
        d2_dxdy[:, jj] = interp1d.first_derivative(psi_np[:, jj], x_np)
    for ii in range(Nx):
        d2_dy2[ii] = interp1d.second_derivative(psi_np[ii], y_np)
        d2_dxdy[ii] = interp1d.first_derivative(d2_dxdy[ii], y_np)
    return d2_dx2, d2_dxdy, d2_dy2


cpdef higher_order_derivative(double[:, :] psi, double[:] x, double[:] y):
    """Estimate the quintic cross-derivatives at each knot.

    Returns d3psi/dx2dy, d3psi/dxdy2, d4psi/dx2dy2.

    Parameters
    ----------
    psi : double[:,:]
        Function values; shape (Nx, Ny).
    x, y : double[:]
        Knot coordinates.

    Returns
    -------
    (d3_dx2dy, d3_dxdy2, d4_dx2dy2) : tuple of numpy.ndarray, each shape (Nx, Ny)
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    Nx = x_np.shape[0]
    Ny = y_np.shape[0]
    d3_dx2dy = np.zeros((Nx, Ny))
    d3_dxdy2 = np.zeros((Nx, Ny))
    d4_dx2dy2 = np.zeros((Nx, Ny))
    for jj in range(Ny):
        d3_dx2dy[:, jj] = interp1d.second_derivative(psi_np[:, jj], x_np)
        d3_dxdy2[:, jj] = interp1d.first_derivative(psi_np[:, jj], x_np)
        d4_dx2dy2[:, jj] = interp1d.second_derivative(psi_np[:, jj], x_np)
    for ii in range(Nx):
        d3_dx2dy[ii] = interp1d.first_derivative(d3_dx2dy[ii], y_np)
        d3_dxdy2[ii] = interp1d.second_derivative(d3_dxdy2[ii], y_np)
        d4_dx2dy2[ii] = interp1d.second_derivative(d4_dx2dy2[ii], y_np)
    return d3_dx2dy, d3_dxdy2, d4_dx2dy2


########################################################################
# Integral Weight Helpers
########################################################################

cdef _cubic_one_integral(double a, double b, double k0, double k1):
    """1D integral weight vector for a cubic Hermite spline on [a, b].

    Returns (t, dt): weight arrays such that t @ coef_col gives the
    integral of psi and dt @ coef_col gives the integral of dpsi/dk.
    """
    cdef double t2, t3, t4, dt2, dt3, dt4
    t2 = (b - a) * (a + b - 2 * k0) / (2 * (k1 - k0))
    t3 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2)
        / (3 * (k1 - k0) ** 2)
    )
    t4 = ((a - k0) ** 4 - (b - k0) ** 4) / (4 * (k0 - k1) ** 3)
    dt2 = (b - a) / (k1 - k0)
    dt3 = (b - a) * (a + b - 2 * k0) / ((k1 - k0) ** 2)
    dt4 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2)
        / ((k1 - k0) ** 3)
    )
    return np.array([b - a, t2, t3, t4]), np.array([0.0, dt2, dt3, dt4])


cdef _quintic_one_integral(double a, double b, double k0, double k1):
    """1D integral weight vector for a quintic Hermite spline on [a, b]."""
    cdef double t2, t3, t4, t5, t6, dt2, dt3, dt4, dt5, dt6
    t2 = (b - a) * (a + b - 2 * k0) / (2 * (k1 - k0))
    t3 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2)
        / (3 * (k1 - k0) ** 2)
    )
    t4 = ((a - k0) ** 4 - (b - k0) ** 4) / (4 * (k0 - k1) ** 3)
    t5 = (-((a - k0) ** 5) + (b - k0) ** 5) / (5 * (k0 - k1) ** 4)
    t6 = ((a - k0) ** 6 - (b - k0) ** 6) / (6 * (k0 - k1) ** 5)
    dt2 = (b - a) / (k1 - k0)
    dt3 = (b - a) * (a + b - 2 * k0) / ((k1 - k0) ** 2)
    dt4 = (
        (b - a)
        * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2)
        / ((k1 - k0) ** 3)
    )
    dt5 = (-((a - k0) ** 4) + (b - k0) ** 4) / ((k0 - k1) ** 4)
    dt6 = (-((a - k0) ** 5) + (b - k0) ** 5) / ((k1 - k0) ** 5)
    return (
        np.array([b - a, t2, t3, t4, t5, t6]),
        np.array([0.0, dt2, dt3, dt4, dt5, dt6]),
    )


def _cubic_integrals(lim_x0, lim_x1, knot_x0, knot_x1, lim_y0, lim_y1, knot_y0, knot_y1):
    """Cubic integral weights in both x and y for a single 2D cell."""
    tx, dtx = _cubic_one_integral(lim_x0, lim_x1, knot_x0, knot_x1)
    ty, dty = _cubic_one_integral(lim_y0, lim_y1, knot_y0, knot_y1)
    return tx, dtx, ty, dty


def _quintic_integrals(lim_x0, lim_x1, knot_x0, knot_x1, lim_y0, lim_y1, knot_y0, knot_y1):
    """Quintic integral weights in both x and y for a single 2D cell."""
    tx, dtx = _quintic_one_integral(lim_x0, lim_x1, knot_x0, knot_x1)
    ty, dty = _quintic_one_integral(lim_y0, lim_y1, knot_y0, knot_y1)
    return tx, dtx, ty, dty


def _integral_1_spline(func, lim_x, knots_x, lim_y, knots_y, coefs):
    """Integrate a 2D spline over one rectangular region."""
    xk0, xk1 = knots_x
    xa, xb = lim_x
    yk0, yk1 = knots_y
    ya, yb = lim_y
    tx, dtx, ty, dty = func(xa, xb, xk0, xk1, ya, yb, yk0, yk1)
    int_psi = tx.T @ coefs @ ty
    int_dx = dtx.T @ coefs @ ty
    int_dy = tx.T @ coefs @ dty
    return int_psi, int_dx, int_dy


def _integral_2_splines_x(func, lim_x, knots_x, lim_y, knots_y, coefs):
    """Integrate across two y-splines (cell straddles a y knot)."""
    xa, xb = lim_x
    xk0, xk1 = knots_x
    ya, yb = lim_y
    yk0, yc, yk1 = knots_y
    tx1, dtx1, ty1, dty1 = func(xa, xb, xk0, xk1, ya, yc, yk0, yc)
    tx2, dtx2, ty2, dty2 = func(xa, xb, xk0, xk1, yc, yb, yc, yk1)
    int_psi = (tx1.T @ coefs[:, :, 0] @ ty1) + (tx2.T @ coefs[:, :, 1] @ ty2)
    int_dx = (dtx1.T @ coefs[:, :, 0] @ ty1) + (dtx2.T @ coefs[:, :, 1] @ ty2)
    int_dy = (tx1.T @ coefs[:, :, 0] @ dty1) + (tx2.T @ coefs[:, :, 1] @ dty2)
    return int_psi, int_dx, int_dy


def _integral_2_splines_y(func, lim_x, knots_x, lim_y, knots_y, coefs):
    """Integrate across two x-splines (cell straddles an x knot)."""
    xa, xb = lim_x
    xk0, xc, xk1 = knots_x
    ya, yb = lim_y
    yk0, yk1 = knots_y
    tx1, dtx1, ty1, dty1 = func(xa, xc, xk0, xc, ya, yb, yk0, yk1)
    tx2, dtx2, ty2, dty2 = func(xc, xb, xc, xk1, ya, yb, yk0, yk1)
    int_psi = (tx1.T @ coefs[:, :, 0] @ ty1) + (tx2.T @ coefs[:, :, 1] @ ty2)
    int_dx = (dtx1.T @ coefs[:, :, 0] @ ty1) + (dtx2.T @ coefs[:, :, 1] @ ty2)
    int_dy = (tx1.T @ coefs[:, :, 0] @ dty1) + (tx2.T @ coefs[:, :, 1] @ dty2)
    return int_psi, int_dx, int_dy


def _integral_4_splines(func, lim_x, knots_x, lim_y, knots_y, coefs):
    """Integrate across four splines (interior cell straddles both x and y knots)."""
    xk0, xc, xk1 = knots_x
    xa, xb = lim_x
    yk0, yc, yk1 = knots_y
    ya, yb = lim_y
    tx1, dtx1, ty1, dty1 = func(xa, xc, xk0, xc, ya, yc, yk0, yc)
    tx2, dtx2, ty2, dty2 = func(xc, xb, xc, xk1, ya, yc, yc, yk1)
    tx3, dtx3, ty3, dty3 = func(xc, xb, xc, xk1, yc, yb, yc, yk1)
    tx4, dtx4, ty4, dty4 = func(xa, xc, xk0, xc, yc, yb, yk0, yc)
    int_psi = (
        (tx1.T @ coefs[:, :, 0, 0] @ ty1)
        + (tx2.T @ coefs[:, :, 1, 0] @ ty2)
        + (tx3.T @ coefs[:, :, 1, 1] @ ty3)
        + (tx4.T @ coefs[:, :, 0, 1] @ ty4)
    )
    int_dx = (
        (dtx1.T @ coefs[:, :, 0, 0] @ ty1)
        + (dtx2.T @ coefs[:, :, 1, 0] @ ty2)
        + (dtx3.T @ coefs[:, :, 1, 1] @ ty3)
        + (dtx4.T @ coefs[:, :, 0, 1] @ ty4)
    )
    int_dy = (
        (tx1.T @ coefs[:, :, 0, 0] @ dty1)
        + (tx2.T @ coefs[:, :, 1, 0] @ dty2)
        + (tx3.T @ coefs[:, :, 1, 1] @ dty3)
        + (tx4.T @ coefs[:, :, 0, 1] @ dty4)
    )
    return int_psi, int_dx, int_dy


########################################################################
# Spline Extension Types
########################################################################

cdef class CubicHermite:
    """Bi-cubic Hermite spline over a 2D grid of knots.

    Parameters
    ----------
    psi : array_like, shape (Nx, Ny)
        Function values at the knot points.
    knots_x : array_like, length Nx
        Knot coordinates in x (monotonic).
    knots_y : array_like, length Ny
        Knot coordinates in y (monotonic).
    """

    cdef double[:, :] psi
    cdef double[:] knots_x
    cdef double[:] knots_y
    cdef double[:, :, :, :] coefs    # shape (4, 4, Nx-1, Ny-1)

    def __init__(self, psi, knots_x, knots_y):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int Nx, Ny
        d_dx, d_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        _, d2_dxdy, _ = second_derivative(self.psi, self.knots_x, self.knots_y)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        psi_np = np.asarray(self.psi)
        Nx = knots_x.shape[0]
        Ny = knots_y.shape[0]
        delta_x = knots_x[1:Nx] - knots_x[0:Nx-1]
        delta_y = knots_y[1:Ny] - knots_y[0:Ny-1]
        delta = np.ones((4, 4, Nx - 1, Ny - 1))
        delta[2:] *= delta_x[(None), :, None]
        delta[:, 2:] *= delta_y[(None), :]
        control = np.zeros((4, 4, Nx - 1, Ny - 1))
        control[0:2, 0:2] = np.array(
            [[psi_np[0:Nx-1, 0:Ny-1], psi_np[0:Nx-1, 1:Ny]],
             [psi_np[1:Nx, 0:Ny-1],   psi_np[1:Nx, 1:Ny]]]
        )
        control[2:4, 0:2] = np.array(
            [[d_dx[0:Nx-1, 0:Ny-1], d_dx[0:Nx-1, 1:Ny]],
             [d_dx[1:Nx, 0:Ny-1],   d_dx[1:Nx, 1:Ny]]]
        )
        control[0:2, 2:4] = np.array(
            [[d_dy[0:Nx-1, 0:Ny-1], d_dy[0:Nx-1, 1:Ny]],
             [d_dy[1:Nx, 0:Ny-1],   d_dy[1:Nx, 1:Ny]]]
        )
        control[2:4, 2:4] = np.array(
            [[d2_dxdy[0:Nx-1, 0:Ny-1], d2_dxdy[0:Nx-1, 1:Ny]],
             [d2_dxdy[1:Nx, 0:Ny-1],   d2_dxdy[1:Nx, 1:Ny]]]
        )
        B = CUBIC_BASIS
        self.coefs = np.ascontiguousarray(
            np.einsum('ij,jk...,kl->il...', B, control * delta, B.T),
            dtype=np.float64,
        )

    def _find_zone(self, n, bins):
        cdef int Nb = bins.shape[0]
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == Nb - 1] = Nb - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny):
        """Interpolate values at all (nx[i], ny[j]) pairs."""
        cdef int ii, jj
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        idx_x = self._find_zone(nx, knots_x)
        tx_n = (nx - knots_x[idx_x]) / (knots_x[idx_x + 1] - knots_x[idx_x])
        tx = np.array([[1] * len(nx), tx_n, tx_n**2, tx_n**3])
        idx_y = self._find_zone(ny, knots_y)
        ty_n = (ny - knots_y[idx_y]) / (knots_y[idx_y + 1] - knots_y[idx_y])
        ty = np.array([[1] * len(ny), ty_n, ty_n**2, ty_n**3])
        coefs = np.asarray(self.coefs)
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        for _ii in np.unique(np.sort(idx_x)):
            ii = _ii
            loc_x = np.argwhere(idx_x == ii).flatten()
            ix = loc_x[:, None]
            for _jj in np.unique(np.sort(idx_y)):
                jj = _jj
                loc_y = np.argwhere(idx_y == jj).flatten()
                iy = loc_y[None, :]
                splines_psi[ix, iy] = tx[:, loc_x].T @ coefs[:, :, ii, jj] @ ty[:, loc_y]
        return splines_psi

    def integrate_edges(self):
        """Return (int_psi, int_dx, int_dy) integrated over each knot-defined cell."""
        cdef int Nx, Ny
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        Nx = knots_x.shape[0] - 1
        Ny = knots_y.shape[0] - 1
        delta_x = knots_x[1:Nx+1] - knots_x[0:Nx]
        delta_y = knots_y[1:Ny+1] - knots_y[0:Ny]
        tx = np.array([delta_x, 0.5 * delta_x, delta_x / 3.0, 0.25 * delta_x])
        dtx = np.ones((4, Nx))
        dtx[0] = 0.0
        ty = np.array([delta_y, 0.5 * delta_y, delta_y / 3.0, 0.25 * delta_y])
        dty = np.ones((4, Ny))
        dty[0] = 0.0
        coefs = np.asarray(self.coefs)
        int_psi = np.einsum("xi,ijxy,jy->xy", tx.T, coefs, ty)
        int_dx = np.einsum("xi,ijxy,jy->xy", dtx.T, coefs, ty)
        int_dy = np.einsum("xi,ijxy,jy->xy", tx.T, coefs, dty)
        return int_psi, int_dx, int_dy

    def integrate_centers(self, limits_x, limits_y):
        """Integrate spline over cell-centered limits.

        limits_x/y should be cell boundaries (length Nx+1, Ny+1).
        """
        limits_x = np.asarray(limits_x, dtype=np.float64)
        limits_y = np.asarray(limits_y, dtype=np.float64)
        coefs = np.asarray(self.coefs)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        Nx = int(knots_x.shape[0])
        Ny = int(knots_y.shape[0])
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        Lx = int(limits_x.shape[0]) - 1
        Ly = int(limits_y.shape[0]) - 1
        for ii in range(Lx):
            xa = limits_x[ii]
            xb = limits_x[ii + 1]
            for jj in range(Ly):
                ya = limits_y[jj]
                yb = limits_y[jj + 1]
                if ((jj == 0) or (jj == Ny - 1)) and ((ii == 0) or (ii == Nx - 1)):
                    # Corner: 1 spline
                    lx = [0, 1] if ii == 0 else [Nx - 1, Nx]
                    ly = [0, 1] if jj == 0 else [Ny - 1, Ny]
                    ix = 0 if ii == 0 else Nx - 2
                    jy = 0 if jj == 0 else Ny - 2
                    _psi, _dx, _dy = _integral_1_spline(
                        _cubic_integrals,
                        limits_x[lx], knots_x[[0, 1] if ii == 0 else [Nx-2, Nx-1]],
                        limits_y[ly], knots_y[[0, 1] if jj == 0 else [Ny-2, Ny-1]],
                        coefs[:, :, ix, jy],
                    )
                elif (ii == 0) or (ii == Nx - 1):
                    # x-edge: 2 y-splines
                    lx = [0, 1] if ii == 0 else [Nx - 1, Nx]
                    ix = 0 if ii == 0 else Nx - 2
                    _psi, _dx, _dy = _integral_2_splines_x(
                        _cubic_integrals,
                        limits_x[lx],
                        knots_x[[0, 1] if ii == 0 else [Nx-2, Nx-1]],
                        (ya, yb),
                        knots_y[jj - 1:jj + 2],
                        coefs[:, :, ix, jj - 1:jj + 1],
                    )
                elif (jj == 0) or (jj == Ny - 1):
                    # y-edge: 2 x-splines
                    ly = [0, 1] if jj == 0 else [Ny - 1, Ny]
                    jy = 0 if jj == 0 else Ny - 2
                    _psi, _dx, _dy = _integral_2_splines_y(
                        _cubic_integrals,
                        (xa, xb),
                        knots_x[ii - 1:ii + 2],
                        limits_y[ly],
                        knots_y[[0, 1] if jj == 0 else [Ny-2, Ny-1]],
                        coefs[:, :, ii - 1:ii + 1, jy],
                    )
                else:
                    # Interior: 4 splines
                    _psi, _dx, _dy = _integral_4_splines(
                        _cubic_integrals,
                        (xa, xb), knots_x[ii - 1:ii + 2],
                        (ya, yb), knots_y[jj - 1:jj + 2],
                        coefs[:, :, ii - 1:ii + 1, jj - 1:jj + 1],
                    )
                int_psi[ii, jj] = _psi
                int_dx[ii, jj] = _dx
                int_dy[ii, jj] = _dy
        return int_psi, int_dx, int_dy


cdef class QuinticHermite:
    """Bi-quintic Hermite spline over a 2D grid of knots.

    Parameters
    ----------
    psi : array_like, shape (Nx, Ny)
        Function values at the knot points.
    knots_x : array_like, length Nx
        Knot coordinates in x (monotonic).
    knots_y : array_like, length Ny
        Knot coordinates in y (monotonic).
    """

    cdef double[:, :] psi
    cdef double[:] knots_x
    cdef double[:] knots_y
    cdef double[:, :, :, :] coefs    # shape (6, 6, Nx-1, Ny-1)

    def __init__(self, psi, knots_x, knots_y):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int Nx, Ny
        d_dx, d_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        d2_dx2, d2_dxdy, d2_dy2 = second_derivative(
            self.psi, self.knots_x, self.knots_y
        )
        d3_dx2dy, d3_dxdy2, d4_dx2dy2 = higher_order_derivative(
            self.psi, self.knots_x, self.knots_y
        )
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        psi_np = np.asarray(self.psi)
        Nx = knots_x.shape[0]
        Ny = knots_y.shape[0]
        delta_x = knots_x[1:Nx] - knots_x[0:Nx-1]
        delta_y = knots_y[1:Ny] - knots_y[0:Ny-1]
        delta = np.ones((6, 6, Nx - 1, Ny - 1))
        delta[2:] *= delta_x[(None), :, None]
        delta[4:] *= delta_x[(None), :, None]
        delta[:, 2:] *= delta_y[(None), :]
        delta[:, 4:] *= delta_y[(None), :]
        control = np.zeros((6, 6, Nx - 1, Ny - 1))
        control[0:2, 0:2] = np.array(
            [[psi_np[0:Nx-1, 0:Ny-1], psi_np[0:Nx-1, 1:Ny]],
             [psi_np[1:Nx, 0:Ny-1],   psi_np[1:Nx, 1:Ny]]]
        )
        control[0:2, 2:4] = np.array(
            [[d_dy[0:Nx-1, 0:Ny-1], d_dy[0:Nx-1, 1:Ny]],
             [d_dy[1:Nx, 0:Ny-1],   d_dy[1:Nx, 1:Ny]]]
        )
        control[0:2, 4:6] = np.array(
            [[d2_dy2[0:Nx-1, 0:Ny-1], d2_dy2[0:Nx-1, 1:Ny]],
             [d2_dy2[1:Nx, 0:Ny-1],   d2_dy2[1:Nx, 1:Ny]]]
        )
        control[2:4, 0:2] = np.array(
            [[d_dx[0:Nx-1, 0:Ny-1], d_dx[0:Nx-1, 1:Ny]],
             [d_dx[1:Nx, 0:Ny-1],   d_dx[1:Nx, 1:Ny]]]
        )
        control[2:4, 2:4] = np.array(
            [[d2_dxdy[0:Nx-1, 0:Ny-1], d2_dxdy[0:Nx-1, 1:Ny]],
             [d2_dxdy[1:Nx, 0:Ny-1],   d2_dxdy[1:Nx, 1:Ny]]]
        )
        control[2:4, 4:6] = np.array(
            [[d3_dxdy2[0:Nx-1, 0:Ny-1], d3_dxdy2[0:Nx-1, 1:Ny]],
             [d3_dxdy2[1:Nx, 0:Ny-1],   d3_dxdy2[1:Nx, 1:Ny]]]
        )
        control[4:6, 0:2] = np.array(
            [[d2_dx2[0:Nx-1, 0:Ny-1], d2_dx2[0:Nx-1, 1:Ny]],
             [d2_dx2[1:Nx, 0:Ny-1],   d2_dx2[1:Nx, 1:Ny]]]
        )
        control[4:6, 2:4] = np.array(
            [[d3_dx2dy[0:Nx-1, 0:Ny-1], d3_dx2dy[0:Nx-1, 1:Ny]],
             [d3_dx2dy[1:Nx, 0:Ny-1],   d3_dx2dy[1:Nx, 1:Ny]]]
        )
        control[4:6, 4:6] = np.array(
            [[d4_dx2dy2[0:Nx-1, 0:Ny-1], d4_dx2dy2[0:Nx-1, 1:Ny]],
             [d4_dx2dy2[1:Nx, 0:Ny-1],   d4_dx2dy2[1:Nx, 1:Ny]]]
        )
        B = QUINTIC_BASIS
        self.coefs = np.ascontiguousarray(
            np.einsum('ij,jk...,kl->il...', B, control * delta, B.T),
            dtype=np.float64,
        )

    def _find_zone(self, n, bins):
        cdef int Nb = bins.shape[0]
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == Nb - 1] = Nb - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny):
        """Evaluate the quintic spline at all (nx[i], ny[j]) pairs."""
        cdef int ii, jj
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        idx_x = self._find_zone(nx, knots_x)
        tx_n = (nx - knots_x[idx_x]) / (knots_x[idx_x + 1] - knots_x[idx_x])
        tx = np.array([[1] * len(nx), tx_n, tx_n**2, tx_n**3, tx_n**4, tx_n**5])
        idx_y = self._find_zone(ny, knots_y)
        ty_n = (ny - knots_y[idx_y]) / (knots_y[idx_y + 1] - knots_y[idx_y])
        ty = np.array([[1] * len(ny), ty_n, ty_n**2, ty_n**3, ty_n**4, ty_n**5])
        coefs = np.asarray(self.coefs)
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        for _ii in np.unique(np.sort(idx_x)):
            ii = _ii
            loc_x = np.argwhere(idx_x == ii).flatten()
            ix = loc_x[:, None]
            for _jj in np.unique(np.sort(idx_y)):
                jj = _jj
                loc_y = np.argwhere(idx_y == jj).flatten()
                iy = loc_y[None, :]
                splines_psi[ix, iy] = tx[:, loc_x].T @ coefs[:, :, ii, jj] @ ty[:, loc_y]
        return splines_psi

    def integrate_edges(self):
        """Return (int_psi, int_dx, int_dy) integrated over each knot-defined cell."""
        cdef int Nx, Ny
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        Nx = knots_x.shape[0] - 1
        Ny = knots_y.shape[0] - 1
        delta_x = knots_x[1:Nx+1] - knots_x[0:Nx]
        delta_y = knots_y[1:Ny+1] - knots_y[0:Ny]
        tx = np.array(
            [delta_x, 0.5 * delta_x, delta_x / 3.0, 0.25 * delta_x,
             0.2 * delta_x, delta_x / 6.0]
        )
        dtx = np.ones((6, Nx))
        dtx[0] = 0.0
        ty = np.array(
            [delta_y, 0.5 * delta_y, delta_y / 3.0, 0.25 * delta_y,
             0.2 * delta_y, delta_y / 6.0]
        )
        dty = np.ones((6, Ny))
        dty[0] = 0.0
        coefs = np.asarray(self.coefs)
        int_psi = np.einsum("xi,ijxy,jy->xy", tx.T, coefs, ty)
        int_dx = np.einsum("xi,ijxy,jy->xy", dtx.T, coefs, ty)
        int_dy = np.einsum("xi,ijxy,jy->xy", tx.T, coefs, dty)
        return int_psi, int_dx, int_dy

    def integrate_centers(self, limits_x, limits_y):
        """Integrate quintic spline over cell-centered limits.

        limits_x/y should be cell boundaries (length Nx+1, Ny+1).
        """
        limits_x = np.asarray(limits_x, dtype=np.float64)
        limits_y = np.asarray(limits_y, dtype=np.float64)
        coefs = np.asarray(self.coefs)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        Nx = int(knots_x.shape[0])
        Ny = int(knots_y.shape[0])
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        Lx = int(limits_x.shape[0]) - 1
        Ly = int(limits_y.shape[0]) - 1
        for ii in range(Lx):
            xa = limits_x[ii]
            xb = limits_x[ii + 1]
            for jj in range(Ly):
                ya = limits_y[jj]
                yb = limits_y[jj + 1]
                if ((jj == 0) or (jj == Ny - 1)) and ((ii == 0) or (ii == Nx - 1)):
                    # Corner: 1 spline
                    lx = [0, 1] if ii == 0 else [Nx - 1, Nx]
                    ly = [0, 1] if jj == 0 else [Ny - 1, Ny]
                    ix = 0 if ii == 0 else Nx - 2
                    jy = 0 if jj == 0 else Ny - 2
                    _psi, _dx, _dy = _integral_1_spline(
                        _quintic_integrals,
                        limits_x[lx], knots_x[[0, 1] if ii == 0 else [Nx-2, Nx-1]],
                        limits_y[ly], knots_y[[0, 1] if jj == 0 else [Ny-2, Ny-1]],
                        coefs[:, :, ix, jy],
                    )
                elif (ii == 0) or (ii == Nx - 1):
                    # x-edge: 2 y-splines
                    lx = [0, 1] if ii == 0 else [Nx - 1, Nx]
                    ix = 0 if ii == 0 else Nx - 2
                    _psi, _dx, _dy = _integral_2_splines_x(
                        _quintic_integrals,
                        limits_x[lx],
                        knots_x[[0, 1] if ii == 0 else [Nx-2, Nx-1]],
                        (ya, yb),
                        knots_y[jj - 1:jj + 2],
                        coefs[:, :, ix, jj - 1:jj + 1],
                    )
                elif (jj == 0) or (jj == Ny - 1):
                    # y-edge: 2 x-splines
                    ly = [0, 1] if jj == 0 else [Ny - 1, Ny]
                    jy = 0 if jj == 0 else Ny - 2
                    _psi, _dx, _dy = _integral_2_splines_y(
                        _quintic_integrals,
                        (xa, xb),
                        knots_x[ii - 1:ii + 2],
                        limits_y[ly],
                        knots_y[[0, 1] if jj == 0 else [Ny-2, Ny-1]],
                        coefs[:, :, ii - 1:ii + 1, jy],
                    )
                else:
                    # Interior: 4 splines
                    _psi, _dx, _dy = _integral_4_splines(
                        _quintic_integrals,
                        (xa, xb), knots_x[ii - 1:ii + 2],
                        (ya, yb), knots_y[jj - 1:jj + 2],
                        coefs[:, :, ii - 1:ii + 1, jj - 1:jj + 1],
                    )
                int_psi[ii, jj] = _psi
                int_dx[ii, jj] = _dx
                int_dy[ii, jj] = _dy
        return int_psi, int_dx, int_dy


########################################################################
# Block and Convenience Wrappers
########################################################################

class BlockInterpolation:
    """Apply a spline class on disjoint (x, y) blocks.

    Parameters
    ----------
    Splines : class
        Either CubicHermite or QuinticHermite.
    psi, knots_x, knots_y : array_like
        Global arrays.
    medium_map : array_like
        2D per-cell medium mapping (used when x/y_splits are empty).
    x_splits, y_splits : array_like
        Explicit block boundary indices.
    """

    def __init__(self, Splines, psi, knots_x, knots_y, medium_map, x_splits, y_splits):
        self.Splines = Splines
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        if np.asarray(x_splits).shape[0] > 0:
            self.x_splits = np.asarray(x_splits).copy()
            self.y_splits = np.asarray(y_splits).copy()
        else:
            self.x_splits, self.y_splits = pytools._to_block(np.asarray(medium_map))
        self._generate_coefs()

    def _generate_coefs(self):
        cdef int ii, jj, n_xblocks, n_yblocks
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        self.splines = []
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            col = []
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]
                y2 = self.y_splits[jj + 1]
                col.append(
                    self.Splines(
                        self.psi[x1:x2, y1:y2],
                        self.knots_x[x1:x2],
                        self.knots_y[y1:y2],
                    )
                )
            self.splines.append(col)

    def interpolate(self, nx, ny):
        """Interpolate values using the block-wise splines."""
        cdef int ii, jj, n_xblocks, n_yblocks
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            if ii == 0:
                idx_x = np.argwhere(nx < self.knots_x[x2]).flatten()
            elif ii == n_xblocks - 1:
                idx_x = np.argwhere(nx >= self.knots_x[x1]).flatten()
            else:
                idx_x = np.argwhere(
                    (nx >= self.knots_x[x1]) & (nx < self.knots_x[x2])
                ).flatten()
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]
                y2 = self.y_splits[jj + 1]
                if jj == 0:
                    idx_y = np.argwhere(ny < self.knots_y[y2]).flatten()
                elif jj == n_yblocks - 1:
                    idx_y = np.argwhere(ny >= self.knots_y[y1]).flatten()
                else:
                    idx_y = np.argwhere(
                        (ny >= self.knots_y[y1]) & (ny < self.knots_y[y2])
                    ).flatten()
                mesh_x, mesh_y = np.meshgrid(idx_x, idx_y, indexing="ij")
                splines_psi[mesh_x, mesh_y] = self.splines[ii][jj].interpolate(
                    nx[idx_x], ny[idx_y]
                )
        return splines_psi

    def integrate_edges(self):
        """Compute (int_psi, int_dx, int_dy) for all edges across blocks."""
        cdef int ii, jj, n_xblocks, n_yblocks
        Nx = self.knots_x.shape[0] - 1
        Ny = self.knots_y.shape[0] - 1
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]
                y2 = self.y_splits[jj + 1]
                approx = self.Splines(
                    self.psi[x1:x2+1, y1:y2+1],
                    self.knots_x[x1:x2+1],
                    self.knots_y[y1:y2+1],
                )
                b_psi, b_dx, b_dy = approx.integrate_edges()
                int_psi[x1:x2, y1:y2] = b_psi
                int_dx[x1:x2, y1:y2] = b_dx
                int_dy[x1:x2, y1:y2] = b_dy
        return int_psi, int_dx, int_dy

    def integrate_centers(self, limits_x, limits_y):
        """Compute (int_psi, int_dx, int_dy) for cell-centered limits across blocks."""
        cdef int ii, jj, n_xblocks, n_yblocks
        limits_x = np.asarray(limits_x)
        limits_y = np.asarray(limits_y)
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]
            x2 = self.x_splits[ii + 1]
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]
                y2 = self.y_splits[jj + 1]
                approx = self.Splines(
                    self.psi[x1:x2, y1:y2],
                    self.knots_x[x1:x2],
                    self.knots_y[y1:y2],
                )
                b_psi, b_dx, b_dy = approx.integrate_centers(
                    limits_x[x1:x2+1], limits_y[y1:y2+1]
                )
                int_psi[x1:x2, y1:y2] = b_psi
                int_dx[x1:x2, y1:y2] = b_dx
                int_dy[x1:x2, y1:y2] = b_dy
        return int_psi, int_dx, int_dy


class Interpolation:
    """Convenience wrapper that selects the appropriate spline/block combination.

    Parameters
    ----------
    psi, knots_x, knots_y : array_like
        Global arrays.
    medium_map : array_like
        Passed to BlockInterpolation when block is True.
    x_splits, y_splits : array_like
        Explicit block boundaries.
    block : bool
        If True, build a block-wise interpolator.
    quintic : bool
        If True, use the quintic basis; otherwise cubic.
    """

    def __init__(
        self, psi, knots_x, knots_y, medium_map, x_splits, y_splits,
        block=True, quintic=True,
    ):
        if block and quintic:
            self.instance = BlockInterpolation(
                QuinticHermite, psi, knots_x, knots_y, medium_map, x_splits, y_splits
            )
        elif block:
            self.instance = BlockInterpolation(
                CubicHermite, psi, knots_x, knots_y, medium_map, x_splits, y_splits
            )
        elif quintic:
            self.instance = QuinticHermite(psi, knots_x, knots_y)
        else:
            self.instance = CubicHermite(psi, knots_x, knots_y)

    def interpolate(self, nx, ny):
        return self.instance.interpolate(nx, ny)

    def integrate_edges(self):
        return self.instance.integrate_edges()

    def integrate_centers(self, limits_x, limits_y):
        return self.instance.integrate_centers(limits_x, limits_y)
