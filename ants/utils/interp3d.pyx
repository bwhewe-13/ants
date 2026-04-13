########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Three-Dimensional Interpolation Utilities
#
# Provides derivative approximations and tri-Hermite spline interpolation.
# Extends the 2D tensor-product Hermite approach to 3D using a full set
# of mixed partial derivatives at each knot.
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

from ants.utils import interp1d

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
# Axis-wise derivative helper
########################################################################

def _apply_deriv(f_3d, coords, int axis, int order):
    """Apply a 1D Hermite derivative along one axis of a 3D array.

    Parameters
    ----------
    f_3d : numpy.ndarray, shape (Nx, Ny, Nz)
        Input field.
    coords : array_like
        Knot coordinates along the target axis.
    axis : int
        Axis along which to differentiate (0, 1, or 2).
    order : int
        1 for first derivative, 2 for second derivative.

    Returns
    -------
    numpy.ndarray of the same shape as f_3d.
    """
    coords = np.asarray(coords, dtype=np.float64)
    f = np.asarray(f_3d)
    Nx, Ny, Nz = f.shape
    result = np.zeros_like(f)
    fn = interp1d.first_derivative if order == 1 else interp1d.second_derivative
    if axis == 0:
        for jj in range(Ny):
            for kk in range(Nz):
                result[:, jj, kk] = fn(f[:, jj, kk], coords)
    elif axis == 1:
        for ii in range(Nx):
            for kk in range(Nz):
                result[ii, :, kk] = fn(f[ii, :, kk], coords)
    else:
        for ii in range(Nx):
            for jj in range(Ny):
                result[ii, jj, :] = fn(f[ii, jj, :], coords)
    return result


########################################################################
# Derivative Approximations
########################################################################

cpdef first_derivative(double[:, :, :] psi, double[:] x, double[:] y, double[:] z):
    """Estimate first partial derivatives at each knot on a 3D grid.

    Parameters
    ----------
    psi : double[:,:,:]
        Function values; shape (Nx, Ny, Nz).
    x, y, z : double[:]
        Knot coordinates (lengths Nx, Ny, Nz respectively).

    Returns
    -------
    (d_dx, d_dy, d_dz) : tuple of numpy.ndarray, each shape (Nx, Ny, Nz)
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    d_dx = _apply_deriv(psi_np, x_np, 0, 1)
    d_dy = _apply_deriv(psi_np, y_np, 1, 1)
    d_dz = _apply_deriv(psi_np, z_np, 2, 1)
    return d_dx, d_dy, d_dz


cpdef second_derivative(double[:, :, :] psi, double[:] x, double[:] y, double[:] z):
    """Estimate second partial derivatives at each knot on a 3D grid.

    Parameters
    ----------
    psi : double[:,:,:]
        Function values; shape (Nx, Ny, Nz).
    x, y, z : double[:]
        Knot coordinates.

    Returns
    -------
    (d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz) : tuple of arrays
        Each has shape (Nx, Ny, Nz).
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    fx = _apply_deriv(psi_np, x_np, 0, 1)
    fy = _apply_deriv(psi_np, y_np, 1, 1)
    fz = _apply_deriv(psi_np, z_np, 2, 1)
    d2_dx2 = _apply_deriv(psi_np, x_np, 0, 2)
    d2_dy2 = _apply_deriv(psi_np, y_np, 1, 2)
    d2_dz2 = _apply_deriv(psi_np, z_np, 2, 2)
    d2_dxdy = _apply_deriv(fx, y_np, 1, 1)
    d2_dxdz = _apply_deriv(fx, z_np, 2, 1)
    d2_dydz = _apply_deriv(fy, z_np, 2, 1)
    return d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz


cpdef higher_order_derivative(double[:, :, :] psi, double[:] x, double[:] y, double[:] z):
    """Estimate higher-order mixed partial derivatives needed for tri-quintic splines.

    Computes all 20 additional derivative types (beyond the 7 returned by
    second_derivative) required to fill the 6x6x6 tri-quintic control tensor.

    Parameters
    ----------
    psi : double[:,:,:]
        Function values; shape (Nx, Ny, Nz).
    x, y, z : double[:]
        Knot coordinates.

    Returns
    -------
    Tuple of 20 arrays, each shape (Nx, Ny, Nz):
    (fxxy, fxxz, fxyy, fxzz, fyyz, fyzz,
     fxyz,
     fxxyy, fxxzz, fyyzz,
     fxxyz, fxyyz, fxyzz,
     fxxyyz, fxxyzz, fxyyzz,
     fxxyyzz)
    plus the first-derivative pair (fxx, fyy, fzz) not redundantly returned.
    """
    psi_np = np.asarray(psi)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    # Level-1 derivatives
    fx = _apply_deriv(psi_np, x_np, 0, 1)
    fy = _apply_deriv(psi_np, y_np, 1, 1)
    fz = _apply_deriv(psi_np, z_np, 2, 1)
    # Level-2 second-order pure
    fxx = _apply_deriv(psi_np, x_np, 0, 2)
    fyy = _apply_deriv(psi_np, y_np, 1, 2)
    fzz = _apply_deriv(psi_np, z_np, 2, 2)
    # Level-2 mixed first
    fxy = _apply_deriv(fx, y_np, 1, 1)
    fxz = _apply_deriv(fx, z_np, 2, 1)
    fyz = _apply_deriv(fy, z_np, 2, 1)
    # Level-3
    fxxy = _apply_deriv(fxx, y_np, 1, 1)
    fxxz = _apply_deriv(fxx, z_np, 2, 1)
    fxyy = _apply_deriv(fyy, x_np, 0, 1)
    fxzz = _apply_deriv(fzz, x_np, 0, 1)
    fyyz = _apply_deriv(fyy, z_np, 2, 1)
    fyzz = _apply_deriv(fzz, y_np, 1, 1)
    fxyz = _apply_deriv(fxy, z_np, 2, 1)
    # Level-4
    fxxyy = _apply_deriv(fxxy, y_np, 1, 1)
    fxxzz = _apply_deriv(fxxz, z_np, 2, 1)
    fyyzz = _apply_deriv(fyyz, z_np, 2, 1)
    fxxyz = _apply_deriv(fxxy, z_np, 2, 1)
    fxyyz = _apply_deriv(fxyy, z_np, 2, 1)
    fxyzz = _apply_deriv(fxyz, z_np, 2, 1)
    # Level-5
    fxxyyz = _apply_deriv(fxxyy, z_np, 2, 1)
    fxxyzz = _apply_deriv(fxxyz, z_np, 2, 1)
    fxyyzz = _apply_deriv(fxyyz, z_np, 2, 1)
    # Level-6
    fxxyyzz = _apply_deriv(fxxyyz, z_np, 2, 1)
    return (
        fxxy, fxxz, fxyy, fxzz, fyyz, fyzz,
        fxyz,
        fxxyy, fxxzz, fyyzz,
        fxxyz, fxyyz, fxyzz,
        fxxyyz, fxxyzz, fxyyzz,
        fxxyyzz,
    )


########################################################################
# Integral Weight Helpers
########################################################################

cdef _cubic_one_integral(double a, double b, double k0, double k1):
    """1D cubic Hermite integral weight vector for [a, b] with knots [k0, k1]."""
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
    """1D quintic Hermite integral weight vector for [a, b] with knots [k0, k1]."""
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


def _cell_integral(tx, ty, tz, dtx, dty, dtz, coefs):
    """Compute psi and gradient integrals for one spline cell.

    Parameters
    ----------
    tx, ty, tz : array_like, shape (P,)
        Integral weight vectors for psi along each axis.
    dtx, dty, dtz : array_like, shape (P,)
        Integral weight vectors for dpsi/dk along each axis.
    coefs : array_like, shape (P, P, P)
        Coefficient tensor for this cell.

    Returns
    -------
    (int_psi, int_dx, int_dy, int_dz) : floats
    """
    int_psi = np.einsum('i,j,k,ijk->', tx, ty, tz, coefs)
    int_dx  = np.einsum('i,j,k,ijk->', dtx, ty, tz, coefs)
    int_dy  = np.einsum('i,j,k,ijk->', tx, dty, tz, coefs)
    int_dz  = np.einsum('i,j,k,ijk->', tx, ty, dtz, coefs)
    return int_psi, int_dx, int_dy, int_dz


########################################################################
# Spline Extension Types
########################################################################

cdef class CubicHermite:
    """Tri-cubic Hermite spline over a 3D grid of knots.

    Uses function values and first partial derivatives (plus cross-derivatives)
    at each knot point to build a degree-3 polynomial on each cell.

    Parameters
    ----------
    psi : array_like, shape (Nx, Ny, Nz)
        Function values at the knot points.
    knots_x : array_like, length Nx
        Knot coordinates in x (monotonic).
    knots_y : array_like, length Ny
        Knot coordinates in y (monotonic).
    knots_z : array_like, length Nz
        Knot coordinates in z (monotonic).
    """

    cdef double[:, :, :] psi
    cdef double[:] knots_x
    cdef double[:] knots_y
    cdef double[:] knots_z
    cdef double[:, :, :, :, :, :] coefs    # shape (4, 4, 4, Nx-1, Ny-1, Nz-1)

    def __init__(self, psi, knots_x, knots_y, knots_z):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        self.knots_z = np.asarray(knots_z, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int Nx, Ny, Nz
        psi_np = np.asarray(self.psi)
        knots_x = np.asarray(self.knots_x)
        knots_y = np.asarray(self.knots_y)
        knots_z = np.asarray(self.knots_z)
        Nx = knots_x.shape[0]
        Ny = knots_y.shape[0]
        Nz = knots_z.shape[0]
        d_dx, d_dy, d_dz = first_derivative(self.psi, self.knots_x, self.knots_y, self.knots_z)
        _, _, _, d2_dxdy, d2_dxdz, d2_dydz = second_derivative(
            self.psi, self.knots_x, self.knots_y, self.knots_z
        )
        # Tri-mixed derivative fxyz
        fx = _apply_deriv(psi_np, knots_x, 0, 1)
        fxy = _apply_deriv(fx, knots_y, 1, 1)
        d3_dxdydz = _apply_deriv(fxy, knots_z, 2, 1)
        # Delta scaling tensor: shape (4, 4, 4, Nx-1, Ny-1, Nz-1)
        delta_x = knots_x[1:Nx] - knots_x[0:Nx-1]
        delta_y = knots_y[1:Ny] - knots_y[0:Ny-1]
        delta_z = knots_z[1:Nz] - knots_z[0:Nz-1]
        delta = np.ones((4, 4, 4, Nx - 1, Ny - 1, Nz - 1))
        delta[2:] *= delta_x[:, None, None]
        delta[:, 2:] *= delta_y[:, None]
        delta[:, :, 2:] *= delta_z
        # Control tensor: index [r, c, s] corresponds to
        #   r in {0,1}: f at x0/x1;  r in {2,3}: fx at x0/x1
        #   c in {0,1}: at y0/y1;    c in {2,3}: fy at y0/y1
        #   s in {0,1}: at z0/z1;    s in {2,3}: fz at z0/z1
        control = np.zeros((4, 4, 4, Nx - 1, Ny - 1, Nz - 1))
        # f at corners
        control[0, 0, 0] = psi_np[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[1, 0, 0] = psi_np[1:Nx,   0:Ny-1, 0:Nz-1]
        control[0, 1, 0] = psi_np[0:Nx-1, 1:Ny,   0:Nz-1]
        control[1, 1, 0] = psi_np[1:Nx,   1:Ny,   0:Nz-1]
        control[0, 0, 1] = psi_np[0:Nx-1, 0:Ny-1, 1:Nz]
        control[1, 0, 1] = psi_np[1:Nx,   0:Ny-1, 1:Nz]
        control[0, 1, 1] = psi_np[0:Nx-1, 1:Ny,   1:Nz]
        control[1, 1, 1] = psi_np[1:Nx,   1:Ny,   1:Nz]
        # fx at corners
        control[2, 0, 0] = d_dx[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[3, 0, 0] = d_dx[1:Nx,   0:Ny-1, 0:Nz-1]
        control[2, 1, 0] = d_dx[0:Nx-1, 1:Ny,   0:Nz-1]
        control[3, 1, 0] = d_dx[1:Nx,   1:Ny,   0:Nz-1]
        control[2, 0, 1] = d_dx[0:Nx-1, 0:Ny-1, 1:Nz]
        control[3, 0, 1] = d_dx[1:Nx,   0:Ny-1, 1:Nz]
        control[2, 1, 1] = d_dx[0:Nx-1, 1:Ny,   1:Nz]
        control[3, 1, 1] = d_dx[1:Nx,   1:Ny,   1:Nz]
        # fy at corners
        control[0, 2, 0] = d_dy[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[1, 2, 0] = d_dy[1:Nx,   0:Ny-1, 0:Nz-1]
        control[0, 3, 0] = d_dy[0:Nx-1, 1:Ny,   0:Nz-1]
        control[1, 3, 0] = d_dy[1:Nx,   1:Ny,   0:Nz-1]
        control[0, 2, 1] = d_dy[0:Nx-1, 0:Ny-1, 1:Nz]
        control[1, 2, 1] = d_dy[1:Nx,   0:Ny-1, 1:Nz]
        control[0, 3, 1] = d_dy[0:Nx-1, 1:Ny,   1:Nz]
        control[1, 3, 1] = d_dy[1:Nx,   1:Ny,   1:Nz]
        # fz at corners
        control[0, 0, 2] = d_dz[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[1, 0, 2] = d_dz[1:Nx,   0:Ny-1, 0:Nz-1]
        control[0, 1, 2] = d_dz[0:Nx-1, 1:Ny,   0:Nz-1]
        control[1, 1, 2] = d_dz[1:Nx,   1:Ny,   0:Nz-1]
        control[0, 0, 3] = d_dz[0:Nx-1, 0:Ny-1, 1:Nz]
        control[1, 0, 3] = d_dz[1:Nx,   0:Ny-1, 1:Nz]
        control[0, 1, 3] = d_dz[0:Nx-1, 1:Ny,   1:Nz]
        control[1, 1, 3] = d_dz[1:Nx,   1:Ny,   1:Nz]
        # fxy at corners
        control[2, 2, 0] = d2_dxdy[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[3, 2, 0] = d2_dxdy[1:Nx,   0:Ny-1, 0:Nz-1]
        control[2, 3, 0] = d2_dxdy[0:Nx-1, 1:Ny,   0:Nz-1]
        control[3, 3, 0] = d2_dxdy[1:Nx,   1:Ny,   0:Nz-1]
        control[2, 2, 1] = d2_dxdy[0:Nx-1, 0:Ny-1, 1:Nz]
        control[3, 2, 1] = d2_dxdy[1:Nx,   0:Ny-1, 1:Nz]
        control[2, 3, 1] = d2_dxdy[0:Nx-1, 1:Ny,   1:Nz]
        control[3, 3, 1] = d2_dxdy[1:Nx,   1:Ny,   1:Nz]
        # fxz at corners
        control[2, 0, 2] = d2_dxdz[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[3, 0, 2] = d2_dxdz[1:Nx,   0:Ny-1, 0:Nz-1]
        control[2, 1, 2] = d2_dxdz[0:Nx-1, 1:Ny,   0:Nz-1]
        control[3, 1, 2] = d2_dxdz[1:Nx,   1:Ny,   0:Nz-1]
        control[2, 0, 3] = d2_dxdz[0:Nx-1, 0:Ny-1, 1:Nz]
        control[3, 0, 3] = d2_dxdz[1:Nx,   0:Ny-1, 1:Nz]
        control[2, 1, 3] = d2_dxdz[0:Nx-1, 1:Ny,   1:Nz]
        control[3, 1, 3] = d2_dxdz[1:Nx,   1:Ny,   1:Nz]
        # fyz at corners
        control[0, 2, 2] = d2_dydz[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[1, 2, 2] = d2_dydz[1:Nx,   0:Ny-1, 0:Nz-1]
        control[0, 3, 2] = d2_dydz[0:Nx-1, 1:Ny,   0:Nz-1]
        control[1, 3, 2] = d2_dydz[1:Nx,   1:Ny,   0:Nz-1]
        control[0, 2, 3] = d2_dydz[0:Nx-1, 0:Ny-1, 1:Nz]
        control[1, 2, 3] = d2_dydz[1:Nx,   0:Ny-1, 1:Nz]
        control[0, 3, 3] = d2_dydz[0:Nx-1, 1:Ny,   1:Nz]
        control[1, 3, 3] = d2_dydz[1:Nx,   1:Ny,   1:Nz]
        # fxyz at corners
        control[2, 2, 2] = d3_dxdydz[0:Nx-1, 0:Ny-1, 0:Nz-1]
        control[3, 2, 2] = d3_dxdydz[1:Nx,   0:Ny-1, 0:Nz-1]
        control[2, 3, 2] = d3_dxdydz[0:Nx-1, 1:Ny,   0:Nz-1]
        control[3, 3, 2] = d3_dxdydz[1:Nx,   1:Ny,   0:Nz-1]
        control[2, 2, 3] = d3_dxdydz[0:Nx-1, 0:Ny-1, 1:Nz]
        control[3, 2, 3] = d3_dxdydz[1:Nx,   0:Ny-1, 1:Nz]
        control[2, 3, 3] = d3_dxdydz[0:Nx-1, 1:Ny,   1:Nz]
        control[3, 3, 3] = d3_dxdydz[1:Nx,   1:Ny,   1:Nz]
        B = CUBIC_BASIS
        self.coefs = np.ascontiguousarray(
            np.einsum('ai,bj,ck,ijkxyz->abcxyz', B, B, B, control * delta),
            dtype=np.float64,
        )

    def _find_zone(self, n, bins):
        cdef int Nb = bins.shape[0]
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == Nb - 1] = Nb - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny, nz):
        """Interpolate values at all (nx[i], ny[j], nz[k]) triples."""
        cdef int ii, jj, kk
        if isinstance(nx, float): nx = np.array([nx])
        if isinstance(ny, float): ny = np.array([ny])
        if isinstance(nz, float): nz = np.array([nz])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        nz = np.asarray(nz)
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        idx_x = self._find_zone(nx, kx)
        tx_n = (nx - kx[idx_x]) / (kx[idx_x + 1] - kx[idx_x])
        tx = np.array([[1] * len(nx), tx_n, tx_n**2, tx_n**3])
        idx_y = self._find_zone(ny, ky)
        ty_n = (ny - ky[idx_y]) / (ky[idx_y + 1] - ky[idx_y])
        ty = np.array([[1] * len(ny), ty_n, ty_n**2, ty_n**3])
        idx_z = self._find_zone(nz, kz)
        tz_n = (nz - kz[idx_z]) / (kz[idx_z + 1] - kz[idx_z])
        tz = np.array([[1] * len(nz), tz_n, tz_n**2, tz_n**3])
        coefs = np.asarray(self.coefs)
        splines_psi = np.zeros((nx.shape[0], ny.shape[0], nz.shape[0]))
        for _ii in np.unique(np.sort(idx_x)):
            ii = _ii
            lx = np.argwhere(idx_x == ii).flatten()
            ix = lx[:, None, None]
            for _jj in np.unique(np.sort(idx_y)):
                jj = _jj
                ly = np.argwhere(idx_y == jj).flatten()
                iy = ly[None, :, None]
                for _kk in np.unique(np.sort(idx_z)):
                    kk = _kk
                    lz = np.argwhere(idx_z == kk).flatten()
                    iz = lz[None, None, :]
                    splines_psi[ix, iy, iz] = np.einsum(
                        'ax,by,cz,abc->xyz',
                        tx[:, lx], ty[:, ly], tz[:, lz],
                        coefs[:, :, :, ii, jj, kk],
                    )
        return splines_psi

    def integrate_edges(self):
        """Return (int_psi, int_dx, int_dy, int_dz) over each knot-defined cell."""
        cdef int Nx, Ny, Nz
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        Nx = kx.shape[0] - 1
        Ny = ky.shape[0] - 1
        Nz = kz.shape[0] - 1
        dx = kx[1:Nx+1] - kx[0:Nx]
        dy = ky[1:Ny+1] - ky[0:Ny]
        dz = kz[1:Nz+1] - kz[0:Nz]
        tx  = np.array([dx, 0.5*dx, dx/3.0, 0.25*dx])
        dtx = np.ones((4, Nx)); dtx[0] = 0.0
        ty  = np.array([dy, 0.5*dy, dy/3.0, 0.25*dy])
        dty = np.ones((4, Ny)); dty[0] = 0.0
        tz  = np.array([dz, 0.5*dz, dz/3.0, 0.25*dz])
        dtz = np.ones((4, Nz)); dtz[0] = 0.0
        coefs = np.asarray(self.coefs)
        int_psi = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, ty.T, tz.T, coefs)
        int_dx  = np.einsum("xi,yj,zk,ijkxyz->xyz", dtx.T, ty.T, tz.T, coefs)
        int_dy  = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, dty.T, tz.T, coefs)
        int_dz  = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, ty.T, dtz.T, coefs)
        return int_psi, int_dx, int_dy, int_dz

    def integrate_centers(self, limits_x, limits_y, limits_z):
        """Integrate tri-cubic spline over cell-centered limits.

        limits_x/y/z should be cell boundaries (length Nx+1, Ny+1, Nz+1).
        """
        limits_x = np.asarray(limits_x, dtype=np.float64)
        limits_y = np.asarray(limits_y, dtype=np.float64)
        limits_z = np.asarray(limits_z, dtype=np.float64)
        coefs = np.asarray(self.coefs)
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        Nx = int(kx.shape[0])
        Ny = int(ky.shape[0])
        Nz = int(kz.shape[0])
        int_psi = np.zeros((Nx, Ny, Nz))
        int_dx  = np.zeros((Nx, Ny, Nz))
        int_dy  = np.zeros((Nx, Ny, Nz))
        int_dz  = np.zeros((Nx, Ny, Nz))
        Lx = int(limits_x.shape[0]) - 1
        Ly = int(limits_y.shape[0]) - 1
        Lz = int(limits_z.shape[0]) - 1
        for ii in range(Lx):
            xa, xb = limits_x[ii], limits_x[ii + 1]
            on_x0 = (ii == 0)
            on_x1 = (ii == Nx - 1)
            for jj in range(Ly):
                ya, yb = limits_y[jj], limits_y[jj + 1]
                on_y0 = (jj == 0)
                on_y1 = (jj == Ny - 1)
                for kk in range(Lz):
                    za, zb = limits_z[kk], limits_z[kk + 1]
                    on_z0 = (kk == 0)
                    on_z1 = (kk == Nz - 1)
                    x_bdry = on_x0 or on_x1
                    y_bdry = on_y0 or on_y1
                    z_bdry = on_z0 or on_z1
                    ix = 0 if on_x0 else (Nx - 2 if on_x1 else ii - 1)
                    jy = 0 if on_y0 else (Ny - 2 if on_y1 else jj - 1)
                    kz_idx = 0 if on_z0 else (Nz - 2 if on_z1 else kk - 1)
                    acc_psi = acc_dx = acc_dy = acc_dz = 0.0
                    if x_bdry:
                        x_intervals = [(xa, xb, ix)]
                    else:
                        xc = kx[ii]
                        x_intervals = [(xa, xc, ii - 1), (xc, xb, ii)]
                    if y_bdry:
                        y_intervals = [(ya, yb, jy)]
                    else:
                        yc = ky[jj]
                        y_intervals = [(ya, yc, jj - 1), (yc, yb, jj)]
                    if z_bdry:
                        z_intervals = [(za, zb, kz_idx)]
                    else:
                        zc = kz[kk]
                        z_intervals = [(za, zc, kk - 1), (zc, zb, kk)]
                    for (xa_s, xb_s, ci) in x_intervals:
                        for (ya_s, yb_s, cj) in y_intervals:
                            for (za_s, zb_s, ck) in z_intervals:
                                _tx, _dtx = _cubic_one_integral(xa_s, xb_s, kx[ci], kx[ci + 1])
                                _ty, _dty = _cubic_one_integral(ya_s, yb_s, ky[cj], ky[cj + 1])
                                _tz, _dtz = _cubic_one_integral(za_s, zb_s, kz[ck], kz[ck + 1])
                                p, dx_, dy_, dz_ = _cell_integral(_tx, _ty, _tz, _dtx, _dty, _dtz, coefs[:, :, :, ci, cj, ck])
                                acc_psi += p; acc_dx += dx_; acc_dy += dy_; acc_dz += dz_
                    int_psi[ii, jj, kk] = acc_psi
                    int_dx[ii, jj, kk]  = acc_dx
                    int_dy[ii, jj, kk]  = acc_dy
                    int_dz[ii, jj, kk]  = acc_dz
        return int_psi, int_dx, int_dy, int_dz


cdef class QuinticHermite:
    """Tri-quintic Hermite spline over a 3D grid of knots.

    Uses function values, first, and second partial derivatives (plus all
    cross-derivatives) at each knot to build a degree-5 polynomial on each cell.

    Parameters
    ----------
    psi : array_like, shape (Nx, Ny, Nz)
        Function values at the knot points.
    knots_x : array_like, length Nx
        Knot coordinates in x (monotonic).
    knots_y : array_like, length Ny
        Knot coordinates in y (monotonic).
    knots_z : array_like, length Nz
        Knot coordinates in z (monotonic).
    """

    cdef double[:, :, :] psi
    cdef double[:] knots_x
    cdef double[:] knots_y
    cdef double[:] knots_z
    cdef double[:, :, :, :, :, :] coefs    # shape (6, 6, 6, Nx-1, Ny-1, Nz-1)

    def __init__(self, psi, knots_x, knots_y, knots_z):
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        self.knots_z = np.asarray(knots_z, dtype=np.float64)
        self._generate_coefs()

    cdef void _generate_coefs(self):
        cdef int Nx, Ny, Nz
        psi_np = np.asarray(self.psi)
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        Nx = kx.shape[0]
        Ny = ky.shape[0]
        Nz = kz.shape[0]
        # All first derivatives
        d_dx, d_dy, d_dz = first_derivative(self.psi, self.knots_x, self.knots_y, self.knots_z)
        # All second derivatives
        d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz = second_derivative(
            self.psi, self.knots_x, self.knots_y, self.knots_z
        )
        # Higher-order derivatives
        (fxxy, fxxz, fxyy, fxzz, fyyz, fyzz,
         fxyz,
         fxxyy, fxxzz, fyyzz,
         fxxyz, fxyyz, fxyzz,
         fxxyyz, fxxyzz, fxyyzz,
         fxxyyzz) = higher_order_derivative(
            self.psi, self.knots_x, self.knots_y, self.knots_z
        )
        # Delta scaling tensor: shape (6, 6, 6, Nx-1, Ny-1, Nz-1)
        delta_x = kx[1:Nx] - kx[0:Nx-1]
        delta_y = ky[1:Ny] - ky[0:Ny-1]
        delta_z = kz[1:Nz] - kz[0:Nz-1]
        delta = np.ones((6, 6, 6, Nx - 1, Ny - 1, Nz - 1))
        # Rows 2,3 get one factor of delta_x; rows 4,5 get a second factor
        delta[2:] *= delta_x[:, None, None]
        delta[4:] *= delta_x[:, None, None]
        delta[:, 2:] *= delta_y[:, None]
        delta[:, 4:] *= delta_y[:, None]
        delta[:, :, 2:] *= delta_z
        delta[:, :, 4:] *= delta_z
        # Helper: build all 8 corner slices for a 3D field
        def _corners(f):
            return {
                (0, 0, 0): f[0:Nx-1, 0:Ny-1, 0:Nz-1],
                (1, 0, 0): f[1:Nx,   0:Ny-1, 0:Nz-1],
                (0, 1, 0): f[0:Nx-1, 1:Ny,   0:Nz-1],
                (1, 1, 0): f[1:Nx,   1:Ny,   0:Nz-1],
                (0, 0, 1): f[0:Nx-1, 0:Ny-1, 1:Nz],
                (1, 0, 1): f[1:Nx,   0:Ny-1, 1:Nz],
                (0, 1, 1): f[0:Nx-1, 1:Ny,   1:Nz],
                (1, 1, 1): f[1:Nx,   1:Ny,   1:Nz],
            }
        # Assign control tensor.
        # Row indices: 0,1 = value at x0,x1; 2,3 = fx at x0,x1; 4,5 = fxx at x0,x1
        # Same for col (y) and slab (z).
        control = np.zeros((6, 6, 6, Nx - 1, Ny - 1, Nz - 1))
        def _fill(r, c, s, f):
            for (xi, yi, zi), val in _corners(f).items():
                control[r + xi, c + yi, s + zi] = val
        # Pure function values
        _fill(0, 0, 0, psi_np)
        # Pure first derivatives
        _fill(2, 0, 0, d_dx)
        _fill(0, 2, 0, d_dy)
        _fill(0, 0, 2, d_dz)
        # Pure second derivatives
        _fill(4, 0, 0, d2_dx2)
        _fill(0, 4, 0, d2_dy2)
        _fill(0, 0, 4, d2_dz2)
        # Mixed first-first
        _fill(2, 2, 0, d2_dxdy)
        _fill(2, 0, 2, d2_dxdz)
        _fill(0, 2, 2, d2_dydz)
        # Mixed second-first
        _fill(4, 2, 0, fxxy)
        _fill(4, 0, 2, fxxz)
        _fill(2, 4, 0, fxyy)
        _fill(2, 0, 4, fxzz)
        _fill(0, 4, 2, fyyz)
        _fill(0, 2, 4, fyzz)
        # Triple mixed
        _fill(2, 2, 2, fxyz)
        # Mixed second-second
        _fill(4, 4, 0, fxxyy)
        _fill(4, 0, 4, fxxzz)
        _fill(0, 4, 4, fyyzz)
        # Mixed second-first-first and permutations
        _fill(4, 2, 2, fxxyz)
        _fill(2, 4, 2, fxyyz)
        _fill(2, 2, 4, fxyzz)
        # Mixed second-second-first
        _fill(4, 4, 2, fxxyyz)
        _fill(4, 2, 4, fxxyzz)
        _fill(2, 4, 4, fxyyzz)
        # Full mixed second-second-second
        _fill(4, 4, 4, fxxyyzz)
        B = QUINTIC_BASIS
        self.coefs = np.ascontiguousarray(
            np.einsum('ai,bj,ck,ijkxyz->abcxyz', B, B, B, control * delta),
            dtype=np.float64,
        )

    def _find_zone(self, n, bins):
        cdef int Nb = bins.shape[0]
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == Nb - 1] = Nb - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny, nz):
        """Evaluate the tri-quintic spline at all (nx[i], ny[j], nz[k]) triples."""
        cdef int ii, jj, kk
        if isinstance(nx, float): nx = np.array([nx])
        if isinstance(ny, float): ny = np.array([ny])
        if isinstance(nz, float): nz = np.array([nz])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        nz = np.asarray(nz)
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        idx_x = self._find_zone(nx, kx)
        tx_n = (nx - kx[idx_x]) / (kx[idx_x + 1] - kx[idx_x])
        tx = np.array([[1]*len(nx), tx_n, tx_n**2, tx_n**3, tx_n**4, tx_n**5])
        idx_y = self._find_zone(ny, ky)
        ty_n = (ny - ky[idx_y]) / (ky[idx_y + 1] - ky[idx_y])
        ty = np.array([[1]*len(ny), ty_n, ty_n**2, ty_n**3, ty_n**4, ty_n**5])
        idx_z = self._find_zone(nz, kz)
        tz_n = (nz - kz[idx_z]) / (kz[idx_z + 1] - kz[idx_z])
        tz = np.array([[1]*len(nz), tz_n, tz_n**2, tz_n**3, tz_n**4, tz_n**5])
        coefs = np.asarray(self.coefs)
        splines_psi = np.zeros((nx.shape[0], ny.shape[0], nz.shape[0]))
        for _ii in np.unique(np.sort(idx_x)):
            ii = _ii
            lx = np.argwhere(idx_x == ii).flatten()
            ix = lx[:, None, None]
            for _jj in np.unique(np.sort(idx_y)):
                jj = _jj
                ly = np.argwhere(idx_y == jj).flatten()
                iy = ly[None, :, None]
                for _kk in np.unique(np.sort(idx_z)):
                    kk = _kk
                    lz = np.argwhere(idx_z == kk).flatten()
                    iz = lz[None, None, :]
                    splines_psi[ix, iy, iz] = np.einsum(
                        'ax,by,cz,abc->xyz',
                        tx[:, lx], ty[:, ly], tz[:, lz],
                        coefs[:, :, :, ii, jj, kk],
                    )
        return splines_psi

    def integrate_edges(self):
        """Return (int_psi, int_dx, int_dy, int_dz) over each knot-defined cell."""
        cdef int Nx, Ny, Nz
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        Nx = kx.shape[0] - 1
        Ny = ky.shape[0] - 1
        Nz = kz.shape[0] - 1
        dx = kx[1:Nx+1] - kx[0:Nx]
        dy = ky[1:Ny+1] - ky[0:Ny]
        dz = kz[1:Nz+1] - kz[0:Nz]
        tx  = np.array([dx, 0.5*dx, dx/3.0, 0.25*dx, 0.2*dx, dx/6.0])
        dtx = np.ones((6, Nx)); dtx[0] = 0.0
        ty  = np.array([dy, 0.5*dy, dy/3.0, 0.25*dy, 0.2*dy, dy/6.0])
        dty = np.ones((6, Ny)); dty[0] = 0.0
        tz  = np.array([dz, 0.5*dz, dz/3.0, 0.25*dz, 0.2*dz, dz/6.0])
        dtz = np.ones((6, Nz)); dtz[0] = 0.0
        coefs = np.asarray(self.coefs)
        int_psi = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, ty.T, tz.T, coefs)
        int_dx  = np.einsum("xi,yj,zk,ijkxyz->xyz", dtx.T, ty.T, tz.T, coefs)
        int_dy  = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, dty.T, tz.T, coefs)
        int_dz  = np.einsum("xi,yj,zk,ijkxyz->xyz", tx.T, ty.T, dtz.T, coefs)
        return int_psi, int_dx, int_dy, int_dz

    def integrate_centers(self, limits_x, limits_y, limits_z):
        """Integrate tri-quintic spline over cell-centered limits.

        limits_x/y/z should be cell boundaries (length Nx+1, Ny+1, Nz+1).
        """
        limits_x = np.asarray(limits_x, dtype=np.float64)
        limits_y = np.asarray(limits_y, dtype=np.float64)
        limits_z = np.asarray(limits_z, dtype=np.float64)
        coefs = np.asarray(self.coefs)
        kx = np.asarray(self.knots_x)
        ky = np.asarray(self.knots_y)
        kz = np.asarray(self.knots_z)
        Nx = int(kx.shape[0])
        Ny = int(ky.shape[0])
        Nz = int(kz.shape[0])
        int_psi = np.zeros((Nx, Ny, Nz))
        int_dx  = np.zeros((Nx, Ny, Nz))
        int_dy  = np.zeros((Nx, Ny, Nz))
        int_dz  = np.zeros((Nx, Ny, Nz))
        Lx = int(limits_x.shape[0]) - 1
        Ly = int(limits_y.shape[0]) - 1
        Lz = int(limits_z.shape[0]) - 1
        for ii in range(Lx):
            xa, xb = limits_x[ii], limits_x[ii + 1]
            on_x0 = (ii == 0)
            on_x1 = (ii == Nx - 1)
            for jj in range(Ly):
                ya, yb = limits_y[jj], limits_y[jj + 1]
                on_y0 = (jj == 0)
                on_y1 = (jj == Ny - 1)
                for kk in range(Lz):
                    za, zb = limits_z[kk], limits_z[kk + 1]
                    on_z0 = (kk == 0)
                    on_z1 = (kk == Nz - 1)
                    x_bdry = on_x0 or on_x1
                    y_bdry = on_y0 or on_y1
                    z_bdry = on_z0 or on_z1
                    ix = 0 if on_x0 else (Nx - 2 if on_x1 else ii - 1)
                    jy = 0 if on_y0 else (Ny - 2 if on_y1 else jj - 1)
                    kz_ = 0 if on_z0 else (Nz - 2 if on_z1 else kk - 1)
                    acc_psi = acc_dx = acc_dy = acc_dz = 0.0
                    if x_bdry:
                        x_intervals = [(xa, xb, ix)]
                    else:
                        xc = kx[ii]
                        x_intervals = [(xa, xc, ii - 1), (xc, xb, ii)]
                    if y_bdry:
                        y_intervals = [(ya, yb, jy)]
                    else:
                        yc = ky[jj]
                        y_intervals = [(ya, yc, jj - 1), (yc, yb, jj)]
                    if z_bdry:
                        z_intervals = [(za, zb, kz_)]
                    else:
                        zc = kz[kk]
                        z_intervals = [(za, zc, kk - 1), (zc, zb, kk)]
                    for (xa_s, xb_s, ci) in x_intervals:
                        for (ya_s, yb_s, cj) in y_intervals:
                            for (za_s, zb_s, ck) in z_intervals:
                                _tx, _dtx = _quintic_one_integral(xa_s, xb_s, kx[ci], kx[ci+1])
                                _ty, _dty = _quintic_one_integral(ya_s, yb_s, ky[cj], ky[cj+1])
                                _tz, _dtz = _quintic_one_integral(za_s, zb_s, kz[ck], kz[ck+1])
                                p, dx_, dy_, dz_ = _cell_integral(
                                    _tx, _ty, _tz, _dtx, _dty, _dtz,
                                    coefs[:, :, :, ci, cj, ck],
                                )
                                acc_psi += p; acc_dx += dx_; acc_dy += dy_; acc_dz += dz_
                    int_psi[ii, jj, kk] = acc_psi
                    int_dx[ii, jj, kk]  = acc_dx
                    int_dy[ii, jj, kk]  = acc_dy
                    int_dz[ii, jj, kk]  = acc_dz
        return int_psi, int_dx, int_dy, int_dz


########################################################################
# Block and Convenience Wrappers
########################################################################

def _to_block_3d(medium_map):
    """Derive x/y/z split indices from a 3D medium map."""
    from ants.utils import pytools
    Nx, Ny, Nz = medium_map.shape
    x_splits = []
    for jj in range(Ny):
        for kk in range(Nz):
            x_splits.extend(list(pytools._material_index(medium_map[:, jj, kk])))
    x_splits = np.unique(x_splits)
    y_splits = []
    for ii in range(Nx):
        for kk in range(Nz):
            y_splits.extend(list(pytools._material_index(medium_map[ii, :, kk])))
    y_splits = np.unique(y_splits)
    z_splits = []
    for ii in range(Nx):
        for jj in range(Ny):
            z_splits.extend(list(pytools._material_index(medium_map[ii, jj, :])))
    z_splits = np.unique(z_splits)
    return x_splits, y_splits, z_splits


class BlockInterpolation:
    """Apply a spline class on disjoint (x, y, z) blocks.

    Parameters
    ----------
    Splines : class
        Either CubicHermite or QuinticHermite.
    psi, knots_x, knots_y, knots_z : array_like
        Global arrays.
    medium_map : array_like
        3D per-cell medium mapping (used when x/y/z_splits are empty).
    x_splits, y_splits, z_splits : array_like
        Explicit block boundary indices.
    """

    def __init__(self, Splines, psi, knots_x, knots_y, knots_z,
                 medium_map, x_splits, y_splits, z_splits):
        self.Splines = Splines
        self.psi = np.asarray(psi, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)
        self.knots_y = np.asarray(knots_y, dtype=np.float64)
        self.knots_z = np.asarray(knots_z, dtype=np.float64)
        if np.asarray(x_splits).shape[0] > 0:
            self.x_splits = np.asarray(x_splits).copy()
            self.y_splits = np.asarray(y_splits).copy()
            self.z_splits = np.asarray(z_splits).copy()
        else:
            self.x_splits, self.y_splits, self.z_splits = _to_block_3d(
                np.asarray(medium_map)
            )
        self._generate_coefs()

    def _generate_coefs(self):
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        n_zblocks = self.z_splits.shape[0] - 1
        self.splines = []
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]; x2 = self.x_splits[ii + 1]
            col = []
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]; y2 = self.y_splits[jj + 1]
                slab = []
                for kk in range(n_zblocks):
                    z1 = self.z_splits[kk]; z2 = self.z_splits[kk + 1]
                    slab.append(
                        self.Splines(
                            self.psi[x1:x2, y1:y2, z1:z2],
                            self.knots_x[x1:x2],
                            self.knots_y[y1:y2],
                            self.knots_z[z1:z2],
                        )
                    )
                col.append(slab)
            self.splines.append(col)

    def interpolate(self, nx, ny, nz):
        """Interpolate values using the block-wise splines."""
        if isinstance(nx, float): nx = np.array([nx])
        if isinstance(ny, float): ny = np.array([ny])
        if isinstance(nz, float): nz = np.array([nz])
        nx = np.asarray(nx); ny = np.asarray(ny); nz = np.asarray(nz)
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        n_zblocks = self.z_splits.shape[0] - 1
        splines_psi = np.zeros((nx.shape[0], ny.shape[0], nz.shape[0]))
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]; x2 = self.x_splits[ii + 1]
            if ii == 0:
                idx_x = np.argwhere(nx < self.knots_x[x2]).flatten()
            elif ii == n_xblocks - 1:
                idx_x = np.argwhere(nx >= self.knots_x[x1]).flatten()
            else:
                idx_x = np.argwhere(
                    (nx >= self.knots_x[x1]) & (nx < self.knots_x[x2])
                ).flatten()
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]; y2 = self.y_splits[jj + 1]
                if jj == 0:
                    idx_y = np.argwhere(ny < self.knots_y[y2]).flatten()
                elif jj == n_yblocks - 1:
                    idx_y = np.argwhere(ny >= self.knots_y[y1]).flatten()
                else:
                    idx_y = np.argwhere(
                        (ny >= self.knots_y[y1]) & (ny < self.knots_y[y2])
                    ).flatten()
                for kk in range(n_zblocks):
                    z1 = self.z_splits[kk]; z2 = self.z_splits[kk + 1]
                    if kk == 0:
                        idx_z = np.argwhere(nz < self.knots_z[z2]).flatten()
                    elif kk == n_zblocks - 1:
                        idx_z = np.argwhere(nz >= self.knots_z[z1]).flatten()
                    else:
                        idx_z = np.argwhere(
                            (nz >= self.knots_z[z1]) & (nz < self.knots_z[z2])
                        ).flatten()
                    mesh_x, mesh_y, mesh_z = np.meshgrid(
                        idx_x, idx_y, idx_z, indexing="ij"
                    )
                    splines_psi[mesh_x, mesh_y, mesh_z] = (
                        self.splines[ii][jj][kk].interpolate(
                            nx[idx_x], ny[idx_y], nz[idx_z]
                        )
                    )
        return splines_psi

    def integrate_edges(self):
        """Compute (int_psi, int_dx, int_dy, int_dz) for all edges across blocks."""
        Nx = self.knots_x.shape[0] - 1
        Ny = self.knots_y.shape[0] - 1
        Nz = self.knots_z.shape[0] - 1
        int_psi = np.zeros((Nx, Ny, Nz))
        int_dx  = np.zeros((Nx, Ny, Nz))
        int_dy  = np.zeros((Nx, Ny, Nz))
        int_dz  = np.zeros((Nx, Ny, Nz))
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        n_zblocks = self.z_splits.shape[0] - 1
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]; x2 = self.x_splits[ii + 1]
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]; y2 = self.y_splits[jj + 1]
                for kk in range(n_zblocks):
                    z1 = self.z_splits[kk]; z2 = self.z_splits[kk + 1]
                    approx = self.Splines(
                        self.psi[x1:x2+1, y1:y2+1, z1:z2+1],
                        self.knots_x[x1:x2+1],
                        self.knots_y[y1:y2+1],
                        self.knots_z[z1:z2+1],
                    )
                    b_psi, b_dx, b_dy, b_dz = approx.integrate_edges()
                    int_psi[x1:x2, y1:y2, z1:z2] = b_psi
                    int_dx[x1:x2, y1:y2, z1:z2]  = b_dx
                    int_dy[x1:x2, y1:y2, z1:z2]  = b_dy
                    int_dz[x1:x2, y1:y2, z1:z2]  = b_dz
        return int_psi, int_dx, int_dy, int_dz

    def integrate_centers(self, limits_x, limits_y, limits_z):
        """Compute (int_psi, int_dx, int_dy, int_dz) for cell-centered limits."""
        limits_x = np.asarray(limits_x)
        limits_y = np.asarray(limits_y)
        limits_z = np.asarray(limits_z)
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        Nz = self.knots_z.shape[0]
        int_psi = np.zeros((Nx, Ny, Nz))
        int_dx  = np.zeros((Nx, Ny, Nz))
        int_dy  = np.zeros((Nx, Ny, Nz))
        int_dz  = np.zeros((Nx, Ny, Nz))
        n_xblocks = self.x_splits.shape[0] - 1
        n_yblocks = self.y_splits.shape[0] - 1
        n_zblocks = self.z_splits.shape[0] - 1
        for ii in range(n_xblocks):
            x1 = self.x_splits[ii]; x2 = self.x_splits[ii + 1]
            for jj in range(n_yblocks):
                y1 = self.y_splits[jj]; y2 = self.y_splits[jj + 1]
                for kk in range(n_zblocks):
                    z1 = self.z_splits[kk]; z2 = self.z_splits[kk + 1]
                    approx = self.Splines(
                        self.psi[x1:x2, y1:y2, z1:z2],
                        self.knots_x[x1:x2],
                        self.knots_y[y1:y2],
                        self.knots_z[z1:z2],
                    )
                    b_psi, b_dx, b_dy, b_dz = approx.integrate_centers(
                        limits_x[x1:x2+1], limits_y[y1:y2+1], limits_z[z1:z2+1]
                    )
                    int_psi[x1:x2, y1:y2, z1:z2] = b_psi
                    int_dx[x1:x2, y1:y2, z1:z2]  = b_dx
                    int_dy[x1:x2, y1:y2, z1:z2]  = b_dy
                    int_dz[x1:x2, y1:y2, z1:z2]  = b_dz
        return int_psi, int_dx, int_dy, int_dz


class Interpolation:
    """Convenience wrapper that selects the appropriate spline/block combination.

    Parameters
    ----------
    psi, knots_x, knots_y, knots_z : array_like
        Global arrays.
    medium_map : array_like
        Passed to BlockInterpolation when block is True.
    x_splits, y_splits, z_splits : array_like
        Explicit block boundaries.
    block : bool
        If True, build a block-wise interpolator.
    quintic : bool
        If True, use the quintic basis; otherwise cubic.
    """

    def __init__(
        self, psi, knots_x, knots_y, knots_z,
        medium_map, x_splits, y_splits, z_splits,
        block=True, quintic=True,
    ):
        if block and quintic:
            self.instance = BlockInterpolation(
                QuinticHermite, psi, knots_x, knots_y, knots_z,
                medium_map, x_splits, y_splits, z_splits,
            )
        elif block:
            self.instance = BlockInterpolation(
                CubicHermite, psi, knots_x, knots_y, knots_z,
                medium_map, x_splits, y_splits, z_splits,
            )
        elif quintic:
            self.instance = QuinticHermite(psi, knots_x, knots_y, knots_z)
        else:
            self.instance = CubicHermite(psi, knots_x, knots_y, knots_z)

    def interpolate(self, nx, ny, nz):
        return self.instance.interpolate(nx, ny, nz)

    def integrate_edges(self):
        return self.instance.integrate_edges()

    def integrate_centers(self, limits_x, limits_y, limits_z):
        return self.instance.integrate_centers(limits_x, limits_y, limits_z)
