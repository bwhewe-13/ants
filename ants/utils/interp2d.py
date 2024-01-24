########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

from ants.utils import interp1d, pytools


def first_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize derivatives
    d_dx = np.zeros((psi.shape))
    d_dy = np.zeros((psi.shape))
    # Iterate over every row for dpsi/dx
    for jj in range(y.shape[0]):
        d_dx[:,jj] = interp1d.first_derivative(psi[:,jj], x)
    # Iterate over every column for dpsi/dy
    for ii in range(x.shape[0]):
        d_dy[ii] = interp1d.first_derivative(psi[ii], y)
    return d_dx, d_dy


def second_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize derivatives
    d2_dx2 = np.zeros((psi.shape))
    d2_dy2 = np.zeros((psi.shape))
    d2_dxdy = np.zeros((psi.shape))
    # Take derivative in x direction
    for jj in range(y.shape[0]):
        d2_dx2[:,jj] = interp1d.second_derivative(psi[:,jj], x)
        d2_dxdy[:,jj] = interp1d.first_derivative(psi[:,jj], x)
    # Take derivative in y direction
    for ii in range(x.shape[0]):
        d2_dy2[ii] = interp1d.second_derivative(psi[ii], y)
        d2_dxdy[ii] = interp1d.first_derivative(d2_dxdy[ii], y)
    return d2_dx2, d2_dxdy, d2_dy2


def higher_order_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize derivatives
    d3_dx2dy = np.zeros((psi.shape))
    d3_dxdy2 = np.zeros((psi.shape))
    d4_dx2dy2 = np.zeros((psi.shape))
    # Take derivative in x direction
    for jj in range(y.shape[0]):
        d3_dx2dy[:,jj] = interp1d.second_derivative(psi[:,jj], x)
        d3_dxdy2[:,jj] = interp1d.first_derivative(psi[:,jj], x)
        d4_dx2dy2[:,jj] = interp1d.second_derivative(psi[:,jj], x)
    # Take derivative in y direction
    for ii in range(x.shape[0]):
        d3_dx2dy[ii] = interp1d.first_derivative(d3_dx2dy[ii], y)
        d3_dxdy2[ii] = interp1d.second_derivative(d3_dxdy2[ii], y)
        d4_dx2dy2[ii] = interp1d.second_derivative(d4_dx2dy2[ii], y)
    return d3_dx2dy, d3_dxdy2, d4_dx2dy2


def _integral_1_spline(func, lim_x, knots_x, lim_y, knots_y, coefs):
    # Unpack values
    xk0, xk1 = knots_x
    xa, xb = lim_x
    yk0, yk1 = knots_y
    ya, yb = lim_y
    # Integrate terms
    tx, dtx, ty, dty = func(xa, xb, xk0, xk1, ya, yb, yk0, yk1)
    # Calculate integral of flux
    int_psi = tx.T @ coefs @ ty
    # Calculate integral of derivative of flux
    # int_dpsi = dtx.T @ coefs @ dty
    int_dx = dtx.T @ coefs @ ty
    int_dy = tx.T @ coefs @ dty
    return int_psi, int_dx, int_dy


def _integral_2_splines_x(func, lim_x, knots_x, lim_y, knots_y, coefs):
    # Unpack values
    xa, xb = lim_x
    xk0, xk1 = knots_x
    ya, yb = lim_y
    yk0, yc, yk1 = knots_y
    # Left half
    tx1, dtx1, ty1, dty1 = func(xa, xb, xk0, xk1, ya, yc, yk0, yc)
    # Right half
    tx2, dtx2, ty2, dty2 = func(xa, xb, xk0, xk1, yc, yb, yc, yk1)
    # Calculate integral of flux
    int_psi = (tx1.T @ coefs[:,:,0] @ ty1) + (tx2.T @ coefs[:,:,1] @ ty2)
    # Calculate integral of derivative of flux
    # int_dpsi = (dtx1.T @ coefs[:,:,0] @ dty1) + (dtx2.T @ coefs[:,:,1] @ dty2)
    int_dx = (dtx1.T @ coefs[:,:,0] @ ty1) + (dtx2.T @ coefs[:,:,1] @ ty2)
    int_dy = (tx1.T @ coefs[:,:,0] @ dty1) + (tx2.T @ coefs[:,:,1] @ dty2)
    return int_psi, int_dx, int_dy


def _integral_2_splines_y(func, lim_x, knots_x, lim_y, knots_y, coefs):
    # Unpack values
    xa, xb = lim_x
    xk0, xc, xk1 = knots_x
    ya, yb = lim_y
    yk0, yk1 = knots_y
    # Left half
    tx1, dtx1, ty1, dty1 = func(xa, xc, xk0, xc, ya, yb, yk0, yk1)
    # Right half
    tx2, dtx2, ty2, dty2 = func(xc, xb, xc, xk1, ya, yb, yk0, yk1)
    # Calculate integral of flux
    int_psi = (tx1.T @ coefs[:,:,0] @ ty1) + (tx2.T @ coefs[:,:,1] @ ty2)
    # Calculate integral of derivative of flux
    # int_dpsi = (dtx1.T @ coefs[:,:,0] @ dty1) + (dtx2.T @ coefs[:,:,1] @ dty2)
    int_dx = (dtx1.T @ coefs[:,:,0] @ ty1) + (dtx2.T @ coefs[:,:,1] @ ty2)
    int_dy = (tx1.T @ coefs[:,:,0] @ dty1) + (tx2.T @ coefs[:,:,1] @ dty2)
    return int_psi, int_dx, int_dy


def _integral_4_splines(func, lim_x, knots_x, lim_y, knots_y, coefs):
    # Unpack values
    xk0, xc, xk1 = knots_x
    xa, xb = lim_x
    yk0, yc, yk1 = knots_y
    ya, yb = lim_y
    # Bottom Left quadrant
    tx1, dtx1, ty1, dty1 = func(xa, xc, xk0, xc, ya, yc, yk0, yc)
    # Bottom Right quadrant
    tx2, dtx2, ty2, dty2 = func(xc, xb, xc, xk1, ya, yc, yc, yk1)
    # Top Right quadrant
    tx3, dtx3, ty3, dty3 = func(xc, xb, xc, xk1, yc, yb, yc, yk1)
    # Top Left quadrant
    tx4, dtx4, ty4, dty4 = func(xa, xc, xk0, xc, yc, yb, yk0, yc)
    # Calculate integral of flux
    int_psi = (tx1.T @ coefs[:,:,0,0] @ ty1) + (tx2.T @ coefs[:,:,1,0] @ ty2) \
            + (tx3.T @ coefs[:,:,1,1] @ ty3) + (tx4.T @ coefs[:,:,0,1] @ ty4)
    # Calculate integral of derivative of flux
    # int_dpsi = (dtx1.T @ coefs[:,:,0,0] @ dty1) + (dtx2.T @ coefs[:,:,1,0] @ dty2) \
    #         + (dtx3.T @ coefs[:,:,1,1] @ dty3) + (dtx4.T @ coefs[:,:,0,1] @ dty4)
    int_dx = (dtx1.T @ coefs[:,:,0,0] @ ty1) + (dtx2.T @ coefs[:,:,1,0] @ ty2) \
              + (dtx3.T @ coefs[:,:,1,1] @ ty3) + (dtx4.T @ coefs[:,:,0,1] @ ty4)
    int_dy = (tx1.T @ coefs[:,:,0,0] @ dty1) + (tx2.T @ coefs[:,:,1,0] @ dty2) \
              + (tx3.T @ coefs[:,:,1,1] @ dty3) + (tx4.T @ coefs[:,:,0,1] @ dty4)
    return int_psi, int_dx, int_dy


class CubicHermite:
    basis = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                      [-3, 3, -2, -1], [2, -2, 1, 1]])

    def __init__(self, psi, knots_x, knots_y):
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self.knots_y = np.asarray(knots_y)
        self._generate_coefs()

    def _generate_coefs(self):
        # Estimate derivatives
        d_dx, d_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        _, d2_dxdy, _ = second_derivative(self.psi, self.knots_x, self.knots_y)
        # Find knot width
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        delta_y = self.knots_y[1:] - self.knots_y[:-1]
        # Delta matrix
        delta = np.ones((4, 4, delta_x.shape[0], delta_y.shape[0]))
        delta[2:] *= delta_x[(...),:,None]
        delta[:,2:] *= delta_y[(...),:]
        # Control Matrix
        control = np.zeros((4, 4, delta_x.shape[0], delta_y.shape[0]))
        control[:2,:2] = np.array([[self.psi[:-1,:-1], self.psi[:-1,1:]], \
                                   [self.psi[1:,:-1], self.psi[1:,1:]]])
        control[2:,:2] = np.array([[d_dx[:-1,:-1], d_dx[:-1,1:]], \
                                   [d_dx[1:,:-1], d_dx[1:,1:]]])
        control[:2,2:] = np.array([[d_dy[:-1,:-1], d_dy[:-1,1:]], \
                                   [d_dy[1:,:-1], d_dy[1:,1:]]])
        control[2:,2:] = np.array([[d2_dxdy[:-1,:-1], d2_dxdy[:-1,1:]], \
                                   [d2_dxdy[1:,:-1], d2_dxdy[1:,1:]]])
        # Create coefficient matrix
        self.coefs = np.einsum('ij,jk...,kl->il...', CubicHermite.basis, \
                               control * delta, CubicHermite.basis.T)

    def _find_zone(self, n, bins):
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == len(bins) - 1] = len(bins) - 2
        idx[idx == -1] = 0
        return idx

    def _normalize_input(self, n, bins):
        idx = self._find_zone(n, bins)
        t = (n - bins[idx]) / (bins[idx+1] - bins[idx])
        t = np.array([[1] * len(n), t, t**2, t**3])
        return t, idx

    def interpolate(self, nx, ny):
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        # Normalize x input
        tx, idx_x = self._normalize_input(nx, self.knots_x)
        # Normalize y input
        ty, idx_y = self._normalize_input(ny, self.knots_y)
        # Iterate over each zone
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        for ii in np.unique(np.sort(idx_x)):
            loc_x = np.argwhere(idx_x == ii).flatten()
            ix = loc_x[:,None] 
            for jj in np.unique(np.sort(idx_y)):
                loc_y = np.argwhere(idx_y == jj).flatten()
                iy = loc_y[None,:]
                splines_psi[ix,iy] = tx[:,loc_x].T @ self.coefs[:,:,ii,jj] @ ty[:,loc_y]
        return splines_psi

    # Integral of X/Y - edges
    def integrate_edges(self):
        # Take integral, integral of derivative - X
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        Nx = delta_x.shape[0]
        tx = np.array([delta_x, 0.5 * delta_x, 1/3. * delta_x, 0.25 * delta_x])
        dtx = np.ones((4, Nx))
        dtx[0] = 0.0
        # Take integral, integral of derivative - Y
        delta_y = self.knots_y[1:] - self.knots_y[:-1]
        Ny = delta_y.shape[0]
        ty = np.array([delta_y, 0.5 * delta_y, 1/3. * delta_y, 0.25 * delta_y])
        dty = np.ones((4, Ny))
        dty[0] = 0.0
        # Calculate splines
        int_psi = np.einsum("xi, ijxy, jy -> xy", tx.T, self.coefs, ty)
        int_dx = np.einsum("xi, ijxy, jy -> xy", dtx.T, self.coefs, ty)
        int_dy = np.einsum("xi, ijxy, jy -> xy", tx.T, self.coefs, dty)
        return int_psi, int_dx, int_dy

    def _one_integral(a, b, k0, k1):
        # Integral of psi between a and b with knots k0 and k1
        t2 = (b - a) * (a + b - 2 * k0) / (2 * (k1 - k0))
        t3 = (b - a) * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2) \
                    / (3 * (k1 - k0)**2)
        t4 = ((a - k0)**4 - (b - k0)**4) / (4 * (k0 - k1)**3)
        # Integral of dpsi between a and b with knots k0 and k1
        dt2 = (b - a) / (k1 - k0)
        dt3 = (b - a) * (a + b - 2 * k0) / ((k1 - k0)**2)
        dt4 = (b - a) * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2) \
                / ((k1 - k0)**3)
        return np.array([b - a, t2, t3, t4]), np.array([0, dt2, dt3, dt4])

    def _integrals(lim_x0, lim_x1, knot_x0, knot_x1, lim_y0, lim_y1, \
            knot_y0, knot_y1):
        tx, dtx = CubicHermite._one_integral(lim_x0, lim_x1, knot_x0, knot_x1)
        ty, dty = CubicHermite._one_integral(lim_y0, lim_y1, knot_y0, knot_y1)
        return tx, dtx, ty, dty

    # Integral of X/Y - centers
    def integrate_centers(self, limits_x, limits_y):
        limits_x = np.asarray(limits_x)
        limits_y = np.asarray(limits_y)
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        # Interate over spatial cells
        for ii, (xa, xb) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            for jj, (ya, yb) in enumerate(zip(limits_y[:-1], limits_y[1:])):
                # Corner cells - 1 spline
                if ((jj == 0) or (jj == (Ny - 1))) and ((ii == 0) or (ii == (Nx - 1))):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    jy = 0 if jj == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_1_spline( \
                                    CubicHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ix,jy])
                # Edge cells - 2 splines - x dimensions
                elif (ii == 0) or (ii == (Nx - 1)):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_2_splines_x( \
                                    CubicHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    (ya, yb), self.knots_y[jj-1:jj+2], \
                                    self.coefs[:,:,ix,jj-1:jj+1])
                # Edge cells - 2 splines - y dimensions
                elif (jj == 0) or (jj == (Ny - 1)):
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    jy = 0 if jj == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_2_splines_y( \
                                    CubicHermite._integrals, \
                                    (xa, xb), self.knots_x[ii-1:ii+2], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ii-1:ii+1,jy])
                # Center cells - 4 splines
                else:
                    _psi, _dxpsi, _dypsi = _integral_4_splines( \
                                        CubicHermite._integrals, \
                                        (xa, xb), self.knots_x[ii-1:ii+2], \
                                        (ya, yb), self.knots_y[jj-1:jj+2], \
                                        self.coefs[:,:,ii-1:ii+1,jj-1:jj+1])
                int_psi[ii,jj] = _psi.copy()
                int_dx[ii,jj] = _dxpsi.copy()
                int_dy[ii,jj] = _dypsi.copy()
        return int_psi, int_dx, int_dy


class QuinticHermite:
    basis = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0.5, 0], [-10, 10, -6, -4, -1.5, 0.5],
                      [15, -15, 8, 7, 1.5, -1], [-6, 6, -3, -3, -0.5, 0.5]])

    def __init__(self, psi, knots_x, knots_y):
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self.knots_y = np.asarray(knots_y)
        self._generate_coefs()

    def _generate_coefs(self):
        # Estimate derivatives
        d_dx, d_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        d2_dx2, d2_dxdy, d2_dy2 = second_derivative(self.psi, self.knots_x, \
                                                    self.knots_y)
        d3_dx2dy, d3_dxdy2, d4_dx2dy2 = higher_order_derivative(self.psi, \
                                                self.knots_x, self.knots_y)
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        delta_y = self.knots_y[1:] - self.knots_y[:-1]
        # Delta matrix
        delta = np.ones((6, 6, delta_x.shape[0], delta_y.shape[0]))
        delta[2:] *= delta_x[(...),:,None]
        delta[4:] *= delta_x[(...),:,None]
        delta[:,2:] *= delta_y[(...),:]
        delta[:,4:] *= delta_y[(...),:]
        # Control Matrix
        control = np.zeros((6, 6, delta_x.shape[0], delta_y.shape[0]))
        # Top Row
        control[0:2,0:2] = np.array([[self.psi[:-1,:-1], self.psi[:-1,1:]], \
                                   [self.psi[1:,:-1], self.psi[1:,1:]]])
        control[0:2,2:4] = np.array([[d_dy[:-1,:-1], d_dy[:-1,1:]], \
                                   [d_dy[1:,:-1], d_dy[1:,1:]]])
        control[0:2,4:6] = np.array([[d2_dy2[:-1,:-1], d2_dy2[:-1,1:]], \
                                   [d2_dy2[1:,:-1], d2_dy2[1:,1:]]])
        # Middle row
        control[2:4,0:2] = np.array([[d_dx[:-1,:-1], d_dx[:-1,1:]], \
                                   [d_dx[1:,:-1], d_dx[1:,1:]]])
        control[2:4,2:4] = np.array([[d2_dxdy[:-1,:-1], d2_dxdy[:-1,1:]], \
                                   [d2_dxdy[1:,:-1], d2_dxdy[1:,1:]]])
        control[2:4,4:6] = np.array([[d3_dxdy2[:-1,:-1], d3_dxdy2[:-1,1:]], \
                                   [d3_dxdy2[1:,:-1], d3_dxdy2[1:,1:]]])
        # Bottom Row
        control[4:6,0:2] = np.array([[d2_dx2[:-1,:-1], d2_dx2[:-1,1:]], \
                                   [d2_dx2[1:,:-1], d2_dx2[1:,1:]]])
        control[4:6,2:4] = np.array([[d3_dx2dy[:-1,:-1], d3_dx2dy[:-1,1:]], \
                                   [d3_dx2dy[1:,:-1], d3_dx2dy[1:,1:]]])
        control[4:6,4:6] = np.array([[d4_dx2dy2[:-1,:-1], d4_dx2dy2[:-1,1:]], \
                                   [d4_dx2dy2[1:,:-1], d4_dx2dy2[1:,1:]]])
        # Create coefficient matrix
        self.coefs = np.einsum('ij,jk...,kl->il...', QuinticHermite.basis, \
                               control * delta, QuinticHermite.basis.T)

    def _find_zone(self, n, bins):
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == len(bins) - 1] = len(bins) - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny):
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        # Normalize x input
        idx_x = self._find_zone(nx, self.knots_x)
        tx = (nx - self.knots_x[idx_x]) / (self.knots_x[idx_x+1] - self.knots_x[idx_x])
        tx = np.array([[1] * len(nx), tx, tx**2, tx**3, tx**4, tx**5])
        # Normalize y input
        idx_y = self._find_zone(ny, self.knots_y)
        ty = (ny - self.knots_y[idx_y]) / (self.knots_y[idx_y+1] - self.knots_y[idx_y])
        ty = np.array([[1] * len(ny), ty, ty**2, ty**3, ty**4, ty**5])
        # Iterate over each zone
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        for ii in np.unique(np.sort(idx_x)):
            loc_x = np.argwhere(idx_x == ii).flatten()
            ix = loc_x[:,None]
            for jj in np.unique(np.sort(idx_y)):
                loc_y = np.argwhere(idx_y == jj).flatten()
                iy = loc_y[None,:]
                splines_psi[ix,iy] = tx[:,loc_x].T @ self.coefs[:,:,ii,jj] @ ty[:,loc_y]
        return splines_psi

    # Integral of X/Y - edges
    def integrate_edges(self):
        # Take integral, integral of derivative - X
        delta_x = self.knots_x[1:] - self.knots_x[:-1]
        Nx = delta_x.shape[0]
        tx = np.array([delta_x, 0.5 * delta_x, 1/3. * delta_x, 0.25 * delta_x, \
                       0.2 * delta_x, 1/6. * delta_x])
        dtx = np.ones((6, Nx))
        dtx[0] = 0.0
        # Take integral, integral of derivative - Y
        delta_y = self.knots_y[1:] - self.knots_y[:-1]
        Ny = delta_y.shape[0]
        ty = np.array([delta_y, 0.5 * delta_y, 1/3. * delta_y, 0.25 * delta_y, \
                       0.2 * delta_y, 1/6. * delta_y])
        dty = np.ones((6, Ny))
        dty[0] = 0.0
        # Calculate splines
        int_psi = np.einsum("xi, ijxy, jy -> xy", tx.T, self.coefs, ty)
        # int_dpsi = np.einsum("xi, ijxy, jy -> xy", dtx.T, self.coefs, dty)
        int_dx = np.einsum("xi, ijxy, jy -> xy", dtx.T, self.coefs, ty)
        int_dy = np.einsum("xi, ijxy, jy -> xy", tx.T, self.coefs, dty)
        return int_psi, int_dx, int_dy

    def _one_integral(a, b, k0, k1):
        # Integral of psi between a and b with knots k0 and k1
        t2 = (b - a) * (a + b - 2 * k0) / (2 * (k1 - k0))
        t3 = (b - a) * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2) \
                    / (3 * (k1 - k0)**2)
        t4 = ((a - k0)**4 - (b - k0)**4) / (4 * (k0 - k1)**3)
        t5 = (-(a - k0)**5 + (b - k0)**5) / (5 * (k0 - k1)**4)
        t6 = ((a - k0)**6 - (b - k0)**6) / (6 * (k0 - k1)**5)
        t = np.array([b - a, t2, t3, t4, t5, t6])
        # Integral of dpsi between a and b with knots k0 and k1
        dt2 = (b - a) / (k1 - k0)
        dt3 = (b - a) * (a + b - 2 * k0) / ((k1 - k0)**2)
        dt4 = (b - a) * (a**2 + a * b + b**2 - 3 * (a + b) * k0 + 3 * k0**2) \
                / ((k1 - k0)**3)
        dt5 = (-(a - k0)**4 + (b - k0)**4) / ((k0 - k1)**4)
        dt6 = (-(a - k0)**5 + (b - k0)**5) / ((k1 - k0)**5)
        dt = np.array([0, dt2, dt3, dt4, dt5, dt6])
        return t, dt

    def _integrals(lim_x0, lim_x1, knot_x0, knot_x1, lim_y0, lim_y1, \
            knot_y0, knot_y1):
        tx, dtx = QuinticHermite._one_integral(lim_x0, lim_x1, knot_x0, knot_x1)
        ty, dty = QuinticHermite._one_integral(lim_y0, lim_y1, knot_y0, knot_y1)
        return tx, dtx, ty, dty

    # Integral of X/Y - centers
    def integrate_centers(self, limits_x, limits_y):
        limits_x = np.asarray(limits_x)
        limits_y = np.asarray(limits_y)
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        # Interate over spatial cells
        for ii, (xa, xb) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            for jj, (ya, yb) in enumerate(zip(limits_y[:-1], limits_y[1:])):
                # Corner cells - 1 spline
                if ((jj == 0) or (jj == (Ny - 1))) and ((ii == 0) or (ii == (Nx - 1))):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    jy = 0 if jj == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_1_spline( \
                                    QuinticHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ix,jy])
                # Edge cells - 2 splines - x dimensions
                elif (ii == 0) or (ii == (Nx - 1)):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_2_splines_x( \
                                    QuinticHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    (ya, yb), self.knots_y[jj-1:jj+2], \
                                    self.coefs[:,:,ix,jj-1:jj+1])
                # Edge cells - 2 splines - y dimensions
                elif (jj == 0) or (jj == (Ny - 1)):
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    jy = 0 if jj == 0 else -1
                    _psi, _dxpsi, _dypsi = _integral_2_splines_y( \
                                    QuinticHermite._integrals, \
                                    (xa, xb), self.knots_x[ii-1:ii+2], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ii-1:ii+1,jy])
                # Center cells - 4 splines
                else:
                    _psi, _dxpsi, _dypsi = _integral_4_splines( \
                                        QuinticHermite._integrals,
                                        (xa, xb), self.knots_x[ii-1:ii+2], \
                                        (ya, yb), self.knots_y[jj-1:jj+2], \
                                        self.coefs[:,:,ii-1:ii+1,jj-1:jj+1])
                int_psi[ii,jj] = _psi.copy()
                int_dx[ii,jj] = _dxpsi.copy()
                int_dy[ii,jj] = _dypsi.copy()
        return int_psi, int_dx, int_dy


class BlockInterpolation:

    def __init__(self, Splines, psi, knots_x, knots_y, medium_map):
        self.Splines = Splines
        self.psi = np.asarray(psi)
        self.knots_x = np.asarray(knots_x)
        self.knots_y = np.asarray(knots_y)
        # Determine knot splits
        self.x_splits, self.y_splits = pytools._to_block(np.asarray(medium_map))
        self._generate_coefs()


    def _generate_coefs(self):
        # Create 2D list of Interpolations
        self.splines = []
        for (x1, x2) in zip(self.x_splits[:-1], self.x_splits[1:]):
            one_col = []
            for (y1, y2) in zip(self.y_splits[:-1], self.y_splits[1:]):
                # Initialize new spline section
                one_col.append(self.Splines(self.psi[x1:x2,y1:y2], \
                               self.knots_x[x1:x2], self.knots_y[y1:y2]))
            self.splines.append(one_col)


    def interpolate(self, nx, ny):
        if isinstance(nx, float):
            nx = np.array([nx])
        if isinstance(ny, float):
            ny = np.array([ny])
        nx = np.asarray(nx)
        ny = np.asarray(ny)
        # Initialize splines
        splines_psi = np.zeros((nx.shape[0], ny.shape[0]))
        # Iterate over x blocks
        for ii, (x1, x2) in enumerate(zip(self.x_splits[:-1], self.x_splits[1:])):
            # Find correct x block
            if ii == 0:
                idx_x = np.argwhere(nx < self.knots_x[x2]).flatten()
            elif ii == (self.x_splits.shape[0] - 2):
                idx_x = np.argwhere(nx >= self.knots_x[x1]).flatten()
            else:
                idx_x = np.argwhere((nx >= self.knots_x[x1]) \
                                    & (nx < self.knots_x[x2])).flatten()
            # Iterate over y block
            for jj, (y1, y2) in enumerate(zip(self.y_splits[:-1], self.y_splits[1:])):
                # Find correct y block
                if jj == 0:
                    idx_y = np.argwhere(ny < self.knots_y[y2]).flatten()
                elif jj == (self.y_splits.shape[0] - 2):
                    idx_y = np.argwhere(ny >= self.knots_y[y1]).flatten()
                else:
                    idx_y = np.argwhere((ny >= self.knots_y[y1]) \
                                    & (ny < self.knots_y[y2])).flatten()
                # Interpolate on block
                mesh_x, mesh_y = np.meshgrid(idx_x, idx_y, indexing="ij")
                splines_psi[mesh_x,mesh_y] = self.splines[ii][jj].interpolate(nx[idx_x], ny[idx_y])
        return splines_psi


   # Integral of X/Y - edges
    def integrate_edges(self):
        # Initialize full matrices
        Nx = self.knots_x.shape[0] - 1
        Ny = self.knots_y.shape[0] - 1
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        # Iterate over each block
        for (x1, x2) in zip(self.x_splits[:-1], self.x_splits[1:]):
            for (y1, y2) in zip(self.y_splits[:-1], self.y_splits[1:]):
                # Initialize new spline section
                approx = self.Splines(self.psi[x1:x2+1,y1:y2+1], \
                                self.knots_x[x1:x2+1], self.knots_y[y1:y2+1])
                # Interpolate on block
                b_int_psi, b_int_dx, b_int_dy = approx.integrate_edges()
                int_psi[x1:x2,y1:y2] = b_int_psi.copy()
                int_dx[x1:x2,y1:y2] = b_int_dx.copy()
                int_dy[x1:x2,y1:y2] = b_int_dy.copy()
        return int_psi, int_dx, int_dy


    # Integral of X/Y - centers
    def integrate_centers(self, limits_x, limits_y):
        limits_x = np.asarray(limits_x)
        limits_y = np.asarray(limits_y)
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dx = np.zeros((Nx, Ny))
        int_dy = np.zeros((Nx, Ny))
        # Iterate over each block
        for (x1, x2) in zip(self.x_splits[:-1], self.x_splits[1:]):
            for (y1, y2) in zip(self.y_splits[:-1], self.y_splits[1:]):
                # Initialize new spline section
                approx = self.Splines(self.psi[x1:x2,y1:y2], \
                                self.knots_x[x1:x2], self.knots_y[y1:y2])
                # Interpolate on block
                b_int_psi, b_int_dx, b_int_dy = approx.integrate_centers( \
                                    limits_x[x1:x2+1], limits_y[y1:y2+1])
                int_psi[x1:x2,y1:y2] = b_int_psi.copy()
                int_dx[x1:x2,y1:y2] = b_int_dx.copy()
                int_dy[x1:x2,y1:y2] = b_int_dy.copy()
        return int_psi, int_dx, int_dy


class Interpolation:
    
    def __init__(self, psi, knots_x, knots_y, medium_map=None, \
                 block=True, quintic=True):
        self.block = block
        self.quintic = quintic
        # Block Quintic
        if (block) and (quintic):
            self.instance = BlockInterpolation(QuinticHermite, psi, \
                                            knots_x, knots_y, medium_map)
        # Quintic
        elif (not block) and (quintic):
            self.instance = QuinticHermite(psi, knots_x, knots_y)

        # Block Cubic
        elif (block) and (not quintic):
            self.instance = BlockInterpolation(CubicHermite, psi, \
                                            knots_x, knots_y, medium_map)
        # Cubic
        elif (not block) and (not quintic):
            self.instance = CubicHermite(psi, knots_x, knots_y)


    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)