########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

from ants.utils import interp1d


def first_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize dpsi/dx
    dpsi_x = np.zeros((psi.shape))
    # Iterate over every row for dpsi/dx
    for jj in range(y.shape[0]):
        dpsi_x[:,jj] = interp1d.first_derivative(psi[:,jj], x)
    # Initialize dpsi/dy
    dpsi_y = np.zeros((psi.shape))
    # Iterate over every column for dpsi/dy
    for ii in range(x.shape[0]):
        dpsi_y[ii] = interp1d.first_derivative(psi[ii], y)
    return dpsi_x, dpsi_y


def second_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize d2psi/dx2
    d2_dxdx = np.zeros((psi.shape))
    # Iterate over every row for d2psi/dx2
    for jj in range(y.shape[0]):
        d2_dxdx[:,jj] = interp1d.second_derivative(psi[:,jj], x)
    # Initialize d2psi/dy2
    d2_dydy = np.zeros((psi.shape))
    # Iterate over every row for d2psi/dy2
    for ii in range(x.shape[0]):
        d2_dydy[ii] = interp1d.second_derivative(psi[ii], y)
    # Initialize d2psi/dxdy
    d2_dxdy = np.zeros((psi.shape))
    # Take first derivative in dx
    for jj in range(y.shape[0]):
        d2_dxdy[:,jj] = interp1d.first_derivative(psi[:,jj], x)
    # Take first derivative in dy
    for ii in range(x.shape[0]):
        d2_dxdy[ii] = interp1d.first_derivative(d2_dxdy[ii], y)
    return d2_dxdx, d2_dxdy, d2_dydy


def higher_order_derivative(psi, x, y):
    assert (x.shape[0], y.shape[0]) == psi.shape, "Need to be the same size"
    assert x.shape[0] > 2 and y.shape[0] > 2, "Need to be at least 3 knots"
    # Initialize d3psi/dx2dy
    d3_dx2dy = np.zeros((psi.shape))
    # Iterate over every row for d2psi/dx2
    for jj in range(y.shape[0]):
        d3_dx2dy[:,jj] = interp1d.second_derivative(psi[:,jj], x)
    # Take first derivative in dy
    for ii in range(x.shape[0]):
        d3_dx2dy[ii] = interp1d.first_derivative(d3_dx2dy[ii], y)
    # Initialize d3psi/dxdy2
    d3_dxdy2 = np.zeros((psi.shape))
    # Iterate over every row for d2psi/dy2
    for ii in range(x.shape[0]):
        d3_dxdy2[ii] = interp1d.second_derivative(psi[ii], y)
    # Take first derivative in dx
    for jj in range(y.shape[0]):
        d3_dxdy2[:,jj] = interp1d.first_derivative(d3_dxdy2[:,jj], x)
    # Initialize d4psi/dx2dy2
    d4_dx2dy2 = np.zeros((psi.shape))
    # Iterate over every row for d2psi/dx2
    for jj in range(y.shape[0]):
        d4_dx2dy2[:,jj] = interp1d.second_derivative(psi[:,jj], x)
    # Iterate over every row for d2psi/dy2
    for ii in range(x.shape[0]):
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
    int_dpsi = dtx.T @ coefs @ dty
    return int_psi, int_dpsi


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
    int_dpsi = (dtx1.T @ coefs[:,:,0] @ dty1) + (dtx2.T @ coefs[:,:,1] @ dty2)
    return int_psi, int_dpsi


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
    int_dpsi = (dtx1.T @ coefs[:,:,0] @ dty1) + (dtx2.T @ coefs[:,:,1] @ dty2)
    return int_psi, int_dpsi


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
    int_dpsi = (dtx1.T @ coefs[:,:,0,0] @ dty1) + (dtx2.T @ coefs[:,:,1,0] @ dty2) \
            + (dtx3.T @ coefs[:,:,1,1] @ dty3) + (dtx4.T @ coefs[:,:,0,1] @ dty4)
    return int_psi, int_dpsi


class CubicHermite:
    basis = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                      [-3, 3, -2, -1], [2, -2, 1, 1]])

    def __init__(self, psi, knots_x, knots_y):
        self.psi = psi
        self.knots_x = knots_x
        self.knots_y = knots_y
        self._generate_coefs()

    def _generate_coefs(self):
        dpsi_dx, dpsi_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        _, d2psi_dxdy, _ = second_derivative(self.psi, self.knots_x, self.knots_y)

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
        control[2:,:2] = np.array([[dpsi_dx[:-1,:-1], dpsi_dx[:-1,1:]], \
                                   [dpsi_dx[1:,:-1], dpsi_dx[1:,1:]]])
        control[:2,2:] = np.array([[dpsi_dy[:-1,:-1], dpsi_dy[:-1,1:]], \
                                   [dpsi_dy[1:,:-1], dpsi_dy[1:,1:]]])
        control[2:,2:] = np.array([[d2psi_dxdy[:-1,:-1], d2psi_dxdy[:-1,1:]], \
                                   [d2psi_dxdy[1:,:-1], d2psi_dxdy[1:,1:]]])
        # Create coefficient matrix
        self.coefs = np.einsum('ij,jk...,kl->il...', CubicHermite.basis, \
                               control * delta, CubicHermite.basis.T)

    def _find_zone(self, n, bins):
        idx = np.digitize(n, bins=bins) - 1
        idx[idx == len(bins) - 1] = len(bins) - 2
        idx[idx == -1] = 0
        return idx

    def interpolate(self, nx, ny):
        # Normalize x input
        idx_x = self._find_zone(nx, self.knots_x)
        tx = (nx - self.knots_x[idx_x]) / (self.knots_x[idx_x+1] - self.knots_x[idx_x])
        tx = np.array([[1] * len(nx), tx, tx**2, tx**3])
        # Normalize y input
        idx_y = self._find_zone(ny, self.knots_y)
        ty = (ny - self.knots_y[idx_y]) / (self.knots_y[idx_y+1] - self.knots_y[idx_y])
        ty = np.array([[1] * len(ny), ty, ty**2, ty**3])
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
        int_dpsi = np.einsum("xi, ijxy, jy -> xy", dtx.T, self.coefs, dty)
        return int_psi, int_dpsi

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
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dpsi = np.zeros((Nx, Ny))
        # Interate over spatial cells
        for ii, (xa, xb) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            for jj, (ya, yb) in enumerate(zip(limits_y[:-1], limits_y[1:])):
                # Corner cells - 1 spline
                if ((jj == 0) or (jj == (Ny - 1))) and ((ii == 0) or (ii == (Nx - 1))):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    jy = 0 if jj == 0 else -1
                    _psi, _dpsi = _integral_1_spline(CubicHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ix,jy])
                # Edge cells - 2 splines - x dimensions
                elif (ii == 0) or (ii == (Nx - 1)):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    _psi, _dpsi = _integral_2_splines_x(CubicHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    (ya, yb), self.knots_y[jj-1:jj+2], \
                                    self.coefs[:,:,ix,jj-1:jj+1])
                # Edge cells - 2 splines - y dimensions
                elif (jj == 0) or (jj == (Ny - 1)):
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    jy = 0 if jj == 0 else -1
                    _psi, _dpsi = _integral_2_splines_y(CubicHermite._integrals, \
                                    (xa, xb), self.knots_x[ii-1:ii+2], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ii-1:ii+1,jy])
                # Center cells - 4 splines
                else:
                    _psi, _dpsi = _integral_4_splines(CubicHermite._integrals,
                                        (xa, xb), self.knots_x[ii-1:ii+2], \
                                        (ya, yb), self.knots_y[jj-1:jj+2], \
                                        self.coefs[:,:,ii-1:ii+1,jj-1:jj+1])
                int_psi[ii,jj] = _psi
                int_dpsi[ii,jj] = _dpsi
        return int_psi, int_dpsi


class QuinticHermite:
    basis = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0.5, 0], [-10, 10, -6, -4, -1.5, 0.5],
                      [15, -15, 8, 7, 1.5, -1], [-6, 6, -3, -3, -0.5, 0.5]])

    def __init__(self, psi, knots_x, knots_y):
        self.psi = psi
        self.knots_x = knots_x
        self.knots_y = knots_y
        self._generate_coefs()

    def _generate_coefs(self):
        d_dx, d_dy = first_derivative(self.psi, self.knots_x, self.knots_y)
        d2_dx2, d2_dxdy, d2_dy2 = second_derivative(self.psi, self.knots_x, self.knots_y)
        d3_dx2dy, d3_dxdy2, d4_dx2dy2 = higher_order_derivative(self.psi, self.knots_x, self.knots_y)

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
        int_dpsi = np.einsum("xi, ijxy, jy -> xy", dtx.T, self.coefs, dty)
        return int_psi, int_dpsi

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
        Nx = self.knots_x.shape[0]
        Ny = self.knots_y.shape[0]
        int_psi = np.zeros((Nx, Ny))
        int_dpsi = np.zeros((Nx, Ny))
        # Interate over spatial cells
        for ii, (xa, xb) in enumerate(zip(limits_x[:-1], limits_x[1:])):
            for jj, (ya, yb) in enumerate(zip(limits_y[:-1], limits_y[1:])):
                # Corner cells - 1 spline
                if ((jj == 0) or (jj == (Ny - 1))) and ((ii == 0) or (ii == (Nx - 1))):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    jy = 0 if jj == 0 else -1
                    _psi, _dpsi = _integral_1_spline(QuinticHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ix,jy])
                # Edge cells - 2 splines - x dimensions
                elif (ii == 0) or (ii == (Nx - 1)):
                    idx_x = [0, 1] if ii == 0 else [-2, -1]
                    ix = 0 if ii == 0 else -1
                    _psi, _dpsi = _integral_2_splines_x(QuinticHermite._integrals, \
                                    limits_x[idx_x], self.knots_x[idx_x], \
                                    (ya, yb), self.knots_y[jj-1:jj+2], \
                                    self.coefs[:,:,ix,jj-1:jj+1])
                # Edge cells - 2 splines - y dimensions
                elif (jj == 0) or (jj == (Ny - 1)):
                    idx_y = [0, 1] if jj == 0 else [-2, -1]
                    jy = 0 if jj == 0 else -1
                    _psi, _dpsi = _integral_2_splines_y(QuinticHermite._integrals, \
                                    (xa, xb), self.knots_x[ii-1:ii+2], \
                                    limits_y[idx_y], self.knots_y[idx_y], \
                                    self.coefs[:,:,ii-1:ii+1,jy])
                # Center cells - 4 splines
                else:
                    _psi, _dpsi = _integral_4_splines(QuinticHermite._integrals,
                                        (xa, xb), self.knots_x[ii-1:ii+2], \
                                        (ya, yb), self.knots_y[jj-1:jj+2], \
                                        self.coefs[:,:,ii-1:ii+1,jj-1:jj+1])
                int_psi[ii,jj] = _psi
                int_dpsi[ii,jj] = _dpsi
        return int_psi, int_dpsi