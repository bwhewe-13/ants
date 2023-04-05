########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np
from scipy.integrate import quad

def first_derivative(x, y):
    assert len(x) == len(y), "Need to be same length"
    assert len(x) > 2, "Need to be at least three points"
    yp = []
    for n in range(len(x)):
        if n == 0:
            # First Order Accurate
            # yp.append((y[1] - y[0]) / (x[1] - x[0]))
            # Second Order Accurate
            # yp.append((-3 * y[0] + 4 * y[1] - y[2]) / (x[2] - x[0]))
            yp.append((y[0] - y[1]) / (x[0] - x[1]) \
                    + (y[0] - y[2]) / (x[0] - x[2]) \
                    + (y[2] - y[1]) / (x[1] - x[2]))
        elif n == (len(x) - 1):
            # First Order Accurate
            # yp.append((y[n] - y[n-1]) / (x[n] - x[n-1]))
            # Second Order Accurate
            # yp.append((3 * y[n] - 4 * y[n-1] + y[n-2]) / (x[n] - x[n-2]))
            yp.append((y[n] - y[n-1]) / (x[n] - x[n-1]) \
                     + (y[n] - y[n-2]) / (x[n] - x[n-2]) \
                     + (-y[n-1] + y[n-2]) / (x[n-1] - x[n-2]))

        else:
            # yp.append((y[n+1] - y[n-1]) / (x[n+1] - x[n-1]))
            half1 = (y[n+1] - y[n]) / (x[n+1] - x[n])
            half2 = (y[n] - y[n-1]) / (x[n] - x[n-1])
            yp.append(0.5 * (half1 + half2))
    return np.array(yp)

def second_derivative(x, y):
    assert len(x) == len(y), "Need to be same length"
    assert len(x) > 2, "Need to be at least three points"
    ypp = []
    for n in range(len(x)):
        if n == 0:
            # First Order Accurate
            ypp.append((y[n] - 2 * y[n+1] + y[n+2]) \
                                / ((x[n+2] - x[n+1]) * (x[n+1] - x[n])))
            # Second Order Accurate
            # ypp.append((2 * y[n] - 5 * y[n+1] + 4 * y[n+2] - y[n+3]) / \
            #             ((x[n+3] - x[n+2]) * (x[n+2] - x[n+1]) * (x[n+1] - x[n])))
        elif n == (len(x) - 1):
            # First Order Accurate
            ypp.append((y[n] - 2 * y[n-1] + y[n-2]) \
                                / ((x[n] - x[n-1]) * (x[n-1] - x[n-2])))
            # Second Order Accurate
            # ypp.append((2 * y[n] - 5 * y[n-1] + 4 * y[n-2] - y[n-3]) / \
            #             ((x[n] - x[n-1]) * (x[n-1] - x[n-2]) * (x[n-2] - x[n-3])))
        else:
            # ypp.append((y[n+1] - 2 * y[n] + y[n-1]) \
            #                     / ((x[n+1] - x[n]) * (x[n] - x[n-1])))
            half1 = (y[n+1] - y[n]) / (x[n+1] - x[n])
            half2 = (y[n] - y[n-1]) / (x[n] - x[n-1])
            ypp.append((half1 - half2) / (0.5 * (x[n+1] - x[n-1])))
    return np.array(ypp)

class CubicNatural:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._generate_coefs()

    def __call__(self, n):
        idx = np.digitize(n, bins=self.x) - 1
        return np.array([CubicNatural._one_spline(self, nn, ii) \
                                        for nn, ii in zip(n, idx)])

    def _generate_coefs(self):
        self.h = self.x[1:] - self.x[:-1]
        b = (self.y[1:] - self.y[:-1]) / self.h
        v = 2 * (self.h[1:] + self.h[:-1])
        u = 6 * (b[1:] - b[:-1])
        length = len(self.x)
        A = np.zeros((length-2, length-2))
        A[np.arange(length-2), np.arange(length-2)] = v.copy()
        A[np.arange(length-3)+1, np.arange(length-3)] = self.h[1:-1]
        A[np.arange(length-3), np.arange(length-3)+1] = self.h[1:-1]
        self.z = np.linalg.solve(A, u)
        self.z = np.concatenate(([[0.], self.z, [0.]]))

    def _one_spline(self, n, idx):
        if idx >= 0 and idx <= len(self.x) - 2:
            return self.z[idx+1] / (6*self.h[idx]) * (n - self.x[idx])**3 \
                + self.z[idx] / (6*self.h[idx]) * (self.x[idx+1] - n)**3 \
                + (self.y[idx+1] / self.h[idx] - self.z[idx+1] * self.h[idx] / 6) \
                * (n - self.x[idx]) + (self.y[idx] / self.h[idx] - self.z[idx] \
                * self.h[idx] / 6) * (self.x[idx+1] - n)
        elif idx < 0:
            temp = self.z[0] / (2 * self.h[0]) * (self.x[1] - n)**2 \
                    + (self.y[1] / self.h[0] - self.z[1] * self.h[0] / 6) \
                    - (self.y[0] / self.h[0] - self.z[0] * self.h[0] / 6)
            return self.y[0] + temp * (n - self.x[0])
        elif idx >= len(self.x) - 2:
            temp = self.z[-1] / (2 * self.h[-2]) * (n - self.x[-2])**2 \
                    + (self.y[-1] / self.h[-2] - self.z[-1] * self.h[-1] / 6) \
                    - (self.y[-2] / self.h[-2] - self.z[-2] * self.h[-1] / 6)
            return self.y[-1] + temp * (n - self.x[-1])

class CubicHermite:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dydx = first_derivative(self.x, self.y)
        self._generate_coefs()

    def __call__(self, n):
        idx = np.digitize(n, bins=self.x) - 1
        idx[idx == len(self.x) - 1] = len(self.x) - 2
        idx[idx == -1] = 0
        return CubicHermite._one_spline(n, self.x[idx], self.x[idx+1],\
                             self.coefs[:,idx])

    @classmethod
    def _generate_new(cls, x, coefs):
        self = object.__new__(cls)
        self.x = x
        self.coefs = coefs
        return self

    def _generate_coefs(self):
        self.coefs = np.zeros((4, len(self.x)-1))
        for ii in range(len(self.x)-1):
            self.coefs[0, ii] = (2 * self.y[ii] - 2 * self.y[ii+1] \
                            + (self.dydx[ii] + self.dydx[ii+1]) \
                            * (self.x[ii+1] - self.x[ii]))
            self.coefs[1, ii] = (-3 * self.y[ii] + 3 * self.y[ii+1] \
                            + (-2*self.dydx[ii] - self.dydx[ii+1]) \
                            * (self.x[ii+1] - self.x[ii])) 
            self.coefs[2, ii] = self.dydx[ii] * (self.x[ii+1] - self.x[ii])
            self.coefs[3, ii] = self.y[ii] 

    def _one_spline(x, tk0, tk1, coef):
        t = (x - tk0) / (tk1 - tk0)
        return t**3 * coef[0] + t**2 * coef[1] + t * coef[2] + coef[3]

    def derivative(self):
        coefs_d = np.zeros((4, len(self.x)-1))
        for ii in range(len(self.x)-1):
            coefs_d[1, ii] = (6 / (self.x[ii+1] - self.x[ii])) \
                            * (self.y[ii] - self.y[ii+1]) \
                            + 3 * (self.dydx[ii] + self.dydx[ii+1])
            coefs_d[2, ii] = (6 / (self.x[ii+1] - self.x[ii])) \
                            * (-self.y[ii] + self.y[ii+1]) \
                            - 4 * self.dydx[ii] - 2 * self.dydx[ii+1]
            coefs_d[3, ii] = self.dydx[ii] 
        return self._generate_new(self.x, coefs_d)

    def basis_functions(x):
        tk0 = x[0]
        tk1 = x[-1]
        t = ((x - tk0) / (tk1 - tk0))
        phi0 = (x - tk0) * (0.5 * t**3 - t**2) + x
        phi1 = (x - tk0) * (-0.5 * t**3 + t**2)
        psi0 = (tk1 - tk0) * ((x - tk0) * (0.25 * t**3 - 2/3 * t**2) \
                + x * (x - 2 * tk0) / (2 * (tk1 - tk0)))
        psi1 = (tk1 - tk0) * (x - tk0) * (0.25 * t**3 - 1/3 * t**2)
        return np.array([phi0, phi1, psi0, psi1])

    # def integrate(self, cell_edges):
    #     integral = []
    #     knots = np.array([self.y[:-1], self.y[1:], self.dydx[:-1], self.dydx[1:]])
    #     for ii in range(len(self.x)-1):
    #         n = np.linspace(cell_edges[ii], cell_edges[ii+2], 3)
    #         temp_int = np.sum(knots[:,ii,None] * CubicHermite.basis_functions(n), axis=0)
    #         integral.append(np.diff(temp_int)[:-1])
    #         if ii == len(self.x) - 2:
    #             integral.append([np.diff(temp_int)[-1]])
    #     integral = np.array([item for sublist in integral for item in sublist])
    #     return integral

    def _single_spline(a, b, t1, t2, y1, y2, yp1, yp2):
        def phi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 2 * t**3 - 3 * t**2 + 1

        def phi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return -2 * t**3 + 3 * t**2

        def psi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (t**3 - 2 * t**2 + t)

        def psi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (t**3 - t**2)

        integral = 0.0
        ys = [y1, y2, yp1, yp2]
        funcs = [phi1, phi2, psi1, psi2]
        for ii, jj in zip(ys, funcs):
            integral += ii * quad(jj, a, b, args=(t1, t2))[0]
        return integral

    def _single_dspline(a, b, t1, t2, y1, y2, yp1, yp2):
        def phi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 6 / (t2 - t1) * (t**2 - t)

        def phi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 6 / (t2 - t1) * (t - t**2)

        def psi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 1 - 4 * t + 3 * t**2

        def psi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 3 * t**2 - 2 * t

        integral = 0.0
        ys = [y1, y2, yp1, yp2]
        funcs = [phi1, phi2, psi1, psi2]
        for ii, jj in zip(ys, funcs):
            integral += ii * quad(jj, a, b, args=(t1, t2))[0]
        return integral

    def integrate_splines_edges(edges_x, flux, params):
        """
        Flux and dflux are at the cell edges (cells + 1)
        Returns integral of spline and spline derivative between cell edges
        """
        # Faster this way - but not general
        # delta_x = np.diff(edges_x)
        # dflux = first_derivative(edges_x, flux)
        # spline = 0.5 * (flux[:-1] + flux[1:]) * delta_x[:,None,None] \
        #         + 1/12 * (dflux[:-1] - dflux[1:]) * delta_x[:,None,None]**2
        # dspline = flux[1:] - flux[:-1]
        dflux = first_derivative(edges_x, flux)
        spline = np.zeros((flux.shape[0] - 1,) + flux.shape[1:])
        # print(spline.shape)
        dspline = np.zeros(spline.shape)
        for gg in range(params["groups"]):
            for nn in range(params["angles"]):
                for ii in range(flux.shape[0] - 1):
                    spline[ii,nn,gg] = CubicHermite._single_spline(edges_x[ii], edges_x[ii+1], \
                            edges_x[ii], edges_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg])
                    dspline[ii,nn,gg] = CubicHermite._single_dspline(edges_x[ii], edges_x[ii+1], \
                            edges_x[ii], edges_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg])
        return spline, dspline

    def integrate_splines_centers(edges_x, flux, params):
        """
        Flux and dflux are at the cell centers (cells)
        Returns integral of spline and spline derivative between cell edges
        """
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        dflux = first_derivative(centers_x, flux)
        spline = np.zeros(flux.shape)
        dspline = np.zeros(spline.shape)
        for gg in range(params["groups"]):
            for nn in range(params["angles"]):
                # Calculate the first cell (one spline)
                spline[0,nn,gg] = CubicHermite._single_spline(edges_x[0], edges_x[1], \
                            centers_x[0], centers_x[1], flux[0,nn,gg], \
                            flux[1,nn,gg], dflux[0,nn,gg], dflux[1,nn,gg])
                dspline[0,nn,gg] = CubicHermite._single_dspline(edges_x[0], edges_x[1], \
                            centers_x[0], centers_x[1], flux[0,nn,gg], \
                            flux[1,nn,gg], dflux[0,nn,gg], dflux[1,nn,gg])
                for ii in range(1, flux.shape[0]-1):
                    # Take half integrals and add together
                    S0 = CubicHermite._single_spline(edges_x[ii], centers_x[ii], \
                            centers_x[ii-1], centers_x[ii], flux[ii-1,nn,gg], \
                            flux[ii,nn,gg], dflux[ii-1,nn,gg], dflux[ii,nn,gg])
                    S1 = CubicHermite._single_spline(centers_x[ii], edges_x[ii+1], \
                            centers_x[ii], centers_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg])
                    spline[ii,nn,gg] = S0 + S1
                    dS0 = CubicHermite._single_dspline(edges_x[ii], centers_x[ii], \
                            centers_x[ii-1], centers_x[ii], flux[ii-1,nn,gg], \
                            flux[ii,nn,gg], dflux[ii-1,nn,gg], dflux[ii,nn,gg])
                    dS1 = CubicHermite._single_dspline(centers_x[ii], edges_x[ii+1], \
                            centers_x[ii], centers_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg])
                    dspline[ii,nn,gg] = dS0 + dS1
                # Calculate the last cell (one spline)
                spline[-1,nn,gg] = CubicHermite._single_spline(edges_x[-2], edges_x[-1], \
                            centers_x[-2], centers_x[-1], flux[-2,nn,gg], \
                            flux[-1,nn,gg], dflux[-2,nn,gg], dflux[-1,nn,gg])
                dspline[-1,nn,gg] = CubicHermite._single_dspline(edges_x[-2], edges_x[-1], \
                            centers_x[-2], centers_x[-1], flux[-2,nn,gg], \
                            flux[-1,nn,gg], dflux[-2,nn,gg], dflux[-1,nn,gg])
        return spline, dspline
        

class QuinticHermite:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dydx = first_derivative(self.x, self.y)
        self.d2ydx2 = second_derivative(self.x, self.y)
        self._generate_coefs()

    def __call__(self, n):
        idx = np.digitize(n, bins=self.x) - 1
        idx[idx == len(self.x) - 1] = len(self.x) - 2
        idx[idx == -1] = 0
        return QuinticHermite._one_spline(n, self.x[idx], \
                            np.diff(self.x)[idx], self.coefs[:,idx])

    @classmethod
    def _generate_new(cls, x, coefs):
        self = object.__new__(cls)
        self.x = x
        self.coefs = coefs
        return self

    def _generate_coefs(self):
        self.coefs = np.zeros((6, len(self.x)-1))
        for ii in range(len(self.x)-1):
            self.coefs[0, ii] = 6 * (-self.y[ii] + self.y[ii+1]) \
                            - 3 * (self.x[ii+1] - self.x[ii]) \
                            * (self.dydx[ii] + self.dydx[ii+1]) \
                            + 0.5 * (self.x[ii+1] - self.x[ii])**2 \
                            * (-self.d2ydx2[ii] + self.d2ydx2[ii+1])
            self.coefs[1, ii] = 15 * (self.y[ii] - self.y[ii+1]) \
                            + (self.x[ii+1] - self.x[ii]) \
                            * (8 * self.dydx[ii] + 7 * self.dydx[ii+1]) \
                            + (self.x[ii+1] - self.x[ii])**2 \
                            * (1.5 * self.d2ydx2[ii] - self.d2ydx2[ii+1])
            self.coefs[2, ii] = 10 * (-self.y[ii] + self.y[ii+1]) \
                            + (self.x[ii+1] - self.x[ii]) \
                            * (-6 * self.dydx[ii] - 4 * self.dydx[ii+1]) \
                            + (self.x[ii+1] - self.x[ii])**2 \
                            * (-1.5 * self.d2ydx2[ii] + 0.5 * self.d2ydx2[ii+1])
            self.coefs[3, ii] = 0.5 * self.d2ydx2[ii] * (self.x[ii+1] - self.x[ii])**2
            self.coefs[4, ii] = self.dydx[ii] * (self.x[ii+1] - self.x[ii])
            self.coefs[5, ii] = self.y[ii] 

    def _one_spline(x, t, dt, coef):
        return ((x-t) / dt)**5 * coef[0] + ((x-t) / dt)**4 * coef[1] \
                + ((x-t) / dt)**3 * coef[2] + ((x-t) / dt)**2 * coef[3] \
                + ((x-t) / dt) * coef[4] + coef[5]

    def derivative(self):
        coefs_d = np.zeros((6, len(self.x)-1))
        for ii in range(len(self.x)-1):
            coefs_d[1, ii] = (30 / (self.x[ii+1] - self.x[ii])) \
                            * (-self.y[ii] + self.y[ii+1]) \
                            - 15 * (self.dydx[ii] + self.dydx[ii+1]) \
                            + 2.5 * (self.x[ii+1] - self.x[ii]) \
                            * (-self.d2ydx2[ii] + self.d2ydx2[ii+1])
            coefs_d[2, ii] = 60 / (self.x[ii+1] - self.x[ii]) \
                            * (self.y[ii] - self.y[ii+1]) \
                            + (32 * self.dydx[ii] + 28 * self.dydx[ii+1]) \
                            + (self.x[ii+1] - self.x[ii]) \
                            * (6 * self.d2ydx2[ii] - 4 * self.d2ydx2[ii+1])
            coefs_d[3, ii] = 30 / (self.x[ii+1] - self.x[ii]) \
                            * (-self.y[ii] + self.y[ii+1]) \
                            + (-18 * self.dydx[ii] - 12 * self.dydx[ii+1]) \
                            + (self.x[ii+1] - self.x[ii]) \
                            * (-4.5 * self.d2ydx2[ii] + 1.5 * self.d2ydx2[ii+1])
            coefs_d[4, ii] = self.d2ydx2[ii] * (self.x[ii+1] - self.x[ii])
            coefs_d[5, ii] = self.dydx[ii]
        return self._generate_new(self.x, coefs_d)
    
    def basis_functions(x):
        tk0 = x[0]
        tk1 = x[-1]
        t = ((x - tk0) / (tk1 - tk0))
        phi0 = (x - tk0) * (-t**5 + 3 * t**4 - 2.5 * t**3) + x
        phi1 = (x - tk0) * (t**5 - 3 * t**4 + 2.5 * t**3)
        psi0 = (tk1 - tk0) * ((x - tk0) * (-0.5 * t**5 + 1.6 * t**4 \
                - 1.5 * t**3) + x * (x - 2 * tk0) / (2 * (tk1 - tk0)))
        psi1 = (tk1 - tk0) * (x - tk0) * (-0.5 * t**5 + 1.4 * t**4 - t**3)
        theta0 = (tk1 - tk0)**2 * (x - tk0) * (-1/12 * t**5 \
                    + 0.3 * t**4 - 0.375 * t**3 + 1/6 * t**2)
        theta1 = (tk1 - tk0)**2 * (x - tk0) * (-1/12 * t**5 \
                    - 0.2 * t**4 + 0.125 * t**3)
        return np.array([phi0, phi1, psi0, psi1, theta0, theta1])

    def integrate(self, cell_edges):
        integral = []
        knots = np.array([self.y[:-1], self.y[1:], self.dydx[:-1], \
                        self.dydx[1:], self.d2ydx2[:-1], self.d2ydx2[1:]])
        for ii in range(len(self.x)-1):
            n = np.linspace(cell_edges[ii], cell_edges[ii+2], 3)
            temp_int = np.sum(knots[:,ii,None] * QuinticHermite.basis_functions(n), axis=0)
            integral.append(np.diff(temp_int)[:-1])
            if ii == len(self.x) - 2:
                integral.append([np.diff(temp_int)[-1]])
        integral = np.array([item for sublist in integral for item in sublist])
        return integral

    def _single_spline(a, b, t1, t2, y1, y2, yp1, yp2, ypp1, ypp2):
        def phi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return -6 * t**5 + 15 * t**4 - 10 * t**3 + 1

        def phi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        def psi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (-3 * t**5 + 8 * t**4 - 6 * t**3 + t)

        def psi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (-3 * t**5 + 7 * t**4 - 4 * t**3)

        def theta1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1)**2 * (-0.5 * t**5 + 1.5 * t**4 - 1.5 * t**3 \
                    + 0.5 * t**2)

        def theta2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1)**2 * (0.5 * t**5 - t**4 + 0.5 * t**3)

        integral = 0.0
        ys = [y1, y2, yp1, yp2, ypp1, ypp2]
        funcs = [phi1, phi2, psi1, psi2, theta1, theta2]
        for ii, jj in zip(ys, funcs):
            integral += ii * quad(jj, a, b, args=(t1, t2))[0]
        return integral

    def _single_dspline(a, b, t1, t2, y1, y2, yp1, yp2, ypp1, ypp2):
        def phi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 30 / (t2 - t1) * (-1 * t**2 + 2 * t**3 - t**4)

        def phi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 30 / (t2 - t1) * (t**2 - 2 * t**3 + t**4)

        def psi1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return 1 - 18 * t**2 + 32 * t**3 - 15 * t**4

        def psi2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return - 12 * t**2 + 28 * t**3 - 15 * t**4

        def theta1(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (t - 4.5 * t**2 + 6 * t**3 - 2.5 * t**4)

        def theta2(x, t1, t2):
            t = (x - t1) / (t2 - t1)
            return (t2 - t1) * (1.5 * t**2 - 4 * t**3 + 2.5 * t**4)

        integral = 0.0
        ys = [y1, y2, yp1, yp2, ypp1, ypp2]
        funcs = [phi1, phi2, psi1, psi2, theta1, theta2]
        for ii, jj in zip(ys, funcs):
            integral += ii * quad(jj, a, b, args=(t1, t2))[0]
        return integral

    def integrate_splines_edges(edges_x, flux, params):
        """
        Flux and dflux are at the cell edges (cells + 1)
        Returns integral of spline and spline derivative between cell edges
        """
        # Faster this way - but not general
        # delta_x = np.diff(edges_x)
        # dflux = first_derivative(edges_x, flux)
        # spline2 = 0.5 * (flux[:-1] + flux[1:]) * delta_x[:,None,None] \
        #         + 1/12 * (dflux[:-1] - dflux[1:]) * delta_x[:,None,None]**2
        # dspline2 = flux[1:] - flux[:-1]
        dflux = first_derivative(edges_x, flux)
        d2flux = second_derivative(edges_x, flux)
        spline = np.zeros((flux.shape[0] - 1,) + flux.shape[1:])
        dspline = np.zeros(spline.shape)
        for gg in range(params["groups"]):
            for nn in range(params["angles"]):
                for ii in range(flux.shape[0] - 1):
                    spline[ii,nn,gg] = QuinticHermite._single_spline(edges_x[ii], edges_x[ii+1], \
                            edges_x[ii], edges_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg], \
                            d2flux[ii,nn,gg], d2flux[ii+1,nn,gg])
                    dspline[ii,nn,gg] = QuinticHermite._single_dspline(edges_x[ii], edges_x[ii+1], \
                            edges_x[ii], edges_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg], \
                            d2flux[ii,nn,gg], d2flux[ii+1,nn,gg])
        return spline, dspline

    def integrate_splines_centers(edges_x, flux, params):
        """
        Flux and dflux are at the cell centers (cells)
        Returns integral of spline and spline derivative between cell edges
        """
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        dflux = first_derivative(centers_x, flux)
        d2flux = second_derivative(centers_x, flux)
        spline = np.zeros(flux.shape)
        dspline = np.zeros(spline.shape)
        for gg in range(params["groups"]):
            for nn in range(params["angles"]):
                # Calculate the first cell (one spline)
                spline[0,nn,gg] = QuinticHermite._single_spline(edges_x[0], edges_x[1], \
                            centers_x[0], centers_x[1], flux[0,nn,gg], \
                            flux[1,nn,gg], dflux[0,nn,gg], dflux[1,nn,gg], \
                            d2flux[0,nn,gg], d2flux[1,nn,gg])
                dspline[0,nn,gg] = QuinticHermite._single_dspline(edges_x[0], edges_x[1], \
                            centers_x[0], centers_x[1], flux[0,nn,gg], \
                            flux[1,nn,gg], dflux[0,nn,gg], dflux[1,nn,gg], \
                            d2flux[0,nn,gg], d2flux[1,nn,gg])
                for ii in range(1, flux.shape[0]-1):
                    # Take half integrals and add together
                    S0 = QuinticHermite._single_spline(edges_x[ii], centers_x[ii], \
                            centers_x[ii-1], centers_x[ii], flux[ii-1,nn,gg], \
                            flux[ii,nn,gg], dflux[ii-1,nn,gg], dflux[ii,nn,gg], \
                            d2flux[ii-1,nn,gg], d2flux[ii,nn,gg])
                    S1 = QuinticHermite._single_spline(centers_x[ii], edges_x[ii+1], \
                            centers_x[ii], centers_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg], \
                            d2flux[ii,nn,gg], d2flux[ii+1,nn,gg])
                    spline[ii,nn,gg] = S0 + S1
                    dS0 = QuinticHermite._single_dspline(edges_x[ii], centers_x[ii], \
                            centers_x[ii-1], centers_x[ii], flux[ii-1,nn,gg], \
                            flux[ii,nn,gg], dflux[ii-1,nn,gg], dflux[ii,nn,gg], \
                            d2flux[ii-1,nn,gg], d2flux[ii,nn,gg])
                    dS1 = QuinticHermite._single_dspline(centers_x[ii], edges_x[ii+1], \
                            centers_x[ii], centers_x[ii+1], flux[ii,nn,gg], \
                            flux[ii+1,nn,gg], dflux[ii,nn,gg], dflux[ii+1,nn,gg], \
                            d2flux[ii,nn,gg], d2flux[ii+1,nn,gg])
                    dspline[ii,nn,gg] = dS0 + dS1
                # Calculate the last cell (one spline)
                spline[-1,nn,gg] = QuinticHermite._single_spline(edges_x[-2], edges_x[-1], \
                            centers_x[-2], centers_x[-1], flux[-2,nn,gg], \
                            flux[-1,nn,gg], dflux[-2,nn,gg], dflux[-1,nn,gg], \
                            d2flux[-2,nn,gg], d2flux[-1,nn,gg])
                dspline[-1,nn,gg] = QuinticHermite._single_dspline(edges_x[-2], edges_x[-1], \
                            centers_x[-2], centers_x[-1], flux[-2,nn,gg], \
                            flux[-1,nn,gg], dflux[-2,nn,gg], dflux[-1,nn,gg], \
                            d2flux[-2,nn,gg], d2flux[-1,nn,gg])
        return spline, dspline