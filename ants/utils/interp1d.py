########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

def first_derivative(x, y):
    assert len(x) == len(y), "Need to be same length"
    assert len(x) > 2, "Need to be at least three points"
    yp = []
    for n in range(len(x)):
        if n == 0:
            # First Order Accurate
            yp.append((y[1] - y[0]) / (x[1] - x[0]))
            # Second Order Accurate
            # yp.append((-3 * y[0] + 4 * y[1] - y[2]) / (x[2] - x[0]))
        elif n == (len(x) - 1):
            # First Order Accurate
            yp.append((y[n] - y[n-1]) / (x[n] - x[n-1]))
            # Second Order Accurate
            # yp.append((3 * y[n] - 4 * y[n-1] + y[n-2]) / (x[n] - x[n-2]))
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

    def integrate(self, cell_edges):
        integral = []
        knots = np.array([self.y[:-1], self.y[1:], self.dydx[:-1], self.dydx[1:]])
        for ii in range(len(self.x)-1):
            n = np.linspace(cell_edges[ii], cell_edges[ii+2], 3)
            temp_int = np.sum(knots[:,ii,None] * CubicHermite.basis_functions(n), axis=0)
            integral.append(np.diff(temp_int)[:-1])
            if ii == len(self.x) - 2:
                integral.append([np.diff(temp_int)[-1]])
        integral = np.array([item for sublist in integral for item in sublist])
        return integral
        

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
