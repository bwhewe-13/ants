########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from . import dimensions 

import numpy as np
import warnings


def first_derivative(x, y):
    # Added second order at endpoints
    yp = []
    assert len(x) == len(y), "Need to be same length"
    if len(x) < 3:
        return np.repeat(np.diff(y) / np.diff(x), len(x))
    for n in range(len(x)):
        if n == 0:
            # yp.append((y[1] - y[0]) / (x[1] - x[0]))
            yp.append((-3 * y[0] + 4 * y[1] - y[2]) / (x[2] - x[0]))
        elif n == (len(x) - 1):
            # yp.append((y[n] - y[n-1]) / (x[n] - x[n-1]))
            yp.append((3 * y[n] - 4 * y[n-1] + y[n-2]) / (x[n] - x[n-2]))
        else:
            yp.append((y[n+1] - y[n-1]) / (x[n+1] - x[n-1]))
    return np.array(yp)

def second_derivative(x, y):
    ypp = []
    assert len(x) == len(y), "Need to be same length"
    if len(x) < 3:
        return np.repeat(np.diff(y) / np.diff(x), len(x))
    for n in range(len(x)):
        if n == 0:
            ypp.append((y[n] - 2 * y[n+1] + y[n+2]) \
                                / ((x[n+2] - x[n+1]) * (x[n+1] - x[n])))
            # ypp.append((2 * y[n] - 5 * y[n+1] + 4 * y[n+2] - y[n+3]) / \
            #             ((x[n+3] - x[n+2]) * (x[n+2] - x[n+1]) * (x[n+1] - x[n])))
        elif n == (len(x) - 1):
            ypp.append((y[n] - 2 * y[n-1] + y[n-2]) \
                                / ((x[n] - x[n-1]) * (x[n-1] - x[n-2])))
            # ypp.append((2 * y[n] - 5 * y[n-1] + 4 * y[n-2] - y[n-3]) / \
            #             ((x[n] - x[n-1]) * (x[n-1] - x[n-2]) * (x[n-2] - x[n-3])))
        else:
            ypp.append((y[n+1] - 2 * y[n] + y[n-1]) \
                                / ((x[n+1] - x[n]) * (x[n] - x[n-1])))
    return np.array(ypp)


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
        return CubicHermite._one_spline(n, self.x[idx], \
                            np.diff(self.x)[idx], self.coefs[:,idx])

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

    def _one_spline(x, t, dt, coef):
        return ((x-t) / dt)**3 * coef[0] + ((x-t) / dt)**2 * coef[1] \
                                    + ((x-t) / dt) * coef[2] + coef[3]

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