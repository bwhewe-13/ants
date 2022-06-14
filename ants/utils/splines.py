########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from .dimensions import index_generator

import numpy as np

def first_derivative(x, y):
    # Added second order at endpoints
    yp = []
    assert len(x) == len(y), "Need to be same length"
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

def hermite(x, y, splines, stype="cubic"):
    # splines are a list of the spline boundary locations in relation to x
    # ie    x = [0, 0.05, ..., 0.995, 1] with len(x) = 100 and
    #       splines = [0, 20, ..., 80, 100]
    # Or splines can be an integer
    if isinstance(splines, int):
        splines = index_generator(len(x) - 1, splines)
    assert len(splines) <= len(x), ("The number of splines cannot be more"
            " than the number of data points, splines < {}".format(len(x)))
    spline_x = []
    spline_y = []
    yp = first_derivative(x, y)
    if stype == "quintic":
        ypp = second_derivative(x, y)

    for n in range(len(splines) - 1):
        # temp_x = np.linspace(x[n], x[n+1], np.max([splines[n]+1, 2]))
        # temp_x = np.linspace(x[n], x[n+1], splines)
        temp_x = x[splines[n]:splines[n+1]+1]
        if stype == "cubic":
            temp_y = CubicHermite.cubic_spline(temp_x, y[splines[n]], \
                      y[splines[n+1]], yp[splines[n]], yp[splines[n+1]],\
                      x[splines[n]], x[splines[n+1]])
        elif stype == "quintic":
            temp_y = QuinticHermite.quintic_spline(temp_x, y[splines[n]], \
                        y[splines[n+1]], yp[splines[n]], yp[splines[n+1]], \
                        ypp[splines[n]], ypp[splines[n+1]], x[splines[n]], \
                        x[splines[n+1]])            
        # spline_x.append(temp_x[:-1])
        spline_y.append(temp_y[:-1])
        if n == len(splines) - 2:
            # spline_x.append([temp_x[-1]])
            spline_y.append([temp_y[-1]])
    # Unnest the lists
    # spline_x = np.array([item for sublist in spline_x for item in sublist])
    spline_y = np.array([item for sublist in spline_y for item in sublist])
    return spline_y, splines


class CubicHermite:
    # Basis Functions for Cubic Hermite Splines
    def _phi0(x, tk0, tk1):
        return 2 * ((x - tk0) / (tk1 - tk0))**3 \
                - 3 * ((x - tk0) / (tk1 - tk0))**2 + 1

    def _phi1(x, tk0, tk1):
        return -2 * ((x - tk0) / (tk1 - tk0))**3 \
                + 3 * ((x - tk0) / (tk1 - tk0))**2

    def _psi0(x, tk0, tk1):
        return (tk1 - tk0) * (((x - tk0) / (tk1 - tk0))**3 \
                - 2 * ((x - tk0) / (tk1 - tk0))**2 \
                + ((x - tk0) / (tk1 - tk0)))

    def _psi1(x, tk0, tk1):
        return (tk1 - tk0) * (((x - tk0) / (tk1 - tk0))**3 \
                - ((x - tk0) / (tk1 - tk0))**2)

    def cubic_spline(x, yk0, yk1, ykp0, ykp1, tk0, tk1):
        return yk0 * CubicHermite._phi0(x, tk0, tk1) \
                + yk1 * CubicHermite._phi1(x, tk0, tk1) \
                + ykp0 * CubicHermite._psi0(x, tk0, tk1) \
                + ykp1 * CubicHermite._psi1(x, tk0, tk1)


class QuinticHermite:
    # Basis Functions for Quintic Hermite Splines
    def _phi0(x, tk0, tk1):
        return -6 * ((x - tk0) / (tk1 - tk0))**5 \
                + 15 * ((x - tk0) / (tk1 - tk0))**4 \
                - 10 * ((x - tk0) / (tk1 - tk0))**3 + 1

    def _phi1(x, tk0, tk1):
        return 6 * ((x - tk0) / (tk1 - tk0))**5 \
                - 15 * ((x - tk0) / (tk1 - tk0))**4 \
                + 10 * ((x - tk0) / (tk1 - tk0))**3

    def _psi0(x, tk0, tk1):
        return (tk1 - tk0) * (-3 * ((x - tk0) / (tk1 - tk0))**5 \
                + 8 * ((x - tk0) / (tk1 - tk0))**4 \
                - 6 * ((x - tk0) / (tk1 - tk0))**3 \
                + ((x - tk0) / (tk1 - tk0)))

    def _psi1(x, tk0, tk1):
        return (tk1 - tk0) * (-3 * ((x - tk0) / (tk1 - tk0))**5 \
                + 7 * ((x - tk0) / (tk1 - tk0))**4 \
                - 4 * ((x - tk0) / (tk1 - tk0))**3)

    def _theta0(x, tk0, tk1):
        return (tk1 - tk0)**2 * (-0.5 * ((x - tk0) / (tk1 - tk0))**5 \
                + 1.5 * ((x - tk0) / (tk1 - tk0))**4 \
                - 1.5 * ((x - tk0) / (tk1 - tk0))**3 \
                + 0.5 * ((x - tk0) / (tk1 - tk0))**2)

    def _theta1(x, tk0, tk1):
        return (tk1 - tk0)**2 * (0.5 * ((x - tk0) / (tk1 - tk0))**5 \
                - ((x - tk0) / (tk1 - tk0))**4 \
                + 0.5 * ((x - tk0) / (tk1 - tk0))**3)

    def quintic_spline(x, yk0, yk1, ykp0, ykp1, ykpp0, ykpp1, tk0, tk1):
        return yk0 * QuinticHermite._phi0(x, tk0, tk1) \
                + yk1 * QuinticHermite._phi1(x, tk0, tk1) \
                + ykp0 * QuinticHermite._psi0(x, tk0, tk1) \
                + ykp1 * QuinticHermite._psi1(x, tk0, tk1) \
                + ykpp0 * QuinticHermite._theta0(x, tk0, tk1) \
                + ykpp1 * QuinticHermite._theta1(x, tk0, tk1)

if __name__ == "__main__":
    print("Additional Work\n","="*25)
    print(" [ ] Add basis function checks (graphs)")
    print(" [ ] Toy problem with known solution")
