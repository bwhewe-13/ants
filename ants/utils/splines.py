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

def hermite(x, y, x_width=None, knots=None, stype="cubic", aux_func="derive"):
    # knots are a list of the knot locations in relation to x
    # ie    x = [0, 0.05, ..., 0.995, 1] with len(x) = 100 and
    #       knots = [0, 20, ..., 80, 100]
    # Or knots can be an integer, None
    if knots is None:
        knots = optimal_knots(x, y)
    elif isinstance(knots, int):
        knots = dimensions.index_generator(len(x)-1, knots)
    if len(knots) > len(x):
        message = ("The number of knots cannot be more than the number of data"
            "points (knots < {}). Knots are reduced to {}".format(len(x), len(x)-1))
        warnings.warn(message)
        knots = dimensions.index_generator(len(x)-1, len(x)-1)
    approx_y = []
    approx_yp = []
    yp = first_derivative(x, y)
    if stype == "quintic":
        ypp = second_derivative(x, y)
    for n in range(len(knots) - 1):
        temp_x = x[knots[n]:knots[n+1]+1].copy()
        # print(temp_x)
        if aux_func == "derive":
            temp_x_edges = temp_x.copy()
        elif aux_func == "integrate":
            temp_x_edges = dimensions.spatial_edges(temp_x, \
                                        x_width[knots[n]:knots[n+1]+1])
        if stype == "cubic":
            temp_y = CubicHermite.cubic_spline(temp_x, y[knots[n]], \
                            y[knots[n+1]], yp[knots[n]], yp[knots[n+1]],\
                            x[knots[n]], x[knots[n+1]])
            temp_yp = CubicHermite.cubic_spline(temp_x_edges, y[knots[n]], \
                            y[knots[n+1]], yp[knots[n]], yp[knots[n+1]],\
                            temp_x_edges[0], temp_x_edges[-1], dtype=aux_func)
        elif stype == "quintic":
            temp_y = QuinticHermite.quintic_spline(temp_x, y[knots[n]], \
                        y[knots[n+1]], yp[knots[n]], yp[knots[n+1]], \
                        ypp[knots[n]], ypp[knots[n+1]], x[knots[n]], \
                        x[knots[n+1]])
            temp_yp = QuinticHermite.quintic_spline(temp_x_edges, \
                            y[knots[n]], y[knots[n+1]], yp[knots[n]], \
                            yp[knots[n+1]], ypp[knots[n]], ypp[knots[n+1]], \
                            temp_x_edges[0], temp_x_edges[-1], dtype=aux_func)
        if aux_func == "integrate":
            temp_yp = np.diff(temp_yp)
        approx_y.append(temp_y[:-1])
        approx_yp.append(temp_yp[:-1])
        if n == len(knots) - 2:
            approx_y.append([temp_y[-1]])
            approx_yp.append([temp_yp[-1]])
    approx_y = np.array([item for sublist in approx_y for item in sublist])
    approx_yp = np.array([item for sublist in approx_yp for item in sublist])
    return approx_y, approx_yp

def ghost_splines(x_centers, y_centers, x_cell_widths, direction, \
                  split, knots=None, stype="quintic", dtype="diamond"):
    # Adding points to short splines
    # x_centers are the centers for the region
    # y_centers are the centers for the entire medium
    x_edges = dimensions.spatial_edges(x_centers, x_cell_widths)
    y_edges = dimensions.flux_edges(y_centers, direction, \
                    slice(split.start, split.stop + 1), dtype=dtype)
    y_both = dimensions.mesh_centers_edges(y_centers[split], y_edges)
    x_both = dimensions.mesh_centers_edges(x_centers, x_edges)
    x_plus, y_plus = dimensions.mesh_refinement(x_both, y_both)
    y_spline, y_deriv = hermite(x_plus, y_plus, knots=knots, stype=stype)
    return y_spline[2::4], y_deriv[2::4]    

def optimal_knots(x, y, atol=5e-5):
    # Taken from Ihtzaz Qamar's "Method to determine optimum number 
    # of knots for cubic splines." - Specific for cubic
    knots = [0]#, 1, 2, 3, len(y)-1, len(y)-2, len(y)-3, len(y)-4]
    for cell in range(len(x)-2):
        area_diff = abs((0.5 * y[cell] - y[cell+1] + 0.5 * y[cell+2]) \
                        * (x[cell+1] - x[cell]))
        if area_diff > atol:
            knots.append(cell+1)
    knots = np.append(knots, len(x)-1)
    try:
        additional = dimensions.index_generator(len(x)-1, int(len(x)*0.2))
    except ZeroDivisionError:
        additional = np.array([])
    knots = np.sort(np.unique(np.concatenate((knots, additional))))
    # print("\n\n", knots, "\n\n")
    return knots.astype(np.int32)

def t(x, tk0, tk1):
    return ((x - tk0) / (tk1 - tk0))

class CubicHermite:
    # Basis Functions for Cubic Hermite Splines
    def _phi0(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 6 / (tk1 - tk0) * (t(x, tk0, tk1)**2 - t(x, tk0, tk1))
        elif dtype == "integrate":
            return 0.5*t(x, tk0, tk1)**3 * (x - tk0) - t(x, tk0, tk1)**2 \
                    * (x - tk0) + x
        return 2*t(x, tk0, tk1)**3 - 3*t(x, tk0, tk1)**2 + 1

    def _phi1(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 6 / (tk1 - tk0) * (t(x, tk0, tk1) - t(x, tk0, tk1)**2)
        elif dtype == "integrate":
            return -0.5*t(x, tk0, tk1)**3 * (x - tk0) + t(x, tk0, tk1)**2 \
                * (x - tk0)
            # return 0
        return -2*t(x, tk0, tk1)**3 + 3*t(x, tk0, tk1)**2

    def _psi0(x, tk0, tk1, dtype=None):
        # print("in func", tk1, tk0, t(x, tk0, tk1))
        if dtype == "derive":
            return 3*t(x, tk0, tk1)**2 - 4*t(x, tk0, tk1) + 1
        elif dtype == "integrate":
            return (tk1 - tk0) * (0.25*t(x, tk0, tk1)**3 * (x - tk0) \
                - 2/3*t(x, tk0, tk1)**2 * (x - tk0) \
                + x**2 / (2 * (tk1 - tk0)) - (tk0 * x) / (tk1 - tk0))
            # return (x**2 / (2 * (tk1 - tk0)) - (tk0 * x) / (tk1 - tk0))
        return (tk1 - tk0) * (t(x, tk0, tk1)**3 \
                - 2*t(x, tk0, tk1)**2 + t(x, tk0, tk1))

    def _psi1(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 3*t(x, tk0, tk1)**2 - 2*t(x, tk0, tk1)
        elif dtype == "integrate":
            return (tk1 - tk0) * (0.25*t(x, tk0, tk1)**3 * (x - tk0) \
                - 1/3*t(x, tk0, tk1)**2 * (x - tk0))
            # return 0
        return (tk1 - tk0) * (t(x, tk0, tk1)**3 - t(x, tk0, tk1)**2)

    def cubic_spline(x, yk0, yk1, ykp0, ykp1, tk0, tk1, dtype=None):
        return yk0 * CubicHermite._phi0(x, tk0, tk1, dtype=dtype) \
                + yk1 * CubicHermite._phi1(x, tk0, tk1, dtype=dtype) \
                + ykp0 * CubicHermite._psi0(x, tk0, tk1, dtype=dtype) \
                + ykp1 * CubicHermite._psi1(x, tk0, tk1, dtype=dtype)

class QuinticHermite:
    # Basis Functions for Quintic Hermite Splines
    def _phi0(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 30 / (tk1 - tk0) * (-t(x, tk0, tk1)**4 \
                    + 2*t(x, tk0, tk1)**3 - t(x, tk0, tk1)**2)
        elif dtype == "integrate":
            return -1*t(x, tk0, tk1)**5 * (x - tk0) + 3*t(x, tk0, tk1)**4 \
                    * (x - tk0) - 2.5*t(x, tk0, tk1)**3 * (x - tk0) + x
        return -6*t(x, tk0, tk1)**5 + 15*t(x, tk0, tk1)**4 \
                - 10*t(x, tk0, tk1)**3 + 1

    def _phi1(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 30 / (tk1 - tk0) * (t(x, tk0, tk1)**4 \
                - 2*t(x, tk0, tk1)**3 + t(x, tk0, tk1)**2)
        elif dtype == "integrate":
            return t(x, tk0, tk1)**5 * (x - tk0) - 3*t(x, tk0, tk1)**4 \
                    * (x - tk0) + 2.5*t(x, tk0, tk1)**3 * (x - tk0)
        return 6*t(x, tk0, tk1)**5 - 15*t(x, tk0, tk1)**4 \
                + 10*t(x, tk0, tk1)**3

    def _psi0(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return 1 - 18*t(x, tk0, tk1)**2 + 32*t(x, tk0, tk1)**3 \
                - 15*t(x, tk0, tk1)**4
        elif dtype == "integrate":
            return (tk1 - tk0) * (-0.5*t(x, tk0, tk1)**5 * (x - tk0) \
                    + 1.6*t(x, tk0, tk1)**4 * (x - tk0) - 1.5*t(x, tk0, tk1)**3 \
                    * (x - tk0) + x**2 / (2 *(tk1 - tk0)) - (tk0 * x) / (tk1 - tk0))
        return (tk1 - tk0) * (-3*t(x, tk0, tk1)**5 \
                + 8*t(x, tk0, tk1)**4 - 6*t(x, tk0, tk1)**3 \
                + t(x, tk0, tk1))

    def _psi1(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return -12*t(x, tk0, tk1)**2 + 28*t(x, tk0, tk1)**3 \
                - 15*t(x, tk0, tk1)**4
        elif dtype == "integrate":
            return (tk1 - tk0) * (-0.5*t(x, tk0, tk1)**5 * (x - tk0) \
                + 1.4*t(x, tk0, tk1)**4 * (x - tk0) - t(x, tk0, tk1)**3 \
                * (x - tk0))
        return (tk1 - tk0) * (-3*t(x, tk0, tk1)**5 \
                + 7*t(x, tk0, tk1)**4 - 4*t(x, tk0, tk1)**3)

    def _theta0(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return (tk1 - tk0) * (t(x, tk0, tk1) - 4.5*t(x, tk0, tk1)**2 \
                + 6*t(x, tk0, tk1)**3 - 2.5*t(x, tk0, tk1)**4)
        elif dtype == "integrate":
            return (tk1 - tk0)**2 * (-1/12*t(x, tk0, tk1)**5 * (x - tk0) \
                    + 0.3*t(x, tk0, tk1)**4 * (x - tk0) - 0.375*t(x, tk0, tk1)**3 \
                    * (x - tk0) + 1/6*t(x, tk0, tk1)**2 * (x - tk0))
        return (tk1 - tk0)**2 * (-0.5*t(x, tk0, tk1)**5 \
                + 1.5*t(x, tk0, tk1)**4 - 1.5*t(x, tk0, tk1)**3 \
                + 0.5*t(x, tk0, tk1)**2)

    def _theta1(x, tk0, tk1, dtype=None):
        if dtype == "derive":
            return (tk1 - tk0) * (1.5*t(x, tk0, tk1)**2 \
                - 4*t(x, tk0, tk1)**3 + 2.5*t(x, tk0, tk1)**4)
        elif dtype == "integrate":
            return (tk1 - tk0)**2 * (-1/12*t(x, tk0, tk1)**5 * (x - tk0) \
                    - 0.2*t(x, tk0, tk1)**4 * (x - tk0) + 0.125*t(x, tk0, tk1)**3 \
                    * (x - tk0))
        return (tk1 - tk0)**2 * (0.5*t(x, tk0, tk1)**5 \
                - t(x, tk0, tk1)**4 + 0.5*t(x, tk0, tk1)**3)

    def quintic_spline(x, yk0, yk1, ykp0, ykp1, ykpp0, ykpp1, tk0, tk1, dtype=None):
        return yk0 * QuinticHermite._phi0(x, tk0, tk1, dtype=dtype) \
                + yk1 * QuinticHermite._phi1(x, tk0, tk1, dtype=dtype) \
                + ykp0 * QuinticHermite._psi0(x, tk0, tk1, dtype=dtype) \
                + ykp1 * QuinticHermite._psi1(x, tk0, tk1, dtype=dtype) \
                + ykpp0 * QuinticHermite._theta0(x, tk0, tk1, dtype=dtype) \
                + ykpp1 * QuinticHermite._theta1(x, tk0, tk1, dtype=dtype)

if __name__ == "__main__":
    print("Additional Work\n","="*25)
    print(" [ ] Add basis function checks (graphs)")
    print(" [ ] Toy problem with known solution")
