########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test for utils.interp2d functions. Includes the first and second
# derivative approximation (second order), Cubic Hermite splines,
# and Quintic Hermite splines functions.
#
########################################################################

import pytest
import numpy as np

from ants.utils import interp2d

def _function_01(x, y, mult=1):
    # On the interval x \in [0, 2], y \in [0, 2]
    psi = 2 * x**2 * y**3 + 2
    if mult == 1:
        d_dx = 4 * x * y**3
        d_dy = 6 * x**2 * y**2
        return psi, d_dx, d_dy
    elif mult == 2:
        d2_dx2 = 4 * y**3
        d2_dy2 = 12 * x**2 * y
        d2_dxdy = 12 * x * y**2
        return psi, d2_dx2, d2_dy2, d2_dxdy
    elif mult == 3:
        d3_dx2dy = 12 * y**2
        d3_dxdy2 = 24 * x * y
        d4_dx2dy2 = 24 * y
        return psi, d3_dx2dy, d3_dxdy2, d4_dx2dy2
    int_psi = 88 / 3.
    int_dxpsi = 32.
    int_dypsi = 128 / 3.
    return psi, int_psi, int_dxpsi, int_dypsi


def _function_02(x, y, mult=1):
    # On the interval x \in [0, pi], y \in [0, pi]
    psi = 2 + np.sin(x) * np.cos(y)
    if mult == 1:
        d_dx = np.cos(x) * np.cos(y)
        d_dy = -np.sin(x) * np.sin(y)
        return psi, d_dx, d_dy
    elif mult == 2:
        d2_dx2 = -np.sin(x) * np.cos(y)
        d2_dy2 = -np.sin(x) * np.cos(y)
        d2_dxdy = -np.cos(x) * np.sin(y)
        return psi, d2_dx2, d2_dy2, d2_dxdy
    elif mult == 3:
        d3_dx2dy = np.sin(x) * np.sin(y)
        d3_dxdy2 = -np.cos(x) * np.cos(y)
        d4_dx2dy2 = np.sin(x) * np.cos(y)
        return psi, d3_dx2dy, d3_dxdy2, d4_dx2dy2
    int_psi = 2 * np.pi**2
    int_dxpsi = 0.
    int_dypsi = -4.
    return psi, int_psi, int_dxpsi, int_dypsi


def _error(approx, reference):
    assert approx.shape == reference.shape, "Not the same array shape"
    cells_x = approx.shape[0]
    return cells_x**(-0.5) * np.linalg.norm(approx - reference)


def _wynn_epsilon(lst, rank):
    """ Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        lst: list of values for convergence
        rank: rank of system
    Returns:
        2D Array where diagonal is convergence """
    N = 2 * rank + 1
    error = np.zeros((N + 1, N + 1))
    for ii in range(1, N + 1):
        error[ii, 1] = lst[ii - 1]
    for ii in range(3, N + 2):
        for jj in range(3, ii + 1):
            if (error[ii-1,jj-2] - error[ii-2,jj-2]) == 0.0:
                error[ii-1,jj-1] = error[ii-2,jj-3]
            else:
                error[ii-1,jj-1] = error[ii-2,jj-3] \
                            + 1 / (error[ii-1,jj-2] - error[ii-2,jj-2])
    return abs(error[-1,-1])


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize(("function"), [_function_01, _function_02])
def test_first_derivative(function):
    # Set interval
    bound = 2 if function == _function_01 else np.pi
    # Create differing grids
    cells_x = np.array([40, 80, 160, 320, 640])
    errors_d_dx = np.zeros(cells_x.shape)
    errors_d_dy = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create mesh
        edges_x = np.linspace(0, bound, ii + 1)
        edges_y = edges_x.copy()
        mesh_x, mesh_y = np.meshgrid(edges_x, edges_y, indexing="ij")
        # Find analytical solution
        psi, d_dx, d_dy = function(mesh_x, mesh_y, mult=1)
        approx_d_dx, approx_d_dy = interp2d.first_derivative(psi, edges_x, edges_y)
        errors_d_dx[cc] = _error(approx_d_dx, d_dx)
        errors_d_dy[cc] = _error(approx_d_dy, d_dy)
    convergence_d_dx = _wynn_epsilon(errors_d_dx, 2)
    convergence_d_dy = _wynn_epsilon(errors_d_dy, 2)
    assert convergence_d_dx < 1e-6, "d/dx not converged correctly"
    assert convergence_d_dy < 1e-6, "d/dy not converged correctly"


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize(("function"), [_function_01, _function_02])
def test_second_derivative(function):
    # Set interval
    bound = 2 if function == _function_01 else np.pi
    # Create differing grids
    cells_x = np.array([40, 80, 160, 320, 640])
    errors_d2_dx2 = np.zeros(cells_x.shape)
    errors_d2_dy2 = np.zeros(cells_x.shape)
    errors_d2_dxdy = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create mesh
        edges_x = np.linspace(0, bound, ii + 1)
        edges_y = edges_x.copy()
        mesh_x, mesh_y = np.meshgrid(edges_x, edges_y, indexing="ij")
        # Find analytical solution
        psi, d2_dx2, d2_dy2, d2_dxdy = function(mesh_x, mesh_y, mult=2)
        approx_d2_dx2, approx_d2_dxdy, approx_d2_dy2 \
                    = interp2d.second_derivative(psi, edges_x, edges_y)
        errors_d2_dx2[cc] = _error(approx_d2_dx2, d2_dx2)
        errors_d2_dy2[cc] = _error(approx_d2_dy2, d2_dy2)
        errors_d2_dxdy[cc] = _error(approx_d2_dxdy, d2_dxdy)
    # Find convergence
    convergence_d2_dx2 = _wynn_epsilon(errors_d2_dx2, 2)
    convergence_d2_dy2 = _wynn_epsilon(errors_d2_dy2, 2)
    convergence_d2_dxdy = _wynn_epsilon(errors_d2_dxdy, 2)
    assert convergence_d2_dx2 < 5e-6, "d2/dx2 not converged correctly"
    assert convergence_d2_dy2 < 5e-6, "d2/dy2 not converged correctly"
    assert convergence_d2_dxdy < 5e-6, "d2/dxdy not converged correctly"


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize(("function"), [_function_01, _function_02])
def test_higher_order_derivative(function):
    # Set interval
    bound = 2 if function == _function_01 else np.pi
    # Create differing grids
    cells_x = np.array([40, 80, 160, 320, 640])
    errors_d3_dx2dy = np.zeros(cells_x.shape)
    errors_d3_dxdy2 = np.zeros(cells_x.shape)
    errors_d4_dx2dy2 = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create mesh
        edges_x = np.linspace(0, bound, ii + 1)
        edges_y = edges_x.copy()
        mesh_x, mesh_y = np.meshgrid(edges_x, edges_y, indexing="ij")
        # Find analytical solution
        psi, d3_dx2dy, d3_dxdy2, d4_dx2dy2 = function(mesh_x, mesh_y, mult=3)
        approx_d3_dx2dy, approx_d3_dxdy2, approx_d4_dx2dy2 \
                = interp2d.higher_order_derivative(psi, edges_x, edges_y)
        errors_d3_dx2dy[cc] = _error(approx_d3_dx2dy, d3_dx2dy)
        errors_d3_dxdy2[cc] = _error(approx_d3_dxdy2, d3_dxdy2)
        errors_d4_dx2dy2[cc] = _error(approx_d4_dx2dy2, d4_dx2dy2)
    convergence_d3_dx2dy = _wynn_epsilon(errors_d3_dx2dy, 2)
    convergence_d3_dxdy2 = _wynn_epsilon(errors_d3_dxdy2, 2)
    convergence_d4_dx2dy2 = _wynn_epsilon(errors_d4_dx2dy2, 2)
    assert convergence_d3_dx2dy < 5e-6, "d3/dx2dy not converged correctly"
    assert convergence_d3_dxdy2 < 5e-6, "d3/dxdy2 not converged correctly"
    assert convergence_d4_dx2dy2 < 5e-6, "d4/dx2dy2 not converged correctly"


@pytest.mark.math
@pytest.mark.parametrize(("function", "edges"), [(_function_01, 0), \
            (_function_01, 1), (_function_02, 0), (_function_02, 1)])
def test_cubic_hermite_integrate(function, edges):
    # Set interval
    bound = 2 if function == _function_01 else np.pi
    # Create differing grids
    cells_x = np.array([20, 40, 80, 160, 320])
    errors_int_psi = np.zeros(cells_x.shape)
    errors_int_dxpsi = np.zeros(cells_x.shape)
    errors_int_dypsi = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create mesh
        edges_x = np.linspace(0, bound, ii + 1)
        edges_y = edges_x.copy()
        mesh_x, mesh_y = np.meshgrid(edges_x, edges_y, indexing="ij")
        # Find analytical solution
        psi, int_psi, int_dxpsi, int_dypsi = function(mesh_x, mesh_y, mult=4)
        # Get splines
        splines = interp2d.CubicHermite(psi, edges_x, edges_y)
        if edges:
            approx_int_psi, approx_int_dxpsi, approx_int_dypsi \
                    = splines.integrate_edges()
        else:
            approx_int_psi, approx_int_dxpsi, approx_int_dypsi \
                    = splines.integrate_centers(edges_x, edges_y)
        # Get absolute error
        errors_int_psi[cc] = abs(np.sum(approx_int_psi) - int_psi)
        errors_int_dxpsi[cc] = abs(np.sum(approx_int_dxpsi) - int_dxpsi)
        errors_int_dypsi[cc] = abs(np.sum(approx_int_dypsi) - int_dypsi)
    # Get convergence
    convergence_int_psi = _wynn_epsilon(errors_int_psi, 2)
    convergence_int_dxpsi = _wynn_epsilon(errors_int_dxpsi, 2)
    convergence_int_dypsi = _wynn_epsilon(errors_int_dypsi, 2)
    assert convergence_int_psi < 1e-6, "int_psi not converged correctly"
    assert convergence_int_dxpsi < 1e-6, "int_dxpsi not converged correctly"
    assert convergence_int_dypsi < 1e-6, "int_dypsi not converged correctly"


@pytest.mark.math
@pytest.mark.parametrize(("function", "edges"), [(_function_01, 0), \
            (_function_01, 1), (_function_02, 0), (_function_02, 1)])
def test_quintic_hermite_integrate(function, edges):
    # Set interval
    bound = 2 if function == _function_01 else np.pi
    # Create differing grids
    cells_x = np.array([20, 40, 80, 160, 320])
    errors_int_psi = np.zeros(cells_x.shape)
    errors_int_dxpsi = np.zeros(cells_x.shape)
    errors_int_dypsi = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create mesh
        edges_x = np.linspace(0, bound, ii + 1)
        edges_y = edges_x.copy()
        mesh_x, mesh_y = np.meshgrid(edges_x, edges_y, indexing="ij")
        # Find analytical solution
        psi, int_psi, int_dxpsi, int_dypsi = function(mesh_x, mesh_y, mult=4)
        # Get splines
        splines = interp2d.QuinticHermite(psi, edges_x, edges_y)
        if edges:
            approx_int_psi, approx_int_dxpsi, approx_int_dypsi \
                    = splines.integrate_edges()
        else:
            approx_int_psi, approx_int_dxpsi, approx_int_dypsi \
                    = splines.integrate_centers(edges_x, edges_y)
        # Get absolute error
        errors_int_psi[cc] = abs(np.sum(approx_int_psi) - int_psi)
        errors_int_dxpsi[cc] = abs(np.sum(approx_int_dxpsi) - int_dxpsi)
        errors_int_dypsi[cc] = abs(np.sum(approx_int_dypsi) - int_dypsi)
    # Get convergence
    convergence_int_psi = _wynn_epsilon(errors_int_psi, 2)
    convergence_int_dxpsi = _wynn_epsilon(errors_int_dxpsi, 2)
    convergence_int_dypsi = _wynn_epsilon(errors_int_dypsi, 2)
    assert convergence_int_psi < 1e-6, "int_psi not converged correctly"
    assert convergence_int_dxpsi < 1e-6, "int_dxpsi not converged correctly"
    assert convergence_int_dypsi < 1e-6, "int_dypsi not converged correctly"
