########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test for utils.interp3d functions. Includes first and second
# derivative approximations, CubicHermite, and QuinticHermite splines.
#
########################################################################

import numpy as np
import pytest

from ants.utils import interp3d


# f(x,y,z) = 2*x^2*y^3*z + 2  on [0,2]^3
def _function_01(x, y, z, mult=1):
    psi = 2 * x**2 * y**3 * z + 2
    if mult == 1:
        d_dx = 4 * x * y**3 * z
        d_dy = 6 * x**2 * y**2 * z
        d_dz = 2 * x**2 * y**3
        return psi, d_dx, d_dy, d_dz
    elif mult == 2:
        d2_dx2 = 4 * y**3 * z
        d2_dy2 = 12 * x**2 * y * z
        d2_dz2 = np.zeros_like(x)
        d2_dxdy = 12 * x * y**2 * z
        d2_dxdz = 4 * x * y**3
        d2_dydz = 6 * x**2 * y**2
        return psi, d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz
    # mult == 4: return analytical integrals over [0,2]^3
    int_psi = 176 / 3.0  # int 2x^2 y^3 z + 2 over [0,2]^3
    int_dx = 64.0  # int 4xy^3z
    int_dy = 256 / 3.0  # int 6x^2 y^2 z
    int_dz = 128 / 3.0  # int 2x^2 y^3
    return psi, int_psi, int_dx, int_dy, int_dz


# f(x,y,z) = 2 + sin(x)*cos(y)*sin(z)  on [0,pi]^3
def _function_02(x, y, z, mult=1):
    psi = 2 + np.sin(x) * np.cos(y) * np.sin(z)
    if mult == 1:
        d_dx = np.cos(x) * np.cos(y) * np.sin(z)
        d_dy = -np.sin(x) * np.sin(y) * np.sin(z)
        d_dz = np.sin(x) * np.cos(y) * np.cos(z)
        return psi, d_dx, d_dy, d_dz
    elif mult == 2:
        d2_dx2 = -np.sin(x) * np.cos(y) * np.sin(z)
        d2_dy2 = -np.sin(x) * np.cos(y) * np.sin(z)
        d2_dz2 = -np.sin(x) * np.cos(y) * np.sin(z)
        d2_dxdy = -np.cos(x) * np.sin(y) * np.sin(z)
        d2_dxdz = np.cos(x) * np.cos(y) * np.cos(z)
        d2_dydz = -np.sin(x) * np.sin(y) * np.cos(z)
        return psi, d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz
    # mult == 4: integrals over [0,pi]^3
    int_psi = 2 * np.pi**3  # trig cross-term integrates to 0
    int_dx = 0.0
    int_dy = -8.0
    int_dz = 0.0
    return psi, int_psi, int_dx, int_dy, int_dz


def _error(approx, reference):
    assert approx.shape == reference.shape
    N = approx.size
    return N ** (-0.5) * np.linalg.norm(approx - reference)


def _int_converged(err, tol=5e-4):
    """True if errors converged: either Wynn-epsilon < tol or already at float noise."""
    if np.max(err) < 1e-10:
        return True
    return _wynn_epsilon(err, 2) < tol


def _wynn_epsilon(lst, rank):
    N = 2 * rank + 1
    error = np.zeros((N + 1, N + 1))
    for ii in range(1, N + 1):
        error[ii, 1] = lst[ii - 1]
    for ii in range(3, N + 2):
        for jj in range(3, ii + 1):
            if (error[ii - 1, jj - 2] - error[ii - 2, jj - 2]) == 0.0:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3]
            else:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3] + 1 / (
                    error[ii - 1, jj - 2] - error[ii - 2, jj - 2]
                )
    return abs(error[-1, -1])


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize("function", [_function_01, _function_02])
def test_first_derivative(function):
    bound = 2.0 if function == _function_01 else np.pi
    # Larger grids needed for 3D Wynn-epsilon to converge; first_derivative is fast
    cells = np.array([20, 30, 40, 60, 80])
    err_dx = np.zeros(cells.shape)
    err_dy = np.zeros(cells.shape)
    err_dz = np.zeros(cells.shape)
    for cc, ii in enumerate(cells):
        ex = np.linspace(0, bound, ii + 1)
        ey = ex.copy()
        ez = ex.copy()
        mx, my, mz = np.meshgrid(ex, ey, ez, indexing="ij")
        psi, d_dx, d_dy, d_dz = function(mx, my, mz, mult=1)
        a_dx, a_dy, a_dz = interp3d.first_derivative(psi, ex, ey, ez)
        err_dx[cc] = _error(a_dx, d_dx)
        err_dy[cc] = _error(a_dy, d_dy)
        err_dz[cc] = _error(a_dz, d_dz)
    assert _wynn_epsilon(err_dx, 2) < 1e-2, "d/dx not converged"
    assert _wynn_epsilon(err_dy, 2) < 1e-2, "d/dy not converged"
    assert _wynn_epsilon(err_dz, 2) < 1e-2, "d/dz not converged"


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize("function", [_function_01, _function_02])
def test_second_derivative(function):
    bound = 2.0 if function == _function_01 else np.pi
    cells = np.array([20, 30, 40, 60, 80])
    err_d2_dx2 = np.zeros(cells.shape)
    err_d2_dy2 = np.zeros(cells.shape)
    err_d2_dz2 = np.zeros(cells.shape)
    err_d2_dxdy = np.zeros(cells.shape)
    err_d2_dxdz = np.zeros(cells.shape)
    err_d2_dydz = np.zeros(cells.shape)
    for cc, ii in enumerate(cells):
        ex = np.linspace(0, bound, ii + 1)
        ey = ex.copy()
        ez = ex.copy()
        mx, my, mz = np.meshgrid(ex, ey, ez, indexing="ij")
        psi, d2_dx2, d2_dy2, d2_dz2, d2_dxdy, d2_dxdz, d2_dydz = function(
            mx, my, mz, mult=2
        )
        a_dx2, a_dy2, a_dz2, a_dxdy, a_dxdz, a_dydz = interp3d.second_derivative(
            psi, ex, ey, ez
        )
        err_d2_dx2[cc] = _error(a_dx2, d2_dx2)
        err_d2_dy2[cc] = _error(a_dy2, d2_dy2)
        err_d2_dz2[cc] = _error(a_dz2, d2_dz2)
        err_d2_dxdy[cc] = _error(a_dxdy, d2_dxdy)
        err_d2_dxdz[cc] = _error(a_dxdz, d2_dxdz)
        err_d2_dydz[cc] = _error(a_dydz, d2_dydz)
    assert _wynn_epsilon(err_d2_dx2, 2) < 1e-2, "d2/dx2 not converged"
    assert _wynn_epsilon(err_d2_dy2, 2) < 1e-2, "d2/dy2 not converged"
    assert _wynn_epsilon(err_d2_dz2, 2) < 1e-2, "d2/dz2 not converged"
    assert _wynn_epsilon(err_d2_dxdy, 2) < 1e-2, "d2/dxdy not converged"
    assert _wynn_epsilon(err_d2_dxdz, 2) < 1e-2, "d2/dxdz not converged"
    assert _wynn_epsilon(err_d2_dydz, 2) < 1e-2, "d2/dydz not converged"


@pytest.mark.math
@pytest.mark.parametrize(
    ("function", "edges"),
    [(_function_01, 0), (_function_01, 1), (_function_02, 0), (_function_02, 1)],
)
def test_cubic_hermite_integrate(function, edges):
    bound = 2.0 if function == _function_01 else np.pi
    # Larger cells are needed for the trig function to reach Wynn-epsilon plateau
    cells = np.array([6, 8, 10, 14, 18])
    err_psi = np.zeros(cells.shape)
    err_dx = np.zeros(cells.shape)
    err_dy = np.zeros(cells.shape)
    err_dz = np.zeros(cells.shape)
    for cc, ii in enumerate(cells):
        ex = np.linspace(0, bound, ii + 1)
        ey = ex.copy()
        ez = ex.copy()
        mx, my, mz = np.meshgrid(ex, ey, ez, indexing="ij")
        psi, int_psi, int_dx, int_dy, int_dz = function(mx, my, mz, mult=4)
        splines = interp3d.CubicHermite(psi, ex, ey, ez)
        if edges:
            a_psi, a_dx, a_dy, a_dz = splines.integrate_edges()
        else:
            a_psi, a_dx, a_dy, a_dz = splines.integrate_centers(ex, ey, ez)
        err_psi[cc] = abs(np.sum(a_psi) - int_psi)
        err_dx[cc] = abs(np.sum(a_dx) - int_dx)
        err_dy[cc] = abs(np.sum(a_dy) - int_dy)
        err_dz[cc] = abs(np.sum(a_dz) - int_dz)
    assert _int_converged(err_psi), "int_psi not converged"
    assert _int_converged(err_dx), "int_dx not converged"
    assert _int_converged(err_dy), "int_dy not converged"
    assert _int_converged(err_dz), "int_dz not converged"


@pytest.mark.math
@pytest.mark.parametrize(
    ("function", "edges"),
    [(_function_01, 0), (_function_01, 1), (_function_02, 0), (_function_02, 1)],
)
def test_quintic_hermite_integrate(function, edges):
    bound = 2.0 if function == _function_01 else np.pi
    cells = np.array([6, 8, 10, 14, 18])
    err_psi = np.zeros(cells.shape)
    err_dx = np.zeros(cells.shape)
    err_dy = np.zeros(cells.shape)
    err_dz = np.zeros(cells.shape)
    for cc, ii in enumerate(cells):
        ex = np.linspace(0, bound, ii + 1)
        ey = ex.copy()
        ez = ex.copy()
        mx, my, mz = np.meshgrid(ex, ey, ez, indexing="ij")
        psi, int_psi, int_dx, int_dy, int_dz = function(mx, my, mz, mult=4)
        splines = interp3d.QuinticHermite(psi, ex, ey, ez)
        if edges:
            a_psi, a_dx, a_dy, a_dz = splines.integrate_edges()
        else:
            a_psi, a_dx, a_dy, a_dz = splines.integrate_centers(ex, ey, ez)
        err_psi[cc] = abs(np.sum(a_psi) - int_psi)
        err_dx[cc] = abs(np.sum(a_dx) - int_dx)
        err_dy[cc] = abs(np.sum(a_dy) - int_dy)
        err_dz[cc] = abs(np.sum(a_dz) - int_dz)
    assert _int_converged(err_psi), "int_psi not converged"
    assert _int_converged(err_dx), "int_dx not converged"
    assert _int_converged(err_dy), "int_dy not converged"
    assert _int_converged(err_dz), "int_dz not converged"


def test_interpolate_at_knots_reproduces_values():
    """CubicHermite and QuinticHermite should reproduce psi exactly at knots."""
    x = np.linspace(0.0, 2.0, 6)
    y = np.linspace(0.0, 2.0, 6)
    z = np.linspace(0.0, 2.0, 6)
    mx, my, mz = np.meshgrid(x, y, z, indexing="ij")
    psi = 2 * mx**2 * my**3 * mz + 2
    ch = interp3d.CubicHermite(psi, x, y, z)
    qh = interp3d.QuinticHermite(psi, x, y, z)
    np.testing.assert_allclose(ch.interpolate(x, y, z), psi, rtol=0, atol=1e-10)
    np.testing.assert_allclose(qh.interpolate(x, y, z), psi, rtol=0, atol=1e-10)


def test_interpolation_wrapper_selects_correct_instance():
    """Interpolation wrapper should construct the correct backend instance."""
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    z = np.linspace(0.0, 1.0, 5)
    mx, my, mz = np.meshgrid(x, y, z, indexing="ij")
    psi = mx**2 + my**2 + mz**2
    splits = np.array([0, 5])
    w = interp3d.Interpolation(
        psi,
        x,
        y,
        z,
        medium_map=None,
        x_splits=splits,
        y_splits=splits,
        z_splits=splits,
        block=False,
        quintic=False,
    )
    assert isinstance(w.instance, interp3d.CubicHermite)
    w2 = interp3d.Interpolation(
        psi,
        x,
        y,
        z,
        medium_map=None,
        x_splits=splits,
        y_splits=splits,
        z_splits=splits,
        block=False,
        quintic=True,
    )
    assert isinstance(w2.instance, interp3d.QuinticHermite)
    w3 = interp3d.Interpolation(
        psi,
        x,
        y,
        z,
        medium_map=None,
        x_splits=splits,
        y_splits=splits,
        z_splits=splits,
        block=True,
        quintic=False,
    )
    assert isinstance(w3.instance, interp3d.BlockInterpolation)
