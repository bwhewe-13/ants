########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test for utils.interp1d functions. Includes the first and second
# derivative approximation (second order), Cubic Hermite splines,
# and Quintic Hermite splines functions.
#
########################################################################

import numpy as np
import pytest

from ants.utils import interp1d


def _function_01(x):
    # On the interval [0, 2]
    psi = x**3 - 2 * x**2 + 2
    dpsi_dx = 3 * x**2 - 4 * x
    d2psi_dx2 = 6 * x - 4
    # int_psi = 0.25 * x**4 - 2/3. * x**3 + 2 * x
    # int_dpsi = x**3 - 2 * x**2
    int_psi = 0.25 * 2**4 - 2 / 3.0 * 2**3 + 2 * 2
    int_dpsi = 2**3 - 2 * 2**2
    return psi, dpsi_dx, d2psi_dx2, int_psi, int_dpsi


def _function_02(x):
    # On the interval [0, 2 pi]
    psi = np.sin(x) ** 2 + 2
    dpsi_dx = 2 * np.sin(x) * np.cos(x)
    d2psi_dx2 = 2 * (np.cos(x) ** 2 - np.sin(x) ** 2)
    # int_psi = 2.5 * x - 0.25 * np.sin(2 * x)
    # int_dpsi = -0.5 * np.cos(2 * x)
    int_psi = 5 * np.pi - 0.25 * np.sin(4 * np.pi)
    int_dpsi = -0.5 * np.cos(4 * np.pi) + 0.5
    return psi, dpsi_dx, d2psi_dx2, int_psi, int_dpsi


def _error(approx, reference):
    assert approx.shape == reference.shape, "Not the same array shape"
    cells_x = approx.shape[0]
    return cells_x ** (-0.5) * np.linalg.norm(approx - reference)


def _wynn_epsilon(lst, rank):
    """Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        lst: list of values for convergence
        rank: rank of system
    Returns:
        2D Array where diagonal is convergence"""
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
@pytest.mark.parametrize(("function"), [_function_01, _function_02])
def test_first_derivative(function):
    # Set interval
    bound = 2 if function == _function_01 else 2 * np.pi
    # Create differing grids
    cells_x = np.array([10, 100, 1_000, 10_000, 20_000])
    errors = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        interval = np.linspace(0, bound, ii + 1)
        psi, dpsi_dx, _, _, _ = function(interval)
        approx = interp1d.first_derivative(psi, interval)
        errors[cc] = _error(approx, dpsi_dx)
    convergence = _wynn_epsilon(errors, 2)
    assert convergence < 1e-6, "Not converged correctly"


@pytest.mark.smoke
@pytest.mark.math
@pytest.mark.parametrize(("function"), [_function_01, _function_02])
def test_second_derivative(function):
    # Set interval
    bound = 2 if function == _function_01 else 2 * np.pi
    # Create differing grids
    cells_x = np.array([100, 1_000, 10_000, 100_000, 200_000])
    errors = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        interval = np.linspace(0, bound, ii + 1)
        psi, _, d2psi_dx2, _, _ = function(interval)
        approx = interp1d.second_derivative(psi, interval)
        errors[cc] = _error(approx, d2psi_dx2)
    convergence = _wynn_epsilon(errors, 2)
    assert convergence < 1e-6, "Not converged correctly"


@pytest.mark.math
@pytest.mark.parametrize(
    ("function", "edges"),
    [(_function_01, 0), (_function_01, 1), (_function_02, 0), (_function_02, 1)],
)
def test_cubic_hermite_integrate(function, edges):
    # Set interval
    bound = 2 if function == _function_01 else 2 * np.pi
    # Create differing grids
    cells_x = np.array([10, 100, 1_000, 10_000, 20_000])
    errors_int_psi = np.zeros(cells_x.shape)
    errors_int_dpsi = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create spatial grid
        edges_x = np.linspace(0, bound, ii + 1)
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        interval = edges_x.copy() if edges else centers_x.copy()
        # Get reference solution and
        psi, _, _, int_psi, int_dpsi = function(interval)
        # Get splines
        splines = interp1d.CubicHermite(psi, interval)
        if edges:
            approx_int_psi, approx_int_dpsi = splines.integrate_edges()
        else:
            approx_int_psi, approx_int_dpsi = splines.integrate_centers(edges_x)
        # Get absolute error
        errors_int_psi[cc] = abs(np.sum(approx_int_psi) - int_psi)
        errors_int_dpsi[cc] = abs(np.sum(approx_int_dpsi) - int_dpsi)
    # Get convergence
    convergence_int_psi = _wynn_epsilon(errors_int_psi, 2)
    convergence_int_dpsi = _wynn_epsilon(errors_int_dpsi, 2)
    assert convergence_int_psi < 1e-6, "int_psi not converged correctly"
    assert convergence_int_dpsi < 1e-6, "int_dpsi not converged correctly"


@pytest.mark.math
@pytest.mark.parametrize(
    ("function", "edges"),
    [(_function_01, 0), (_function_01, 1), (_function_02, 0), (_function_02, 1)],
)
def test_quintic_hermite_integrate(function, edges):
    # Set interval
    bound = 2 if function == _function_01 else 2 * np.pi
    # Create differing grids
    cells_x = np.array([10, 100, 1_000, 10_000, 20_000])
    errors_int_psi = np.zeros(cells_x.shape)
    errors_int_dpsi = np.zeros(cells_x.shape)
    # Iterate over cell numbers
    for cc, ii in enumerate(cells_x):
        # Create spatial grid
        edges_x = np.linspace(0, bound, ii + 1)
        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        interval = edges_x.copy() if edges else centers_x.copy()
        # Get reference solution and
        psi, _, _, int_psi, int_dpsi = function(interval)
        # Get splines
        splines = interp1d.QuinticHermite(psi, interval)
        if edges:
            approx_int_psi, approx_int_dpsi = splines.integrate_edges()
        else:
            approx_int_psi, approx_int_dpsi = splines.integrate_centers(edges_x)
        # Get absolute error
        errors_int_psi[cc] = abs(np.sum(approx_int_psi) - int_psi)
        errors_int_dpsi[cc] = abs(np.sum(approx_int_dpsi) - int_dpsi)
    # Get convergence
    convergence_int_psi = _wynn_epsilon(errors_int_psi, 2)
    convergence_int_dpsi = _wynn_epsilon(errors_int_dpsi, 2)
    assert convergence_int_psi < 1e-6, "int_psi not converged correctly"
    assert convergence_int_dpsi < 1e-6, "int_dpsi not converged correctly"


def test_interpolate_at_knots_reproduces_values():
    """Cubic and quintic interpolators should reproduce values at knots.

    This is a basic sanity check: when evaluating at the original knot
    coordinates, the spline should return the same psi values (within
    floating point tolerance).
    """
    x = np.linspace(0.0, 2.0, 11)
    psi = x**3 - 2 * x**2 + 2
    ch = interp1d.CubicHermite(psi, x)
    qh = interp1d.QuinticHermite(psi, x)
    np.testing.assert_allclose(ch.interpolate(x), psi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(qh.interpolate(x), psi, rtol=0, atol=1e-10)


def test_block_interpolation_simple():
    """BlockInterpolation should apply different splines on blocks.

    Build a simple two-block problem where psi is linear on each block.
    Interpolating inside a block should return the linear values.
    """
    # Global knots and piecewise-linear psi (two blocks)
    knots = np.linspace(0.0, 2.5, 6)  # 5 intervals
    # psi: first two intervals follow y=x, last three follow y=1+x
    psi = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])
    # x_splits indices: two blocks [0:3] and [3:6]
    x_splits = np.array([0, 3, 6])
    blk = interp1d.BlockInterpolation(interp1d.CubicHermite, psi, knots, None, x_splits)
    # Evaluate at knot centers and midpoints
    query = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    vals = blk.interpolate(query)
    # Expected by linear interpolation of the two piecewise segments
    expected = np.array([0.25, 0.75, 1.25, 2.25, 2.75])
    assert vals.shape == expected.shape
    np.testing.assert_allclose(vals, expected, atol=5e-2)


def test_interpolation_wrapper_selects_correct_instance():
    """Ensure Interpolation wrapper constructs correct backend instance."""
    x = np.linspace(0.0, 1.0, 5)
    psi = x**2
    # Block + quintic => BlockInterpolation with QuinticHermite
    wrapper = interp1d.Interpolation(
        psi,
        x,
        medium_map=np.zeros((len(x) - 1,)),
        x_splits=np.array([]),
        block=True,
        quintic=True,
    )
    assert hasattr(wrapper.instance, "interpolate")
    # Non-block, cubic => CubicHermite
    wrapper2 = interp1d.Interpolation(
        psi, x, medium_map=None, x_splits=np.array([]), block=False, quintic=False
    )
    assert isinstance(wrapper2.instance, interp1d.CubicHermite)


if __name__ == "__main__":
    knots = np.linspace(0.0, 2.5, 6)  # 4 intervals
    # psi: first two intervals follow y=x, last two follow y=2+x
    psi = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])
    # x_splits indices: split after index 3 (so blocks [0:3] and [3:5])
    x_splits = np.array([0, 3, 6])
    blk = interp1d.BlockInterpolation(interp1d.CubicHermite, psi, knots, None, x_splits)
    # Evaluate at knot centers and midpoints
    query = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    vals = blk.interpolate(query)
    # Expected by linear interpolation of the two piecewise segments
    expected = np.array([0.25, 0.75, 1.25, 2.25, 2.75])
    assert vals.shape == expected.shape
    np.testing.assert_allclose(vals, expected, atol=5e-2)
