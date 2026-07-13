########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for artificial scattering (as-SN) ray effect mitigation.
# Verifies the implementation against Frank et al. (2020),
# "Ray Effect Mitigation for the Discrete Ordinates Method Using
# Artificial Scattering", Nuclear Science and Engineering.
#
########################################################################

import numpy as np
import pytest
from scipy.special import erf

import ants
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
from ants.fixed2d import fixed_source
from ants.quadrature import artificial_scatter_matrix

########################################################################
# Helpers
########################################################################


def _kernel_value(mu, eps):
    """Evaluate s_eps(mu) per Frank et al. Eq. (6)."""
    erf_val = erf(2.0 / eps) if eps > 1e-15 else 1.0
    prefactor = 2.0 / (np.sqrt(np.pi) * eps * erf_val)
    return prefactor * np.exp(-((1.0 - mu) ** 2) / eps**2)


def _quadrature_xy(angles):
    quad = ants.angular_xy(angles)
    return quad.angle_x, quad.angle_y, quad.angle_w


########################################################################
# Kernel matrix M_as
########################################################################


@pytest.mark.smoke
@pytest.mark.math
def test_kernel_row_sums():
    """Per Eq. (17), each row sum equals sigma_as (particle conservation)."""
    for angles in [4, 8]:
        angle_x, angle_y, angle_w = _quadrature_xy(angles)
        for sigma_as in [0.5, 1.0, 5.0]:
            M_as = artificial_scatter_matrix(angle_x, angle_y, angle_w, sigma_as, 4.5)
            np.testing.assert_allclose(
                M_as.sum(axis=1),
                sigma_as,
                rtol=1e-12,
                err_msg=f"Row sums != sigma_as for angles={angles}",
            )


@pytest.mark.math
def test_kernel_formula_values():
    """M_as entries match manual computation of Eq. (6)."""
    angle_x, angle_y, angle_w = _quadrature_xy(4)
    sigma_as = 3.0
    beta = 4.5
    eps = beta / len(angle_x)

    M_as = artificial_scatter_matrix(angle_x, angle_y, angle_w, sigma_as, beta)

    angle_z = np.sqrt(1.0 - angle_x**2 - angle_y**2)
    dots = (
        np.outer(angle_x, angle_x)
        + np.outer(angle_y, angle_y)
        + np.outer(angle_z, angle_z)
    )
    S = _kernel_value(dots, eps)
    WS = S * angle_w[np.newaxis, :]
    c_n = WS.sum(axis=1, keepdims=True)
    M_expected = sigma_as * WS / c_n

    np.testing.assert_allclose(M_as, M_expected, rtol=1e-12)


@pytest.mark.math
def test_kernel_positivity_shape_dtype():
    angle_x, angle_y, angle_w = _quadrature_xy(4)
    N = len(angle_x)
    M_as = artificial_scatter_matrix(angle_x, angle_y, angle_w, 2.0, 4.5)
    assert M_as.shape == (N, N)
    assert M_as.dtype == np.float64
    assert np.all(M_as >= 0.0), "M_as contains negative entries"


@pytest.mark.math
def test_kernel_sigma_as_zero_returns_zero_matrix():
    angle_x, angle_y, angle_w = _quadrature_xy(4)
    M_as = artificial_scatter_matrix(angle_x, angle_y, angle_w, 0.0, 4.5)
    np.testing.assert_array_equal(M_as, 0.0)


@pytest.mark.math
def test_kernel_forward_peaked():
    """The kernel is forward peaked: self-scattering dominates each row."""
    angle_x, angle_y, angle_w = _quadrature_xy(8)
    M_as = artificial_scatter_matrix(angle_x, angle_y, angle_w, 1.0, 4.5)
    # Normalize out the quadrature weights so columns are comparable
    kernel = M_as / angle_w[np.newaxis, :]
    assert np.all(np.argmax(kernel, axis=1) == np.arange(len(angle_x)))


@pytest.mark.math
def test_kernel_uses_full_dot_product():
    """2D kernel uses Omega_q . Omega_p = x_q x_p + y_q y_p."""
    angle_x = np.array([0.5, 0.5])
    angle_w = np.array([0.5, 0.5])
    sigma_as, beta = 1.0, 4.5
    # Same x components, opposing y components (less similar directions)
    M_opposed = artificial_scatter_matrix(
        angle_x, np.array([0.3, -0.3]), angle_w, sigma_as, beta
    )
    # Same x and same y components (identical directions)
    M_aligned = artificial_scatter_matrix(
        angle_x, np.array([0.3, 0.3]), angle_w, sigma_as, beta
    )
    assert M_opposed[0, 1] < M_aligned[0, 1]


########################################################################
# as-SN solver (fixed2d.fixed_source with sigma_as > 0)
########################################################################


def _lattice_like_problem(cells, angles, sigma_as=0.0, angular=False, space_disc=2):
    """Central isotropic source in a pure absorber (prone to ray effects)."""
    length = 3.0
    edges = np.linspace(0, length, cells + 1)
    delta = np.repeat(length / cells, cells)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )
    external = np.zeros((cells, cells, 1, 1))
    center = slice(cells // 2 - 1, cells // 2 + 1)
    external[center, center] = 1.0
    sources = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta.copy(),
        delta_y=delta.copy(),
        geometry=3,
        space_disc=space_disc,
    )
    quadrature = ants.angular_xy(angles)
    solver = SolverData(angular=angular, sigma_as=sigma_as, beta_as=4.5)
    return mat_data, sources, geometry, quadrature, solver, edges


@pytest.mark.smoke
@pytest.mark.slab2d
def test_as_sn_finite_positive():
    """as-SN runs to completion with a finite, non-negative scalar flux.

    Uses the STEP discretization, which preserves positivity (diamond
    difference can go negative near a localized source even without
    artificial scattering).
    """
    problem = _lattice_like_problem(20, 4, sigma_as=5.0, space_disc=1)
    flux = fixed_source(*problem[:5])
    assert np.all(np.isfinite(flux)), "Flux contains NaN or Inf with sigma_as > 0"
    assert np.all(flux >= 0.0), "Flux is negative with sigma_as > 0"


@pytest.mark.slab2d
def test_as_sn_small_sigma_matches_standard():
    """As sigma_as -> 0 the as-SN solution approaches the standard solution."""
    problem_std = _lattice_like_problem(20, 4, sigma_as=0.0)
    flux_std = fixed_source(*problem_std[:5])

    problem_as = _lattice_like_problem(20, 4, sigma_as=1e-4)
    flux_as = fixed_source(*problem_as[:5])

    err = np.max(np.abs(flux_as - flux_std)) / np.max(flux_std)
    assert err < 1e-3, f"sigma_as=1e-4 deviates from standard by {err:.2e}"


@pytest.mark.slab2d
def test_as_sn_particle_conservation():
    """The artificial scattering term conserves particles (M_as row sums
    equal sigma_as, balancing the sigma_as out-scattering): in a nearly
    leakage-free pure absorber, total absorption must equal the total
    source. A normalization error in M_as would break this balance by
    O(sigma_as / sigma_t)."""
    cells, angles = 40, 4
    problem = _lattice_like_problem(cells, angles, sigma_as=5.0)
    mat_data, sources, geometry, quadrature, solver, edges = problem
    # Enlarge the domain to ~5 mean free paths from source to boundary
    # so leakage is negligible (~e^-5)
    geometry.delta_x = np.repeat(10.0 / cells, cells)
    geometry.delta_y = np.repeat(10.0 / cells, cells)

    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    cell_area = geometry.delta_x[0] * geometry.delta_y[0]
    # sigma_t = 1: absorption rate = sum(sigma_t * flux * area)
    absorption = np.sum(flux) * cell_area
    source_rate = np.sum(sources.external) * cell_area
    assert abs(absorption - source_rate) / source_rate < 2e-2


@pytest.mark.slab2d
def test_as_sn_mitigates_ray_effects():
    """Artificial scattering reduces the azimuthal oscillation of the flux
    along a circle around an isotropic source (Frank et al. Sec. IV)."""

    def azimuthal_variation(flux, edges, radius):
        centers = 0.5 * (edges[1:] + edges[:-1])
        mid = 0.5 * edges[-1]
        xx, yy = np.meshgrid(centers - mid, centers - mid, indexing="ij")
        rr = np.sqrt(xx**2 + yy**2)
        ring = (rr > radius * 0.9) & (rr < radius * 1.1)
        values = flux[..., 0][ring]
        return np.std(values) / np.mean(values)

    cells, angles, radius = 50, 4, 1.0

    problem_std = _lattice_like_problem(cells, angles, sigma_as=0.0)
    flux_std = fixed_source(*problem_std[:5])
    variation_std = azimuthal_variation(flux_std, problem_std[5], radius)

    problem_as = _lattice_like_problem(cells, angles, sigma_as=5.0)
    flux_as = fixed_source(*problem_as[:5])
    variation_as = azimuthal_variation(flux_as, problem_as[5], radius)

    assert variation_as < variation_std, (
        f"as-SN did not reduce ray effects: CoV {variation_as:.3f} "
        f"vs standard {variation_std:.3f}"
    )


@pytest.mark.slab2d
def test_as_sn_angular_flux_consistent():
    """The angular=True path returns an angular flux whose weighted sum
    reproduces the as-SN scalar flux."""
    problem_scalar = _lattice_like_problem(20, 4, sigma_as=5.0)
    scalar_flux = fixed_source(*problem_scalar[:5])

    problem_angular = _lattice_like_problem(20, 4, sigma_as=5.0, angular=True)
    angular_flux = fixed_source(*problem_angular[:5])
    quadrature = problem_angular[3]

    assert np.all(np.isfinite(angular_flux))
    collapsed = np.sum(angular_flux * quadrature.angle_w[None, None, :, None], axis=2)
    np.testing.assert_allclose(collapsed, scalar_flux, rtol=1e-5, atol=1e-10)
