########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Anisotropic scattering for three-dimensional fixed source problems
#
########################################################################

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

import ants
from ants.datatypes import Geometry, GeometryData, SolverData, SourceData
from ants.fixed2d import fixed_source as fixed_source_2d
from ants.fixed3d import fixed_source
from tests import problems3d as prob

KWARGS = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)


def with_moments(scatter, ratios):
    # Legendre moments that are a ratio of the isotropic scatter
    return np.stack([ratio * scatter for ratio in ratios], axis=-1)


########################################################################
# Spherical harmonics
########################################################################


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_harmonic_degrees_full_sphere():
    ell = ants.harmonic_degrees(3, full_sphere=True)
    assert np.array_equal(ell, [0, 1, 1, 1, 2, 2, 2, 2, 2])


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_spherical_harmonics_addition_3d():
    # Summing over m gives P_l(a . b) for the full set of harmonics
    quadrature = ants.angular_xyz(6)
    angle_x, angle_y, angle_z = (
        quadrature.angle_x,
        quadrature.angle_y,
        quadrature.angle_z,
    )
    Y = ants.spherical_harmonics(5, angle_x, angle_y, angle_z)
    ell = ants.harmonic_degrees(5, full_sphere=True)
    dot = np.outer(angle_x, angle_x) + np.outer(angle_y, angle_y)
    dot += np.outer(angle_z, angle_z)
    for ll in range(5):
        coefs = np.zeros((ll + 1,))
        coefs[ll] = 1.0
        assert np.allclose(Y[ell == ll].T @ Y[ell == ll], legval(dot, coefs))


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_harmonic_data_3d():
    quadrature = ants.angular_xyz(8)
    P_source, P_weights = ants.harmonic_data(quadrature, 3)
    ell = ants.harmonic_degrees(3, full_sphere=True)
    assert quadrature.P.shape == (9, 128)
    assert np.allclose(P_weights, quadrature.angle_w[None, :] * quadrature.P)
    assert np.allclose(P_source, (2 * ell + 1)[:, None] * quadrature.P)
    # Product quadrature: harmonics are orthogonal with weight 1 / (2l + 1)
    gram = P_weights @ quadrature.P.T
    assert np.allclose(gram, np.diag(1.0 / (2 * ell + 1)))


########################################################################
# Fixed source problems
########################################################################


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.anisotropic
@pytest.mark.parametrize(
    "solver_kwargs",
    [
        {},
        {"angular": True},
        {"mg_solver": 2},
        {"parallel": 2, "num_threads": 3},
        {"parallel": 3, "num_threads": 2},
    ],
)
def test_fixed_source_zero_moments(solver_kwargs):
    # Zero higher moments give the isotropic solution for each solver
    groups = 4
    materials = prob.coupled_materials(groups)
    geometry, sources, quadrature = prob.axis_problem(2, groups)
    solver = SolverData(max_iter_energy=500, **KWARGS, **solver_kwargs)
    reference = fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter, [1.0, 0.0, 0.0])
    approx = fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.anisotropic
def test_uniform_z_matches_2d():
    # Reflecting on both z faces makes the flux even in angle_z, so only the
    # harmonics kept in 2D are nonzero and the P2 solutions match
    cells_x, cells_y, cells_z = 10, 8, 3
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    medium_map[3:7, 2:6] = 1
    external = np.zeros((cells_x, cells_y, 1, 1))
    external[3:7, 2:6] = 1.0
    materials = prob.two_materials()
    materials.scatter = with_moments(materials.scatter, [1.0, 0.4, 0.15])

    geometry_2d = GeometryData(
        medium_map=medium_map,
        delta_x=np.full(cells_x, 0.4),
        delta_y=np.full(cells_y, 0.5),
        geometry=Geometry.SLAB2D,
    )
    sources_2d = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    flux_2d = fixed_source_2d(
        materials, sources_2d, geometry_2d, ants.angular_xy(4), SolverData(**KWARGS)
    )

    geometry = GeometryData(
        medium_map=np.repeat(medium_map[..., None], cells_z, axis=2),
        delta_x=geometry_2d.delta_x,
        delta_y=geometry_2d.delta_y,
        delta_z=np.full(cells_z, 0.7),
        bc_z=[1, 1],
        geometry=Geometry.SLAB3D,
    )
    boundary_x, boundary_y, boundary_z = prob.zero_boundaries()
    sources = SourceData(
        external=np.repeat(external[:, :, None], cells_z, axis=2),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_z=boundary_z,
    )
    flux = fixed_source(
        materials, sources, geometry, ants.angular_xyz(4), SolverData(**KWARGS)
    )
    for kk in range(cells_z):
        assert np.allclose(flux[:, :, kk], flux_2d, rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.anisotropic
def test_axis_permutation():
    # The scattering kernel only depends on Omega . Omega', so relabeling the
    # axes gives the same flux even though the harmonics use z as the pole
    materials = prob.two_materials(groups=2)
    materials.scatter = with_moments(materials.scatter, [1.0, 0.5, 0.2])
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(materials, sources, geometry, quadrature, SolverData(**KWARGS))
    geometry_p, sources_p, quadrature_p = prob.permute_axes(
        geometry, sources, quadrature
    )
    flux_p = fixed_source(
        materials, sources_p, geometry_p, quadrature_p, SolverData(**KWARGS)
    )
    assert np.allclose(flux_p, flux.transpose(2, 0, 1, 3), rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.anisotropic
def test_angular_to_scalar():
    materials = prob.two_materials(groups=2)
    materials.scatter = with_moments(materials.scatter, [1.0, 0.5, 0.2])
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(materials, sources, geometry, quadrature, SolverData(**KWARGS))
    solver = SolverData(angular=True, **KWARGS)
    angular_flux = fixed_source(materials, sources, geometry, quadrature, solver)
    scalar_flux = np.einsum("ijknq,n->ijkq", angular_flux, quadrature.angle_w)
    assert np.allclose(scalar_flux, flux, rtol=1e-6)
