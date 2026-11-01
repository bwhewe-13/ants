########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for anisotropic (spherical harmonic) scattering in two
# dimensional slabs. Includes the Sood benchmarks as slabs that are
# infinite in one direction and checks that zero higher moments match
# isotropic scattering for every solver.
#
########################################################################

from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

import ants
from ants import critical2d, fixed2d, hybrid2d, nearby2d, timed2d, vhybrid2d
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import hybrid as hytools
from tests import problems2d as prob


def with_moments(scatter, moments=2):
    # Isotropic scatter as Legendre moments with zero higher moments
    xs_scatter = np.zeros(np.shape(scatter) + (moments,))
    xs_scatter[..., 0] = scatter
    return xs_scatter


########################################################################
# Spherical harmonics
########################################################################


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_harmonic_degrees():
    assert np.array_equal(ants.harmonic_degrees(3), [0, 1, 1, 2, 2, 2])


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_spherical_harmonics_addition():
    # Summing over m gives P_l(a . b), averaged with the reflection of b
    # through the x-y plane since only harmonics even in z are kept
    quadrature = ants.angular_xy(6)
    angle_x, angle_y = quadrature.angle_x, quadrature.angle_y
    angle_z = np.sqrt(1 - angle_x**2 - angle_y**2)
    Y = ants.spherical_harmonics(5, angle_x, angle_y)
    ell = ants.harmonic_degrees(5)
    dot_xy = np.outer(angle_x, angle_x) + np.outer(angle_y, angle_y)
    dot_z = np.outer(angle_z, angle_z)
    for ll in range(5):
        coefs = np.zeros((ll + 1,))
        coefs[ll] = 1.0
        expected = 0.5 * (legval(dot_xy + dot_z, coefs) + legval(dot_xy - dot_z, coefs))
        assert np.allclose(Y[ell == ll].T @ Y[ell == ll], expected)


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_harmonic_data():
    quadrature = ants.angular_xy(8)
    P_source, P_weights = ants.harmonic_data(quadrature, 3)
    ell = ants.harmonic_degrees(3)
    assert quadrature.P.shape == (6, 64)
    assert np.allclose(P_weights, quadrature.angle_w[None, :] * quadrature.P)
    assert np.allclose(P_source, (2 * ell + 1)[:, None] * quadrature.P)
    # Product quadrature: harmonics are orthogonal with weight 1 / (2l + 1)
    assert np.allclose(P_weights @ quadrature.P.T, np.diag(1.0 / (2 * ell + 1)))


########################################################################
# One Group Anisotropic Scattering - Plutonium Slabs
########################################################################

# (Legendre moments, critical half width)
PUA_SLABS = {
    "PUa-1-2-SL": ([0.733333, 0.2, 0.075], 0.76378),
    "PUa-1-1-SL": ([0.733333, 0.2], 0.77032),
    "PUb-1-2-SL": ([0.733333, 0.333333, 0.125], 0.78396),
    "PUb-1-1-SL": ([0.733333, 0.333333], 0.79606),
}

# The infinite direction is a few very thick cells, so keff converges long
# before the flux shape along it and the power iterations stop at
# max_iter_keff. These tests only check keff.
ignore_power_warning = pytest.mark.filterwarnings(
    "ignore:Power iterations stopped:RuntimeWarning"
)


def critical_slab(moments, radius, boundary, direction, cells=100):
    # Slab that is finite in one direction and infinite in the other,
    # full width for vacuum boundaries, half width when reflected
    if np.sum(boundary) == 0:
        radius = 2 * radius
    materials = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array(moments)[None, None, None, :],
        fission=np.array([[[2.5 * 0.266667]]]),
    )
    finite = np.repeat(radius / cells, cells)
    infinite = np.repeat(500.0, 4)
    if direction == "x":
        geometry = GeometryData(
            medium_map=np.zeros((cells, 4), dtype=np.int32),
            delta_x=finite,
            delta_y=infinite,
            bc_x=boundary,
            bc_y=[0, 0],
            geometry=3,
        )
        quadrature = ants.angular_xy(12, bc_x=boundary)
    else:
        geometry = GeometryData(
            medium_map=np.zeros((4, cells), dtype=np.int32),
            delta_x=infinite,
            delta_y=finite,
            bc_x=[0, 0],
            bc_y=boundary,
            geometry=3,
        )
        quadrature = ants.angular_xy(12, bc_y=boundary)
    return materials, geometry, quadrature


@ignore_power_warning
@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize("name", PUA_SLABS)
@pytest.mark.parametrize("direction", ["x", "y"])
def test_plutonium_slab(name, direction):
    moments, radius = PUA_SLABS[name]
    materials, geometry, quadrature = critical_slab(moments, radius, [0, 0], direction)
    _, keff = critical2d.k_criticality(materials, geometry, quadrature, SolverData())
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@ignore_power_warning
@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize("direction", ["x", "y"])
def test_plutonium_slab_reflected(direction):
    moments, radius = PUA_SLABS["PUa-1-2-SL"]
    materials, geometry, quadrature = critical_slab(moments, radius, [1, 0], direction)
    _, keff = critical2d.k_criticality(materials, geometry, quadrature, SolverData())
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


########################################################################
# Zero higher moments match isotropic scattering
########################################################################


def multigroup_problem(cells=8, angles=4, moments=None):
    # Two material, four group square with down and up scattering
    groups = 4
    xs_total = np.array([[1.0, 1.2, 1.5, 2.0], [0.8, 1.0, 1.1, 1.3]])
    xs_scatter = np.zeros((2, groups, groups))
    for gg in range(groups):
        xs_scatter[:, gg, gg] = 0.5 * xs_total[:, gg]
        if gg < groups - 1:
            xs_scatter[:, gg + 1, gg] = 0.2 * xs_total[:, gg]
        if gg > 0:
            xs_scatter[:, gg - 1, gg] = 0.05 * xs_total[:, gg]
    xs_fission = np.zeros((2, groups, groups))
    xs_fission[0, :2] = 0.05 * xs_total[0][None, :]
    if moments is not None:
        xs_scatter = np.stack([xs_scatter * moment for moment in moments], axis=-1)
    velocity = np.array([4.0, 3.0, 2.0, 1.0])
    materials = MaterialData(
        total=xs_total, scatter=xs_scatter, fission=xs_fission, velocity=velocity
    )
    medium_map = np.zeros((cells, cells), dtype=np.int32)
    medium_map[cells // 2 :] = 1
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(2.0 / cells, cells),
        delta_y=np.repeat(2.0 / cells, cells),
        bc_x=[0, 0],
        bc_y=[0, 0],
        geometry=3,
    )
    quadrature = ants.angular_xy(angles)
    external = np.zeros((cells, cells, 1, groups))
    external[: cells // 4, : cells // 4, 0, 0] = 1.0
    directions = angles * angles
    sources = SourceData(
        initial_flux=np.zeros((cells, cells, directions, groups)),
        initial_flux_x=np.zeros((cells + 1, cells, directions, groups)),
        initial_flux_y=np.zeros((cells, cells + 1, directions, groups)),
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    return materials, sources, geometry, quadrature


def time_sources(sources):
    # Constant in time external and boundary sources
    return replace(
        sources,
        external=sources.external[None, ...].copy(),
        boundary_x=np.zeros((1, 2, 1, 1, 1)),
        boundary_y=np.zeros((1, 2, 1, 1, 1)),
    )


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.parametrize(
    ("angular", "edges"), [(False, 0), (True, 0), (False, 1), (True, 1)]
)
def test_fixed_source_zero_moments(angular, edges):
    solver = SolverData(angular=angular, flux_at_edges=edges)
    materials, sources, geometry, quadrature = multigroup_problem()
    reference = fixed2d.fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter, 3)
    approx = fixed2d.fixed_source(materials, sources, geometry, quadrature, solver)
    if isinstance(reference, tuple):
        for value, expected in zip(approx, reference):
            assert np.allclose(value, expected, rtol=1e-8, atol=1e-12)
    else:
        assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.parametrize(
    "solver_kwargs",
    [
        {"mg_solver": 2},
        {"parallel": 2, "num_threads": 2},
        {"parallel": 3, "num_threads": 2},
        {"sigma_as": 0.1, "beta_as": 4.0},
    ],
)
def test_fixed_source_solvers_zero_moments(solver_kwargs):
    # DMD, group parallel (Jacobi), and artificial scattering
    solver = SolverData(**solver_kwargs)
    materials, sources, geometry, quadrature = multigroup_problem()
    reference = fixed2d.fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter)
    approx = fixed2d.fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(approx, reference, rtol=1e-6, atol=1e-10)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.parametrize("angular", [False, True])
def test_criticality_zero_moments(angular):
    materials = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.733333]]]),
        fission=np.array([[[0.666668]]]),
    )
    geometry = GeometryData(
        medium_map=np.zeros((20, 20), dtype=np.int32),
        delta_x=np.repeat(0.2, 20),
        delta_y=np.repeat(0.2, 20),
        bc_x=[0, 0],
        bc_y=[0, 0],
        geometry=3,
    )
    quadrature = ants.angular_xy(4)
    solver = SolverData(angular=angular)
    reference = critical2d.k_criticality(materials, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter)
    approx = critical2d.k_criticality(materials, geometry, quadrature, solver)
    # Angular flux is returned without keff
    if angular:
        assert np.allclose(approx, reference, rtol=1e-8)
    else:
        assert abs(approx[1] - reference[1]) < 1e-10
        assert np.allclose(approx[0], reference[0], rtol=1e-8)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_time_dependent_zero_moments(time_disc):
    solver = SolverData()
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem()
    sources = time_sources(sources)
    reference = timed2d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    materials.scatter = with_moments(materials.scatter)
    approx = timed2d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 3])
def test_time_dependent_steady_state(time_disc):
    # Long time steps approach the steady state solution
    solver = SolverData()
    time_data = TimeDependentData(steps=6, dt=1e4, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem(
        moments=[1.0, 0.3, 0.1]
    )
    steady = fixed2d.fixed_source(materials, sources, geometry, quadrature, solver)
    approx = timed2d.time_dependent(
        materials, time_sources(sources), geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, steady, rtol=1e-5)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_hybrid_matches_full(time_disc):
    # Collided and uncollided grids are the same, which is the full solution
    solver = SolverData()
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem(
        moments=[1.0, 0.3, 0.1]
    )
    sources = time_sources(sources)
    groups = materials.total.shape[1]
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    full = timed2d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = hybrid2d.time_dependent(
        materials,
        materials,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )
    assert np.allclose(approx, full, rtol=1e-5, atol=1e-10)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_hybrid_zero_moments(time_disc):
    # Coarse collided grid with zero higher moments matches isotropic
    solver = SolverData()
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem()
    sources = time_sources(sources)
    groups = materials.total.shape[1]
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, 2)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    quadrature_c = ants.angular_xy(2)

    results = []
    for moments in [None, 2]:
        materials_u = replace(materials)
        if moments is not None:
            materials_u.scatter = with_moments(materials.scatter, moments)
        total_c, scatter_c, fission_c = hytools.coarsen_materials(
            materials_u.total,
            materials_u.scatter,
            materials_u.fission,
            edges_g[edges_gidx_u],
            edges_gidx_c,
        )
        materials_c = MaterialData(
            total=total_c,
            scatter=scatter_c,
            fission=fission_c,
            velocity=hytools.coarsen_velocity(materials.velocity, edges_gidx_c),
        )
        results.append(
            hybrid2d.time_dependent(
                materials_u,
                materials_c,
                sources,
                geometry,
                quadrature,
                quadrature_c,
                solver,
                time_data,
                hybrid_data,
            )
        )
    assert np.allclose(results[1], results[0], rtol=1e-8, atol=1e-12)


@pytest.mark.slab2d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize("moments", [None, [1.0, 0.3, 0.1]])
def test_vhybrid_matches_full(time_disc, moments):
    solver = SolverData()
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem(moments=moments)
    sources = time_sources(sources)
    groups = materials.total.shape[1]
    edges_g, _ = ants.energy_grid(None, groups)
    full = timed2d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = vhybrid2d.time_dependent(
        materials,
        np.full((time_data.steps,), groups, dtype=np.int32),
        sources,
        geometry,
        quadrature,
        np.full((time_data.steps,), 4, dtype=np.int32),
        solver,
        time_data,
        edges_g,
    )
    assert np.allclose(approx, full, rtol=1e-5, atol=1e-10)


########################################################################
# Nearby problems
########################################################################


@pytest.mark.nearby2d
@pytest.mark.anisotropic
@pytest.mark.parametrize("scalar_residual", [False, True])
def test_nearby_fixed_source_zero_moments(scalar_residual):
    mat_data, sources, geometry, quadrature, solver, *_ = prob.manufactured_ss_03(20, 2)
    solver.angular = True
    kwargs = {"scalar_residual": scalar_residual, "save_residual": False}
    reference = nearby2d.fixed_source(
        mat_data, sources, geometry, quadrature, solver, **kwargs
    )
    mat_data.scatter = with_moments(mat_data.scatter)
    approx = nearby2d.fixed_source(
        mat_data, sources, geometry, quadrature, solver, **kwargs
    )
    for value, expected in zip(approx, reference):
        assert np.allclose(value, expected, rtol=1e-8, atol=1e-12)


@ignore_power_warning
@pytest.mark.nearby2d
@pytest.mark.anisotropic
def test_nearby_k_criticality():
    moments, radius = PUA_SLABS["PUa-1-2-SL"]
    materials, geometry, _ = critical_slab(moments, radius, [0, 0], "x", cells=50)
    quadrature = ants.angular_xy(4)
    result = nearby2d.k_criticality(
        materials, geometry, quadrature, SolverData(), save_residual=False
    )
    _, numerical_keff, _, _, nearby_scalar, nearby_keff, _ = result
    assert nearby_scalar.shape == (50, 4, 1)
    assert abs(nearby_keff - numerical_keff) < 1e-3


@pytest.mark.nearby2d
@pytest.mark.anisotropic
def test_nearby_scalar_flux_error():
    materials, sources, geometry, quadrature = multigroup_problem(moments=[1.0, 0.3])
    scalar_flux = fixed2d.fixed_source(
        materials, sources, geometry, quadrature, SolverData()
    )
    with pytest.raises(ValueError, match="angular flux"):
        nearby2d.fixed_source(
            materials,
            sources,
            geometry,
            quadrature,
            SolverData(angular=True),
            numerical_flux=scalar_flux,
            save_residual=False,
        )
