########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Anisotropic (Legendre P_N) scattering in one dimension. Benchmarks
# from "Analytical Benchmark Test Set for Criticality Code
# Verification" from Los Alamos National Lab, and checks that scatter
# with zero higher moments matches the isotropic solvers.
#
########################################################################

from dataclasses import replace

import numpy as np
import pytest

import ants
from ants import fixed1d, hybrid1d, nearby1d, timed1d, vhybrid1d
from ants.critical1d import k_criticality
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import hybrid as hytools
from tests import problems1d as prob


def with_moments(scatter, moments=2):
    # Isotropic scatter as Legendre moments with zero higher moments
    xs_scatter = np.zeros(np.shape(scatter) + (moments,))
    xs_scatter[..., 0] = scatter
    return xs_scatter


def critical_problem(
    xs_total, xs_scatter, fission, radius, bc_x, cells_x, geometry=1, chi=None
):
    # Full width slab for vacuum boundaries, half width when reflected
    if (geometry == 1) and (np.sum(bc_x) == 0):
        cells_x, radius = 2 * cells_x, 2 * radius
    materials = MaterialData(
        total=np.array(xs_total), scatter=np.array(xs_scatter), fission=fission, chi=chi
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells_x,), dtype=np.int32),
        delta_x=np.repeat(radius / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry,
    )
    return materials, geometry


def one_group_fission(nu_sigma_f, chi):
    # Fission matrix, or nu * sigma_f with a fission spectrum
    if chi:
        return np.array([[nu_sigma_f]]), np.array([[1.0]])
    return np.array([[[nu_sigma_f]]]), None


########################################################################
# Legendre polynomials
########################################################################


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_legendre_polynomials():
    angle_x = ants.angular_x(8).angle_x
    P = ants.legendre_polynomials(5, angle_x)
    for ll in range(5):
        coefs = np.zeros((ll + 1,))
        coefs[ll] = 1.0
        assert np.allclose(P[ll], np.polynomial.legendre.legval(angle_x, coefs))


@pytest.mark.smoke
@pytest.mark.anisotropic
def test_legendre_data():
    quadrature = ants.angular_x(8)
    P_source, P_weights = ants.legendre_data(quadrature, 3)
    assert quadrature.P.shape == (3, 8)
    assert np.allclose(P_weights, quadrature.angle_w[None, :] * quadrature.P)
    assert np.allclose(P_source, np.array([1, 3, 5])[:, None] * quadrature.P)
    # Gauss-Legendre: higher moments integrate to zero over angle
    assert np.allclose(np.sum(P_weights, axis=1), [1.0, 0.0, 0.0])


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


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize("name", PUA_SLABS)
@pytest.mark.parametrize("boundary", [[0, 0], [0, 1], [1, 0]])
@pytest.mark.parametrize("chi", [False, True])
def test_plutonium_slab(name, boundary, chi):
    moments, radius = PUA_SLABS[name]
    fission, chi = one_group_fission(2.5 * 0.266667, chi)
    materials, geometry = critical_problem(
        [[1.0]], [[[moments]]], fission, radius, boundary, 75, chi=chi
    )
    quadrature = ants.angular_x(16, boundary)
    _, keff = k_criticality(materials, geometry, quadrature, SolverData())
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


########################################################################
# One Group Anisotropic Scattering - Heavy Water Spheres
########################################################################

# (P1 moment, nu)
UD2O_SPHERES = {
    "UD2O(a)-1-1-SP": (0.056312624, 1.808381),
    "UD2O(b)-1-1-SP": (0.112982569, 1.841086),
    "UD2O(c)-1-1-SP": (-0.27850447, 1.6964),
}


@pytest.mark.sphere1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize("name", UD2O_SPHERES)
@pytest.mark.parametrize("chi", [False, True])
def test_heavy_water_sphere(name, chi):
    scatter_1, nu = UD2O_SPHERES[name]
    fission, chi = one_group_fission(nu * 0.054628, chi)
    materials, geometry = critical_problem(
        [[0.54628]],
        [[[[0.464338, scatter_1]]]],
        fission,
        18.30563081,
        [1, 0],
        200,
        geometry=2,
        chi=chi,
    )
    quadrature = ants.angular_x(16, [1, 0])
    _, keff = k_criticality(materials, geometry, quadrature, SolverData())
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


########################################################################
# Two Group Anisotropic Scattering - Uranium Reactor
########################################################################


def uranium_reactor(boundary, off_group_p1=True):
    # URRa-2-1-SL: group 1 is fast (fission spectrum), group 0 is thermal
    xs_total = np.array([[2.52025, 0.65696]])
    scatter_0 = np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T
    scatter_1 = np.array([[0.83318, 0.0], [0.0075737, 0.27459]]).T
    if not off_group_p1:
        scatter_1 = np.diag(np.diag(scatter_1))
    xs_scatter = np.stack([scatter_0, scatter_1], axis=-1)[None, ...]
    chi = np.array([[0.0], [1.0]])
    nu_sigma_f = np.array([[2.5 * 0.050632, 2.5 * 0.0010484]])
    fission = np.array([chi @ nu_sigma_f])
    return critical_problem(xs_total, xs_scatter, fission, 9.4916, boundary, 100)


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize("boundary", [[0, 0], [0, 1], [1, 0]])
def test_uranium_reactor_slab(boundary):
    materials, geometry = uranium_reactor(boundary)
    quadrature = ants.angular_x(20, boundary)
    _, keff = k_criticality(materials, geometry, quadrature, SolverData())
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_uranium_reactor_off_group_moments():
    # Off-group P1 scattering is included in the source
    quadrature = ants.angular_x(20, [1, 0])
    keffs = []
    for off_group_p1 in [True, False]:
        materials, geometry = uranium_reactor([1, 0], off_group_p1)
        keffs.append(k_criticality(materials, geometry, quadrature, SolverData())[1])
    assert abs(keffs[0] - 1.0) < 2e-3
    assert abs(keffs[0] - keffs[1]) > 1e-5


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("mg_solver", "parallel"), [(1, 2), (1, 3), (2, 1)])
def test_uranium_reactor_solvers(mg_solver, parallel):
    # Group parallel (Jacobi) and DMD multigroup solvers
    materials, geometry = uranium_reactor([1, 0])
    quadrature = ants.angular_x(20, [1, 0])
    solver = SolverData(mg_solver=mg_solver, parallel=parallel, num_threads=2)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


########################################################################
# Zero higher moments match isotropic scattering
########################################################################


def multigroup_problem(cells_x=40, angles=8, moments=None, edges=False):
    # Two material, four group slab with down and up scattering
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
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    medium_map[cells_x // 2 :] = 1
    geometry = GeometryData(
        medium_map=medium_map, delta_x=np.repeat(4.0 / cells_x, cells_x), bc_x=[0, 0]
    )
    quadrature = ants.angular_x(angles, [0, 0])
    external = np.zeros((cells_x, 1, groups))
    external[: cells_x // 4, 0, 0] = 1.0
    initial_flux = np.zeros((cells_x + edges, angles, groups))
    sources = SourceData(
        initial_flux=initial_flux,
        external=external,
        boundary_x=np.zeros((2, 1, 1)),
    )
    return materials, sources, geometry, quadrature


def time_sources(sources):
    # Constant in time external and boundary sources
    return replace(
        sources,
        external=sources.external[None, ...].copy(),
        boundary_x=np.zeros((1, 2, 1, 1)),
    )


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize(("angular", "edges"), [(False, 0), (True, 0), (False, 1)])
def test_fixed_source_zero_moments(angular, edges):
    solver = SolverData(angular=angular, flux_at_edges=edges)
    materials, sources, geometry, quadrature = multigroup_problem()
    reference = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter, 3)
    approx = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.sphere1d
@pytest.mark.anisotropic
def test_fixed_source_sphere_zero_moments():
    solver = SolverData()
    materials, sources, geometry, quadrature = multigroup_problem()
    geometry.geometry = 2
    geometry.bc_x = [1, 0]
    quadrature = ants.angular_x(8, [1, 0])
    reference = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter)
    approx = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize(("mg_solver", "parallel"), [(1, 2), (1, 3), (2, 1)])
def test_fixed_source_solvers_zero_moments(mg_solver, parallel):
    solver = SolverData(mg_solver=mg_solver, parallel=parallel, num_threads=2)
    materials, sources, geometry, quadrature = multigroup_problem()
    reference = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter)
    approx = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(approx, reference, rtol=1e-6, atol=1e-10)


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize("angular", [False, True])
def test_criticality_zero_moments(angular):
    materials, geometry = critical_problem(
        [[1.0]], [[[0.733333]]], np.array([[[0.666668]]]), 0.8, [0, 0], 50
    )
    quadrature = ants.angular_x(8)
    solver = SolverData(angular=angular)
    reference = k_criticality(materials, geometry, quadrature, solver)
    materials.scatter = with_moments(materials.scatter)
    approx = k_criticality(materials, geometry, quadrature, solver)
    # Angular flux is returned without keff
    if angular:
        assert np.allclose(approx, reference, rtol=1e-8)
    else:
        assert abs(approx[1] - reference[1]) < 1e-10
        assert np.allclose(approx[0], reference[0], rtol=1e-8)


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_time_dependent_zero_moments(time_disc):
    edges = int(time_disc in (2, 4))
    solver = SolverData()
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem(edges=edges)
    sources = time_sources(sources)
    reference = timed1d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    materials.scatter = with_moments(materials.scatter)
    approx = timed1d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 3])
def test_time_dependent_steady_state(time_disc):
    # Long time steps approach the steady state solution
    solver = SolverData()
    time_data = TimeDependentData(steps=6, dt=1e4, time_disc=time_disc)
    materials, sources, geometry, quadrature = multigroup_problem(
        moments=[1.0, 0.3, 0.1]
    )
    steady = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    approx = timed1d.time_dependent(
        materials, time_sources(sources), geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, steady, rtol=1e-5)


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.parametrize("moments", [None, [1.0, 0.3, 0.1]])
def test_known_source_calculation(moments):
    # Angular flux at each step from the scalar flux or flux moments
    solver = SolverData(angular=True)
    materials, sources, geometry, quadrature = multigroup_problem(moments=moments)
    angular = fixed1d.fixed_source(materials, sources, geometry, quadrature, solver)
    if moments is None:
        flux = np.einsum("n,ing->ig", quadrature.angle_w, angular)
    else:
        _, P_weights = ants.legendre_data(quadrature, len(moments))
        flux = np.einsum("ln,ing->igl", P_weights, angular)
    time_data = TimeDependentData(steps=2, dt=0.1)
    approx = timed1d.known_source_calculation(
        np.stack([flux, flux]),
        materials,
        time_sources(sources),
        geometry,
        quadrature,
        solver,
        time_data,
    )
    assert approx.shape == (2,) + angular.shape
    assert np.allclose(approx, angular[None], rtol=1e-6, atol=1e-10)


def hybrid_problem(time_disc, moments):
    # Collided and uncollided grids are the same, which is the full solution
    edges = int(time_disc in (2, 4))
    materials, sources, geometry, quadrature = multigroup_problem(
        moments=moments, edges=edges
    )
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    return materials, time_sources(sources), geometry, quadrature, time_data


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_hybrid_matches_full(time_disc):
    solver = SolverData()
    materials, sources, geometry, quadrature, time_data = hybrid_problem(
        time_disc, [1.0, 0.3, 0.1]
    )
    groups = materials.total.shape[1]
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    full = timed1d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = hybrid1d.time_dependent(
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


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_hybrid_zero_moments(time_disc):
    # Coarse collided grid with zero higher moments matches isotropic
    solver = SolverData()
    materials, sources, geometry, quadrature, time_data = hybrid_problem(
        time_disc, None
    )
    groups = materials.total.shape[1]
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, 2)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    quadrature_c = ants.angular_x(4)

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
            hybrid1d.time_dependent(
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


@pytest.mark.slab1d
@pytest.mark.anisotropic
@pytest.mark.hybrid
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize("moments", [None, [1.0, 0.3, 0.1]])
def test_vhybrid_matches_full(time_disc, moments):
    solver = SolverData()
    materials, sources, geometry, quadrature, time_data = hybrid_problem(
        time_disc, moments
    )
    groups = materials.total.shape[1]
    angles = quadrature.angle_x.size
    edges_g, _ = ants.energy_grid(None, groups)
    full = timed1d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = vhybrid1d.time_dependent(
        materials,
        np.full((time_data.steps,), groups, dtype=np.int32),
        sources,
        geometry,
        quadrature,
        np.full((time_data.steps,), angles, dtype=np.int32),
        solver,
        time_data,
        edges_g,
    )
    assert np.allclose(approx, full, rtol=1e-5, atol=1e-10)


########################################################################
# Nearby problems
########################################################################


@pytest.mark.nearby1d
@pytest.mark.anisotropic
@pytest.mark.parametrize("scalar_residual", [False, True])
def test_nearby_fixed_source_zero_moments(scalar_residual):
    mat_data, sources, geometry, quadrature, solver = prob.manufactured_ss_03(50, 4)
    solver.angular = True
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    kwargs = {"scalar_residual": scalar_residual, "save_residual": False}
    reference = nearby1d.fixed_source(
        mat_data, sources, geometry, quadrature, solver, centers_x, **kwargs
    )
    mat_data.scatter = with_moments(mat_data.scatter)
    approx = nearby1d.fixed_source(
        mat_data, sources, geometry, quadrature, solver, centers_x, **kwargs
    )
    for value, expected in zip(approx, reference):
        assert np.allclose(value, expected, rtol=1e-8, atol=1e-12)


@pytest.mark.nearby1d
@pytest.mark.anisotropic
def test_nearby_fixed_source():
    # The nearby solution is close to the curve fit of the numerical flux
    materials, sources, geometry, quadrature = multigroup_problem(
        cells_x=100, angles=4, moments=[1.0, 0.3, 0.1]
    )
    solver = SolverData(angular=True)
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    result = nearby1d.fixed_source(
        materials,
        sources,
        geometry,
        quadrature,
        solver,
        centers_x,
        return_residual=True,
        save_residual=False,
    )
    _, curve_fit_flux, _, _, nearby_flux = result
    assert np.allclose(nearby_flux, curve_fit_flux, rtol=5e-3, atol=1e-6)


@pytest.mark.nearby1d
@pytest.mark.anisotropic
def test_nearby_k_criticality():
    moments, radius = PUA_SLABS["PUa-1-2-SL"]
    materials, geometry = critical_problem(
        [[1.0]], [[[moments]]], np.array([[[2.5 * 0.266667]]]), radius, [0, 0], 50
    )
    quadrature = ants.angular_x(4)
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    result = nearby1d.k_criticality(
        materials, geometry, quadrature, SolverData(), centers_x, save_residual=False
    )
    _, numerical_keff, _, _, nearby_scalar, nearby_keff = result
    assert nearby_scalar.shape == (geometry.delta_x.size, 1)
    assert abs(nearby_keff - numerical_keff) < 1e-3


@pytest.mark.nearby1d
@pytest.mark.anisotropic
def test_nearby_scalar_flux_error():
    materials, sources, geometry, quadrature = multigroup_problem(moments=[1.0, 0.3])
    scalar_flux = fixed1d.fixed_source(
        materials, sources, geometry, quadrature, SolverData()
    )
    with pytest.raises(ValueError, match="angular flux"):
        nearby1d.fixed_source(
            materials,
            sources,
            geometry,
            quadrature,
            SolverData(angular=True),
            np.ones((geometry.delta_x.size,)),
            numerical_flux=scalar_flux,
            save_residual=False,
        )
