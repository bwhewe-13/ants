########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for three-dimensional hybrid time-dependent problems. Includes
# tests for Backward Euler, Crank-Nicolson, BDF2, and TR-BDF2.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import hybrid3d, timed3d
from ants.datatypes import MaterialData, SolverData, TimeDependentData
from tests import problems3d as prob

KWARGS = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)


def _manufactured_error(temporal):
    # Final scalar flux error of the hybrid method with the same collided
    # and uncollided grids
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        prob.manufactured_td_01(4, 4, edges_t, dt, temporal)
    )
    _, hybrid_data, _ = prob.collided_data(mat_data, 1)
    approx = hybrid3d.time_dependent(
        mat_data,
        mat_data,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = prob.solution_td_01(
        centers_x, centers_x, centers_x, quadrature.angle_x, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[:, None], axis=4)
    return np.max(np.fabs(approx - exact[-1]))


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.bdf1
def test_backward_euler_01():
    assert _manufactured_error(temporal=1) < 2e-2


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.cn
def test_crank_nicolson_01():
    assert _manufactured_error(temporal=2) < 5e-4


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.bdf2
def test_bdf2_01():
    assert _manufactured_error(temporal=3) < 1e-3


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    assert _manufactured_error(temporal=4) < 5e-4


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize("ratios", [None, [1.0, 0.3, 0.1]])
def test_same_grids(time_disc, ratios):
    # The same collided and uncollided grids give the timed3d solution
    solver = SolverData(**KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(
        time_disc, ratios=ratios
    )
    materials_c, hybrid_data, _ = prob.collided_data(materials, 4)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = hybrid3d.time_dependent(
        materials,
        materials_c,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize(("angles_c", "groups_c"), [(2, 4), (4, 2), (2, 2)])
def test_coarse_collided(time_disc, angles_c, groups_c):
    # Fewer collided angles and groups stay close to the timed3d solution
    solver = SolverData(**KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(time_disc)
    materials_c, hybrid_data, _ = prob.collided_data(materials, groups_c)
    quadrature_c = ants.angular_xyz(angles_c, bc_x=geometry.bc_x, bc_z=geometry.bc_z)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = hybrid3d.time_dependent(
        materials,
        materials_c,
        sources,
        geometry,
        quadrature,
        quadrature_c,
        solver,
        time_data,
        hybrid_data,
    )
    assert np.max(np.fabs(approx - reference)) < 1e-2 * np.max(reference)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_artificial_scattering(time_disc):
    # The same grids give the timed3d as-SN solution
    solver = SolverData(sigma_as=0.2, beta_as=1.0, **KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(time_disc)
    materials_c, hybrid_data, _ = prob.collided_data(materials, 4)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    approx = hybrid3d.time_dependent(
        materials,
        materials_c,
        sources,
        geometry,
        quadrature,
        quadrature,
        solver,
        time_data,
        hybrid_data,
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.bdf2
def test_restart_bdf2(tmp_path):
    # Restart from the angular flux of two steps, rebuilt from the saved
    # scalar flux. The same grids are used so the hybrid angular flux is
    # the one from timed3d.angular_bdf2
    solver = SolverData(**KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(3)
    materials_c, hybrid_data, _ = prob.collided_data(materials, 4)
    quadrature_c = quadrature
    flux_file = str(tmp_path / "flux.npy")
    time_data = TimeDependentData(steps=6, dt=0.1, time_disc=3, save_to_file=flux_file)
    full = hybrid3d.time_dependent(
        materials,
        materials_c,
        sources,
        geometry,
        quadrature,
        quadrature_c,
        solver,
        time_data,
        hybrid_data,
    )
    scalar_flux = np.load(flux_file)
    assert np.allclose(scalar_flux[-1], full)

    time_steps = np.array([2, 3], dtype=np.int32)
    angular = timed3d.angular_bdf2(
        time_steps,
        scalar_flux,
        sources.initial_flux,
        materials,
        sources,
        geometry,
        quadrature,
        solver,
        time_data,
    )
    time_data = TimeDependentData(steps=2, dt=0.1, time_disc=3)
    approx = hybrid3d.restart_bdf2(
        angular[1],
        angular[0],
        materials,
        materials_c,
        sources,
        geometry,
        quadrature,
        quadrature_c,
        solver,
        time_data,
        hybrid_data,
    )
    assert np.allclose(approx, full, rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
def test_collided_moments():
    # The collided materials need the same scattering moments
    materials, sources, geometry, quadrature = prob.coupled_time_problem(
        1, ratios=[1.0, 0.3]
    )
    materials_c, hybrid_data, _ = prob.collided_data(materials, 4)
    materials_c = MaterialData(
        total=materials.total,
        scatter=materials.scatter[..., 0],
        fission=materials.fission,
        velocity=materials.velocity,
    )
    time_data = TimeDependentData(steps=1, dt=0.1)
    with pytest.raises(AssertionError, match="same moments"):
        hybrid3d.time_dependent(
            materials,
            materials_c,
            sources,
            geometry,
            quadrature,
            quadrature,
            SolverData(),
            time_data,
            hybrid_data,
        )
