########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Three-dimensional time dependent problems
#
########################################################################

import numpy as np
import pytest

import ants
from ants import timed3d
from ants.datatypes import (
    GeometryData,
    ParallelType,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.fixed3d import fixed_source
from tests import problems3d as prob

KWARGS = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)


def axis_problem(spatial=2, time_disc=1):
    # prob.axis_problem with a velocity and the sources of a time-dependent
    # problem, constant in time with a zero initial flux
    geometry, sources, quadrature = prob.axis_problem(spatial)
    materials = prob.two_materials(groups=2)
    materials.velocity = np.array([2.0, 1.0])
    cells = geometry.medium_map.shape
    shape = (quadrature.angle_w.size, 2)
    kwargs = dict(
        external=sources.external[None],
        boundary_x=sources.boundary_x[None],
        boundary_y=sources.boundary_y[None],
        boundary_z=sources.boundary_z[None],
    )
    if time_disc in (1, 3):
        kwargs["initial_flux"] = np.zeros(cells + shape)
    else:
        kwargs["initial_flux_x"] = np.zeros((cells[0] + 1,) + cells[1:] + shape)
        kwargs["initial_flux_y"] = np.zeros((cells[0], cells[1] + 1, cells[2]) + shape)
        kwargs["initial_flux_z"] = np.zeros(cells[:2] + (cells[2] + 1,) + shape)
    return materials, sources, SourceData(**kwargs), geometry, quadrature


def with_moments(scatter, ratios):
    # Legendre moments that are a ratio of the isotropic scatter
    return np.stack([ratio * scatter for ratio in ratios], axis=-1)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.parametrize(
    ("spatial", "time_disc"), [(1, 1), (1, 3), (2, 1), (2, 3), (2, 4)]
)
def test_steady_state(spatial, time_disc):
    # Long time steps approach the steady state solution. CN does not damp
    # the transient, SC and TR-BDF2 with step are biased as in timed2d
    solver = SolverData(**KWARGS)
    materials, steady_sources, sources, geometry, quadrature = axis_problem(
        spatial, time_disc
    )
    steady = fixed_source(materials, steady_sources, geometry, quadrature, solver)
    time_data = TimeDependentData(steps=6, dt=1e4, time_disc=time_disc)
    approx = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, steady, rtol=1e-6, atol=1e-12)


@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_anisotropic_zero_moments(time_disc):
    # Zero higher scattering moments give the isotropic solution
    solver = SolverData(**KWARGS)
    materials, _, sources, geometry, quadrature = axis_problem(2, time_disc)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    materials.scatter = with_moments(materials.scatter, [1.0, 0.0])
    approx = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 3])
def test_anisotropic_steady_state(time_disc):
    # Long time steps approach the anisotropic steady state solution
    solver = SolverData(**KWARGS)
    materials, steady_sources, sources, geometry, quadrature = axis_problem(
        2, time_disc
    )
    materials.scatter = with_moments(materials.scatter, [1.0, 0.3, 0.1])
    steady = fixed_source(materials, steady_sources, geometry, quadrature, solver)
    time_data = TimeDependentData(steps=6, dt=1e4, time_disc=time_disc)
    approx = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, steady, rtol=1e-6, atol=1e-12)


@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize("parallel", [ParallelType.GROUP, ParallelType.BOTH])
def test_group_parallel(time_disc, parallel):
    # Group parallel (Jacobi) matches the Gauss-Seidel source iteration
    materials, _, sources, geometry, quadrature = axis_problem(2, time_disc)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, SolverData(**KWARGS), time_data
    )
    solver = SolverData(parallel=parallel, num_threads=2, **KWARGS)
    approx = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.bdf2
def test_restart_bdf2(tmp_path):
    # Angular flux of two steps from the saved scalar flux, then the rest
    # of the steps from a restart
    solver = SolverData(**KWARGS)
    materials, _, sources, geometry, quadrature = axis_problem(2, 3)
    flux_file = str(tmp_path / "flux.npy")
    time_data = TimeDependentData(steps=6, dt=0.1, time_disc=3, save_to_file=flux_file)
    full = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
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
    assert angular.shape == (2,) + sources.initial_flux.shape
    angular_flux = np.sum(angular * quadrature.angle_w[:, None], axis=-2)
    assert np.allclose(angular_flux, scalar_flux[time_steps], rtol=1e-8)

    time_data = TimeDependentData(steps=2, dt=0.1, time_disc=3)
    approx = timed3d.restart_bdf2(
        angular[1],
        angular[0],
        materials,
        sources,
        geometry,
        quadrature,
        solver,
        time_data,
    )
    assert np.allclose(approx, full, rtol=1e-8, atol=1e-12)


@pytest.mark.slab3d
@pytest.mark.time_dependent
@pytest.mark.parametrize("time_disc", [1, 3, 4])
def test_artificial_scattering(time_disc):
    # The lagged as-SN in-scatter source approaches the fixed source as-SN
    # solution with long time steps
    solver = SolverData(sigma_as=0.2, beta_as=1.0, **KWARGS)
    materials, steady_sources, sources, geometry, quadrature = axis_problem(
        2, time_disc
    )
    steady = fixed_source(materials, steady_sources, geometry, quadrature, solver)
    time_data = TimeDependentData(steps=20, dt=1e4, time_disc=time_disc)
    approx = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(approx, steady, rtol=1e-6, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.time_dependent
def test_initial_flux_edges():
    # CN and TR-BDF2 need the initial flux on the faces
    materials, _, sources, geometry, quadrature = axis_problem(2, 2)
    _, _, centers, _, _ = axis_problem(2, 1)
    sources.initial_flux_z = centers.initial_flux
    time_data = TimeDependentData(steps=1, dt=0.1, time_disc=2)
    with pytest.raises(AssertionError, match="cell edges"):
        timed3d.time_dependent(
            materials, sources, geometry, quadrature, SolverData(), time_data
        )


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.time_dependent
def test_quadrature_not_ordered():
    # The angular flux of each step is found with one sweep of each
    # ordinate, so the angles need to be ordered for the reflecting faces
    materials, _, sources, geometry, quadrature = axis_problem(1, 1)
    geometry = GeometryData(
        medium_map=geometry.medium_map,
        delta_x=geometry.delta_x,
        delta_y=geometry.delta_y,
        delta_z=geometry.delta_z,
        bc_x=[0, 0],
        bc_y=[0, 0],
        bc_z=[1, 0],
        geometry=geometry.geometry,
        space_disc=geometry.space_disc,
    )
    time_data = TimeDependentData(steps=1, dt=0.1)
    with pytest.raises(ValueError, match="not ordered for bc_z"):
        timed3d.time_dependent(
            materials, sources, geometry, ants.angular_xyz(4), SolverData(), time_data
        )
    quadrature = ants.angular_xyz(4, bc_z=[1, 0])
    flux = timed3d.time_dependent(
        materials, sources, geometry, quadrature, SolverData(), time_data
    )
    assert np.all(flux >= 0.0)
