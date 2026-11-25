########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for three-dimensional variable hybrid time-dependent problems.
# Includes tests for Backward Euler, Crank-Nicolson, BDF2, and TR-BDF2.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import hybrid3d, timed3d, vhybrid3d
from ants.datatypes import SolverData, TimeDependentData
from tests import problems3d as prob

KWARGS = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)


def _hybrid_fluxes(time_disc, angles_c, groups_c, ratios=None):
    # Hybrid and variable hybrid solutions with the same collided grid on
    # each time step
    solver = SolverData(**KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(
        time_disc, ratios=ratios
    )
    materials_c, hybrid_data, edges_g = prob.collided_data(materials, groups_c)
    quadrature_c = ants.angular_xyz(angles_c, bc_x=geometry.bc_x, bc_z=geometry.bc_z)
    time_data = TimeDependentData(steps=3, dt=0.1, time_disc=time_disc)
    hy_flux = hybrid3d.time_dependent(
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

    # Variable groups and angles
    vgroups = np.array([groups_c] * time_data.steps, dtype=np.int32)
    vangles = np.array([angles_c] * time_data.steps, dtype=np.int32)
    vhy_flux = vhybrid3d.time_dependent(
        materials,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )
    return hy_flux, vhy_flux


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_same_grids(time_disc):
    hy_flux, vhy_flux = _hybrid_fluxes(time_disc, 4, 4)
    assert np.allclose(hy_flux, vhy_flux, rtol=1e-8, atol=1e-12)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize(("angles_c", "groups_c"), [(2, 4), (4, 2), (2, 2)])
def test_coarse_collided(time_disc, angles_c, groups_c):
    hy_flux, vhy_flux = _hybrid_fluxes(time_disc, angles_c, groups_c)
    assert np.allclose(hy_flux, vhy_flux, rtol=1e-8, atol=1e-12)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.anisotropic
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_coarse_collided_anisotropic(time_disc):
    hy_flux, vhy_flux = _hybrid_fluxes(time_disc, 2, 2, ratios=[1.0, 0.3, 0.1])
    assert np.allclose(hy_flux, vhy_flux, rtol=1e-8, atol=1e-12)


@pytest.mark.hybrid
@pytest.mark.slab3d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_variable_collided(time_disc):
    # The collided angles and groups change between time steps and stay
    # close to the timed3d solution
    solver = SolverData(**KWARGS)
    materials, sources, geometry, quadrature = prob.coupled_time_problem(time_disc)
    _, _, edges_g = prob.collided_data(materials, 4)
    time_data = TimeDependentData(steps=4, dt=0.1, time_disc=time_disc)
    reference = timed3d.time_dependent(
        materials, sources, geometry, quadrature, solver, time_data
    )
    vgroups = np.array([4, 2, 1, 2], dtype=np.int32)
    vangles = np.array([2, 2, 4, 2], dtype=np.int32)
    approx = vhybrid3d.time_dependent(
        materials,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )
    assert np.max(np.fabs(approx - reference)) < 1e-2 * np.max(reference)
