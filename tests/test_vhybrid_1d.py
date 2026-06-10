########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for large multigroup problems (G >= 87) and focuses on fixed
# source, time dependent, and criticality problems.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import hybrid1d, vhybrid1d
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import hybrid as hytools
from ants.utils import manufactured_1d as mms
from tests import problems1d

# Path for reference solutions
PATH = "data/references_multigroup/"


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab1d
@pytest.mark.bdf1
def test_backward_euler_01():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
def test_backward_euler_02():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab1d
@pytest.mark.cn
def test_crank_nicolson_01():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.cn
def test_crank_nicolson_02():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab1d
@pytest.mark.bdf2
def test_bdf2_01():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf2
def test_bdf2_02():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab1d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # General parameters
    cells_x = 200
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)
    )
    edges_g, _ = ants.energy_grid(None, groups)

    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.time_dependent(
        mat_data,
        vgroups,
        sources,
        geometry,
        quadrature,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


###############################################################################
# Multigroup Problems
###############################################################################
def _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=1):
    # General Parameters
    cells_x = 1000
    length_x = 10.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    bc_x = [0, 0]

    # Energy Grid
    edges_g, _, _ = ants.energy_grid(87, groups_u, groups_c)

    # Medium Map
    layout = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
    mat_data_u = ants.materials(groups_u, np.array(layout)[:, 1], datatype=True)
    mat_data_u.velocity = ants.energy_velocity(groups_u, edges_g)

    geometry = GeometryData(
        medium_map=ants.spatial1d(layout, edges_x),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        bc_x=[1, 0],
        geometry=2,
    )
    quadrature_u = ants.angular_x(angles_u, bc_x=bc_x)
    quadrature_c = ants.angular_x(angles_c, bc_x=bc_x)
    solver = SolverData()
    time_data = TimeDependentData(steps=5, dt=1e-08, time_disc=temporal)

    # Crank-Nicolson / TR-BDF2 initial flux at cell edges
    if temporal in [2, 4]:
        initial_flux = np.zeros((cells_x + 1, angles_u, groups_u))
    else:
        initial_flux = np.zeros((cells_x, angles_u, groups_u))

    # External and boundary sources
    external = np.zeros((1, cells_x, 1, 1))
    edges_t = np.linspace(0, time_data.dt * time_data.steps, time_data.steps + 1)
    # Gamma half time steps
    if temporal == 4:
        edges_t = ants.gamma_time_steps(edges_t)
    boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
    boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)
    sources = SourceData(
        initial_flux=initial_flux.copy(),
        external=external.copy(),
        boundary_x=boundary_x.copy(),
    )

    return (
        mat_data_u,
        sources,
        geometry,
        quadrature_u,
        quadrature_c,
        solver,
        time_data,
        edges_g,
    )


def _get_hybrid_params(groups_u, groups_c, mat_data_u):
    # Get hybrid parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Check for same number of energy groups
    if groups_u == groups_c:
        xs_total_c = mat_data_u.total
        xs_scatter_c = mat_data_u.scatter
        xs_fission_c = mat_data_u.fission
        velocity_c = mat_data_u.velocity
    else:
        xs_collided = hytools.coarsen_materials(
            mat_data_u.total,
            mat_data_u.scatter,
            mat_data_u.fission,
            edges_g[edges_gidx_u],
            edges_gidx_c,
        )
        xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
        velocity_c = hytools.coarsen_velocity(mat_data_u.velocity, edges_gidx_c)
    mat_data_c = MaterialData(
        total=xs_total_c,
        scatter=xs_scatter_c,
        fission=xs_fission_c,
        velocity=velocity_c,
    )

    return mat_data_c, hybrid_data


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_bdf1(angles_c, groups_c):
    angles_u = 8
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, 1)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid1d.time_dependent(
        mat_data_u,
        mat_data_c,
        sources,
        geometry,
        quad_u,
        quad_c,
        solver,
        time_data,
        hybrid_data,
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * time_data.steps, dtype=np.int32)
    vangles = np.array([angles_c] * time_data.steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid1d.time_dependent(
        mat_data_u,
        vgroups,
        sources,
        geometry,
        quad_u,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    assert np.isclose(hy_flux, vhy_flux).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.cn
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_cn(angles_c, groups_c):
    angles_u = 8
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, 2)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid1d.time_dependent(
        mat_data_u,
        mat_data_c,
        sources,
        geometry,
        quad_u,
        quad_c,
        solver,
        time_data,
        hybrid_data,
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * time_data.steps, dtype=np.int32)
    vangles = np.array([angles_c] * time_data.steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid1d.time_dependent(
        mat_data_u,
        vgroups,
        sources,
        geometry,
        quad_u,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    assert np.isclose(hy_flux, vhy_flux).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf2
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_bdf2(angles_c, groups_c):
    angles_u = 8
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, 3)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid1d.time_dependent(
        mat_data_u,
        mat_data_c,
        sources,
        geometry,
        quad_u,
        quad_c,
        solver,
        time_data,
        hybrid_data,
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * time_data.steps, dtype=np.int32)
    vangles = np.array([angles_c] * time_data.steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid1d.time_dependent(
        mat_data_u,
        vgroups,
        sources,
        geometry,
        quad_u,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    assert np.isclose(hy_flux, vhy_flux).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.trbdf2
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_tr_bdf2(angles_c, groups_c):
    angles_u = 8
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, 4)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid1d.time_dependent(
        mat_data_u,
        mat_data_c,
        sources,
        geometry,
        quad_u,
        quad_c,
        solver,
        time_data,
        hybrid_data,
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * time_data.steps, dtype=np.int32)
    vangles = np.array([angles_c] * time_data.steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid1d.time_dependent(
        mat_data_u,
        vgroups,
        sources,
        geometry,
        quad_u,
        vangles,
        solver,
        time_data,
        edges_g,
    )

    assert np.isclose(hy_flux, vhy_flux).all()
