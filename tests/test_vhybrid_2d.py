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
from ants import hybrid2d, vhybrid2d
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import hybrid as hytools
from ants.utils import manufactured_2d as mms
from tests import problems2d


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf1
def test_backward_euler_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.bdf2
def test_bdf2_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_01():
    # General parameters
    cells_x = 100
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.trbdf2
def test_tr_bdf2_02():
    # General parameters
    cells_x = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / steps
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)
    )

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid2d.time_dependent(
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
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y, edges_t[1:]
    )
    exact = np.sum(exact * quadrature.angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


###############################################################################
# Multigroup Problems
###############################################################################
def _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=1):
    cells_x = cells_y = 50
    steps = 2
    dt = 0.1
    bc_x = [0, 0]
    bc_y = [0, 0]

    # Spatial Layout
    radius = 4.279960
    length_x = length_y = 2 * radius

    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)

    # Energy Grid
    edges_g, _, _ = ants.energy_grid(87, groups_u, groups_c)
    velocity_u = ants.energy_velocity(groups_u, edges_g)

    # Angular
    quadrature_u = ants.angular_xy(angles_u, bc_x=bc_x, bc_y=bc_y)
    quadrature_c = ants.angular_xy(angles_c, bc_x=bc_x, bc_y=bc_y)

    # Medium Map and cross sections
    materials = np.array(["uranium-%20%", "vacuum"])
    xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(
        87, materials, datatype=False
    )

    weight_matrix = np.load("data/weight_matrix_2d/cylinder_two_material.npy")
    medium_map, xs_total_u, xs_scatter_u, xs_fission_u = ants.weight_spatial2d(
        weight_matrix, xs_total_u, xs_scatter_u, xs_fission_u
    )

    mat_data_u = MaterialData(
        total=xs_total_u,
        scatter=xs_scatter_u,
        fission=xs_fission_u,
        velocity=velocity_u,
    )

    # Boundary conditions and external source
    external = np.zeros((1, cells_x, cells_y, 1, 1))
    edges_t = np.linspace(0, dt * steps, steps + 1)
    # Gamma half steps for TR-BDF2
    if temporal == 4:
        edges_t = ants.gamma_time_steps(edges_t)

    boundary_y = np.zeros((1, 2, 1, 1, 1))
    boundary_x = np.zeros((2, 1, 1, 1))
    boundary_x[0] = 1.0
    boundary_x = ants.boundary2d.time_dependence_decay_01(boundary_x, edges_t, 8.0)

    if temporal in [2, 4]:
        initial_flux_x = np.zeros((cells_x + 1, cells_y, angles_u**2, groups_u))
        initial_flux_y = np.zeros((cells_x, cells_y + 1, angles_u**2, groups_u))
        sources = SourceData(
            initial_flux_x=initial_flux_x,
            initial_flux_y=initial_flux_y,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
    else:
        initial_flux = np.zeros((cells_x, cells_y, angles_u**2, groups_u))
        sources = SourceData(
            initial_flux=initial_flux,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )

    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=delta_x,
        delta_y=delta_y,
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=steps, dt=dt, time_disc=temporal)

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
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    if groups_u == groups_c:
        mat_data_c = MaterialData(
            total=mat_data_u.total,
            scatter=mat_data_u.scatter,
            fission=mat_data_u.fission,
            velocity=mat_data_u.velocity,
        )
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


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_bdf1(angles_c, groups_c):
    angles_u = 4
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=1)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid2d.time_dependent(
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
    vhy_flux = vhybrid2d.time_dependent(
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

    # Compare each time step
    for tt in range(time_data.steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.cn
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_cn(angles_c, groups_c):
    angles_u = 4
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=2)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid2d.time_dependent(
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
    vhy_flux = vhybrid2d.time_dependent(
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

    # Compare each time step
    for tt in range(time_data.steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.bdf2
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_bdf2(angles_c, groups_c):
    angles_u = 4
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=3)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid2d.time_dependent(
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
    vhy_flux = vhybrid2d.time_dependent(
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

    # Compare each time step
    for tt in range(time_data.steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.trbdf2
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_tr_bdf2(angles_c, groups_c):
    angles_u = 2
    groups_u = 87

    mat_data_u, sources, geometry, quad_u, quad_c, solver, time_data, edges_g = (
        _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=4)
    )
    mat_data_c, hybrid_data = _get_hybrid_params(groups_u, groups_c, mat_data_u)

    # Run Hybrid Method
    hy_flux = hybrid2d.time_dependent(
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
    vhy_flux = vhybrid2d.time_dependent(
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

    # Compare each time step
    for tt in range(time_data.steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()
