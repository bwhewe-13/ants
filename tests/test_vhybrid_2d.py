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

import pytest
import numpy as np

import ants
from ants import vhybrid2d, hybrid2d
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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.backward_euler(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.backward_euler(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.crank_nicolson(
        vgroups,
        vangles,
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


@pytest.mark.hybrid
@pytest.mark.slab2d
@pytest.mark.cn
def test_crank_nicolson_02():
    # General parameters
    cells_x = 50
    cells_y = 50
    angles = 4
    groups = 1
    # Time parameters
    T = 1.0
    steps = 20
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.crank_nicolson(
        vgroups,
        vangles,
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.tr_bdf2(
        vgroups,
        vangles,
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_01(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

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
    dt = T / (steps)
    edges_t = np.linspace(0, T, steps + 1)

    (
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        info,
    ) = problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run Hybrid Method
    approx = vhybrid2d.tr_bdf2(
        vgroups,
        vangles,
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_y,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    exact = mms.solution_td_02(centers_x, centers_y, angle_x, angle_y, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, None, :, None], axis=3)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


###############################################################################
# Multigroup Problems
###############################################################################
def _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=1):
    cells_x = 50
    cells_y = 50
    steps = 2
    dt = 0.1
    bc_x = [0, 0]
    bc_y = [0, 0]

    info_u = {
        "cells_x": cells_x,
        "cells_y": cells_y,
        "angles": angles_u,
        "groups": groups_u,
        "materials": 2,
        "geometry": 1,
        "spatial": 2,
        "bc_x": bc_x,
        "bc_y": bc_y,
        "steps": steps,
        "dt": dt,
    }

    info_c = {
        "cells_x": cells_x,
        "cells_y": cells_y,
        "angles": angles_c,
        "groups": groups_c,
        "materials": 2,
        "geometry": 1,
        "spatial": 2,
        "bc_x": bc_x,
        "bc_y": bc_y,
        "steps": steps,
        "dt": dt,
    }

    # Spatial Layout
    radius = 4.279960
    coords = [[(radius, radius), (0.0, radius)]]
    length_x = length_y = 2 * radius

    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)

    # Energy Grid
    edges_g, _, _ = ants.energy_grid(87, groups_u, groups_c)
    velocity_u = ants.energy_velocity(groups_u, edges_g)

    # Angular
    angle_xu, angle_yu, angle_wu = ants.angular_xy(info_u)
    angle_xc, angle_yc, angle_wc = ants.angular_xy(info_c)

    # Medium Map
    materials = np.array(["uranium-%20%", "vacuum"])
    xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials)

    weight_matrix = np.load(f"data/weight_matrix_2d/cylinder_two_material.npy")
    medium_map, xs_total_u, xs_scatter_u, xs_fission_u = ants.weight_spatial2d(
        weight_matrix, xs_total_u, xs_scatter_u, xs_fission_u
    )
    info_u["materials"] = xs_total_u.shape[0]
    info_c["materials"] = xs_total_u.shape[0]

    # Boundary conditions and external source
    external = np.zeros((1, cells_x, cells_y, 1, 1))
    edges_t = np.linspace(0, dt * steps, steps + 1)
    # Gamma half steps
    if temporal == 4:
        edges_t = ants.gamma_time_steps(edges_t)

    boundary_y = np.zeros((1, 2, 1, 1, 1))
    boundary_x = np.zeros((2, 1, 1, 1))
    boundary_x[0] = 1.0
    boundary_x = ants.boundary2d.time_dependence_decay_01(boundary_x, edges_t, 8.0)

    if temporal in [2, 4]:
        initial_flux_x = np.zeros((cells_x + 1, cells_x, angles_u**2, groups_u))
        initial_flux_y = np.zeros((cells_x, cells_x + 1, angles_u**2, groups_u))
    else:
        initial_flux_x = np.zeros((cells_x, cells_x, angles_u**2, groups_u))
        initial_flux_y = None

    return {
        "initial_flux_x": initial_flux_x,
        "initial_flux_y": initial_flux_y,
        "xs_total_u": xs_total_u,
        "xs_scatter_u": xs_scatter_u,
        "xs_fission_u": xs_fission_u,
        "velocity_u": velocity_u,
        "external": external,
        "boundary_x": boundary_x,
        "boundary_y": boundary_y,
        "medium_map": medium_map,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "angle_xu": angle_xu,
        "angle_xc": angle_xc,
        "angle_yu": angle_yu,
        "angle_yc": angle_yc,
        "angle_wu": angle_wu,
        "angle_wc": angle_wc,
        "info_u": info_u,
        "info_c": info_c,
    }


def _get_hybrid_params(groups_u, groups_c, problem_dict):
    # Get hybrid parameters
    energy_grid = ants.energy_grid(87, groups_u, groups_c)
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid
    fine_idx, coarse_idx, factor = hytools.indexing(*energy_grid)

    # Check for same number of energy groups
    if groups_u == groups_c:
        xs_total_c = problem_dict["xs_total_u"].copy()
        xs_scatter_c = problem_dict["xs_scatter_u"].copy()
        xs_fission_c = problem_dict["xs_fission_u"].copy()
        velocity_c = problem_dict["velocity_u"].copy()
    else:
        xs_collided = hytools.coarsen_materials(
            problem_dict["xs_total_u"],
            problem_dict["xs_scatter_u"],
            problem_dict["xs_fission_u"],
            edges_g[edges_gidx_u],
            edges_gidx_c,
        )
        xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
        velocity_c = hytools.coarsen_velocity(problem_dict["velocity_u"], edges_gidx_c)

    return {
        "energy_grid": energy_grid,
        "edges_g": edges_g,
        "edges_gidx_c": edges_gidx_c,
        "fine_idx": fine_idx,
        "coarse_idx": coarse_idx,
        "factor": factor,
        "xs_total_c": xs_total_c,
        "xs_scatter_c": xs_scatter_c,
        "xs_fission_c": xs_fission_c,
        "velocity_c": velocity_c,
    }


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_bdf1(angles_c, groups_c):
    temporal = 1
    angles_u = 4
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid2d.backward_euler(
        problem_dict["initial_flux_x"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_yu"],
        problem_dict["angle_yc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * steps, dtype=np.int32)
    vangles = np.array([angles_c] * steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid2d.backward_euler(
        vgroups,
        vangles,
        problem_dict["initial_flux_x"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_yu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.cn
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_cn(angles_c, groups_c):
    temporal = 2
    angles_u = 4
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid2d.crank_nicolson(
        problem_dict["initial_flux_x"],
        problem_dict["initial_flux_y"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_yu"],
        problem_dict["angle_yc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * steps, dtype=np.int32)
    vangles = np.array([angles_c] * steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid2d.crank_nicolson(
        vgroups,
        vangles,
        problem_dict["initial_flux_x"],
        problem_dict["initial_flux_y"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_yu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.bdf2
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_bdf2(angles_c, groups_c):
    temporal = 3
    angles_u = 4
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid2d.bdf2(
        problem_dict["initial_flux_x"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_yu"],
        problem_dict["angle_yc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * steps, dtype=np.int32)
    vangles = np.array([angles_c] * steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid2d.bdf2(
        vgroups,
        vangles,
        problem_dict["initial_flux_x"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_yu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab2d
@pytest.mark.hybrid
@pytest.mark.trbdf2
@pytest.mark.multigroup2d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(4, 87), (2, 87), (4, 43), (2, 43)])
def test_mg_01_tr_bdf2(angles_c, groups_c):
    temporal = 4
    angles_u = 2
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid2d.tr_bdf2(
        problem_dict["initial_flux_x"],
        problem_dict["initial_flux_y"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_yu"],
        problem_dict["angle_yc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Variable groups and angles
    vgroups = np.array([groups_c] * steps, dtype=np.int32)
    vangles = np.array([angles_c] * steps, dtype=np.int32)

    # Run vHybrid Method
    vhy_flux = vhybrid2d.tr_bdf2(
        vgroups,
        vangles,
        problem_dict["initial_flux_x"],
        problem_dict["initial_flux_y"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["boundary_y"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["delta_y"],
        problem_dict["angle_xu"],
        problem_dict["angle_yu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        print(
            tt,
            np.sum(hy_flux[tt]),
            np.sum(vhy_flux[tt]),
            np.sum(np.fabs(hy_flux[tt] - vhy_flux[tt])),
        )
        # assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


if __name__ == "__main__":
    test_mg_01_tr_bdf2(2, 87)
