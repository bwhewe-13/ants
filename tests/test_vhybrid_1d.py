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
from ants import vhybrid1d, hybrid1d
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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.backward_euler(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.backward_euler(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=2)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.crank_nicolson(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=2)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.crank_nicolson(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=3)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=3)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_01(cells_x, angles, edges_t, dt, temporal=4)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.tr_bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


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

    (
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        info,
    ) = problems1d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)

    edges_g, _ = ants.energy_grid(None, groups)
    vgroups = np.array([groups] * steps, dtype=np.int32)
    vangles = np.array([angles] * steps, dtype=np.int32)

    # Run vHybrid Method
    approx = vhybrid1d.tr_bdf2(
        vgroups,
        vangles,
        initial_flux,
        xs_total,
        xs_scatter,
        xs_fission,
        velocity,
        external,
        boundary_x,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        edges_g,
        info,
        info,
    )

    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
    exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    for tt in range(steps):
        assert np.isclose(approx[tt], exact[tt], atol=atol).all()


################################################################################
# Multigroup Problems
################################################################################
def _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal=1):
    # General Parameters
    cells_x = 1000
    steps = 5
    dt = 1e-8
    bc_x = [0, 0]

    # Uncollided flux dictionary
    info_u = {
        "cells_x": cells_x,
        "angles": angles_u,
        "groups": groups_u,
        "materials": 2,
        "geometry": 1,
        "spatial": 2,
        "bc_x": bc_x,
        "steps": steps,
        "dt": dt,
    }
    # Collided flux dictionary
    info_c = {
        "cells_x": cells_x,
        "angles": angles_c,
        "groups": groups_c,
        "materials": 2,
        "geometry": 1,
        "spatial": 2,
        "bc_x": bc_x,
        "steps": steps,
        "dt": dt,
    }

    # Spatial
    length = 10.0
    delta_x = np.repeat(length / cells_x, cells_x)
    edges_x = np.linspace(0, length, cells_x + 1)

    # Energy Grid
    edges_g, _, _ = ants.energy_grid(87, groups_u, groups_c)
    velocity_u = ants.energy_velocity(groups_u, edges_g)

    # Angular
    angle_xu, angle_wu = ants.angular_x(info_u)
    angle_xc, angle_wc = ants.angular_x(info_c)

    # Medium Map
    layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
    medium_map = ants.spatial1d(layers, edges_x)

    # Cross Sections - Uncollided
    materials = np.array(layers)[:, 1]
    xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials)
    velocity_u = ants.energy_velocity(groups_u, edges_g)

    # Crank-Nicolson / TR-BDF2 initial flux at cell edges
    if temporal in [2, 4]:
        initial_flux = np.zeros((cells_x + 1, angles_u, groups_u))
    else:
        initial_flux = np.zeros((cells_x, angles_u, groups_u))

    # External and boundary sources
    external = np.zeros((1, cells_x, 1, 1))
    edges_t = np.linspace(0, dt * steps, steps + 1)
    # Gamma half time steps
    if temporal == 4:
        edges_t = ants.gamma_time_steps(edges_t)
    boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
    boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)

    return {
        "initial_flux": initial_flux,
        "xs_total_u": xs_total_u,
        "xs_scatter_u": xs_scatter_u,
        "xs_fission_u": xs_fission_u,
        "velocity_u": velocity_u,
        "external": external,
        "boundary_x": boundary_x,
        "medium_map": medium_map,
        "delta_x": delta_x,
        "angle_xu": angle_xu,
        "angle_xc": angle_xc,
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


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_bdf1(angles_c, groups_c):
    temporal = 1
    angles_u = 8
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid1d.backward_euler(
        problem_dict["initial_flux"],
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
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
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
    vhy_flux = vhybrid1d.backward_euler(
        vgroups,
        vangles,
        problem_dict["initial_flux"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.cn
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_cn(angles_c, groups_c):
    temporal = 2
    angles_u = 8
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid1d.crank_nicolson(
        problem_dict["initial_flux"],
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
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
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
    vhy_flux = vhybrid1d.crank_nicolson(
        vgroups,
        vangles,
        problem_dict["initial_flux"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf2
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_bdf2(angles_c, groups_c):
    temporal = 3
    angles_u = 8
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid1d.bdf2(
        problem_dict["initial_flux"],
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
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
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
    vhy_flux = vhybrid1d.bdf2(
        vgroups,
        vangles,
        problem_dict["initial_flux"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.trbdf2
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_mg_01_tr_bdf2(angles_c, groups_c):
    temporal = 4
    angles_u = 8
    groups_u = 87

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c, temporal)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)
    steps = problem_dict["info_u"]["steps"]

    # Run Hybrid Method
    hy_flux = hybrid1d.tr_bdf2(
        problem_dict["initial_flux"],
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
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
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
    vhy_flux = vhybrid1d.tr_bdf2(
        vgroups,
        vangles,
        problem_dict["initial_flux"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary_x"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        hybrid_dict["edges_g"],
        problem_dict["info_u"],
        problem_dict["info_c"],
    )

    # Compare each time step
    for tt in range(steps):
        assert np.isclose(hy_flux[tt], vhy_flux[tt]).all()
