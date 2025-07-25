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
from ants import hybrid2d
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
    cells_y = 100
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.backward_euler(
        initial_flux,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    cells_y = 50
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.backward_euler(
        initial_flux,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    cells_y = 100
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.crank_nicolson(
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.crank_nicolson(
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    cells_y = 100
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.bdf2(
        initial_flux,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    cells_y = 50
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.bdf2(
        initial_flux,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    cells_y = 100
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

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.tr_bdf2(
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
    ) = problems2d.manufactured_td_02(cells_x, angles, edges_t, dt, temporal=4)

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    # Indexing Parameters
    data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    fine_idx, coarse_idx, factor = data

    # Run Hybrid Method
    approx = hybrid2d.tr_bdf2(
        initial_flux_x,
        initial_flux_y,
        xs_total,
        xs_total,
        xs_scatter,
        xs_scatter,
        xs_fission,
        xs_fission,
        velocity,
        velocity,
        external,
        boundary_x,
        boundary_y,
        medium_map,
        delta_x,
        delta_y,
        angle_x,
        angle_x,
        angle_y,
        angle_y,
        angle_w,
        angle_w,
        fine_idx,
        coarse_idx,
        factor,
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
