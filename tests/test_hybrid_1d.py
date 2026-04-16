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

import os
import tempfile

import numpy as np
import pytest

import ants
from ants import hybrid1d
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.smoke
@pytest.mark.hybrid
@pytest.mark.slab1d
@pytest.mark.bdf1
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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
    exact = mms.solution_td_01(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
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

    # Indexing Parameters
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups, groups)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method
    approx = hybrid1d.time_dependent(
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
    exact = mms.solution_td_02(centers_x, quadrature.angle_x, edges_t[1:])
    exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)

    atol = 5e-3
    assert np.isclose(approx, exact[-1], atol=atol).all()


@pytest.mark.slab1d
@pytest.mark.hybrid
@pytest.mark.bdf1
@pytest.mark.multigroup1d
@pytest.mark.parametrize(("angles_c", "groups_c"), [(8, 87), (2, 87), (8, 43), (2, 43)])
def test_slab_01_bdf1(angles_c, groups_c):
    # General Parameters
    cells_x = 1000
    angles_u = 8
    groups_u = 87
    steps = 5
    dt = 1e-08

    # Spatial
    length = 10.0
    delta_x = np.repeat(length / cells_x, cells_x)
    edges_x = np.linspace(0, length, cells_x + 1)
    layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
    medium_map = ants.spatial1d(layers, edges_x)
    geometry = GeometryData(medium_map=medium_map, delta_x=delta_x, bc_x=[0, 0])

    # Energy Grid
    energy_data = ants.energy_grid(87, groups_u, groups_c, optimize=False)
    edges_g, edges_gidx_u, edges_gidx_c = energy_data

    # Angular
    quadrature_u = ants.angular_x(angles_u, [0, 0])
    quadrature_c = ants.angular_x(angles_c, [0, 0])

    # Cross Sections - Uncollided
    materials = np.array(layers)[:, 1]
    materials_u = ants.materials(87, materials)
    materials_u.velocity = ants.energy_velocity(groups_u, edges_g)

    # Cross Sections - Collided
    xs_collided = hytools.coarsen_materials(
        materials_u.total,
        materials_u.scatter,
        materials_u.fission,
        edges_g[edges_gidx_u],
        edges_gidx_c,
    )
    xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
    velocity_c = hytools.coarsen_velocity(materials_u.velocity, edges_gidx_c)
    materials_c = MaterialData(
        total=xs_total_c,
        scatter=xs_scatter_c,
        fission=xs_fission_c,
        velocity=velocity_c,
    )

    # External and boundary sources
    initial_flux = np.zeros((cells_x, angles_u, groups_u))
    external = np.zeros((1, cells_x, 1, 1))
    edges_t = np.linspace(0, dt * steps, steps + 1)
    boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
    boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)
    sources = SourceData(
        initial_flux=initial_flux,
        external=external,
        boundary_x=boundary_x,
    )

    solver = SolverData()
    time_data = TimeDependentData(dt=dt, steps=steps, time_disc=1)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

    # Run Hybrid Method (stream all steps to temp file)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_path = f.name
    try:
        time_data.save_to_file = tmp_path
        hybrid1d.time_dependent(
            materials_u,
            materials_c,
            sources,
            geometry,
            quadrature_u,
            quadrature_c,
            solver,
            time_data,
            hybrid_data,
        )
        approx = np.load(tmp_path, mmap_mode="r")

        # Load Reference flux
        params = f"g87g{groups_c}_n8n{angles_c}_flux.npy"
        reference = np.load(
            os.path.join(PATH, f"hybrid_uranium_slab_backward_euler_{params}")
        )
        # Compare each time step
        for tt in range(steps):
            assert np.isclose(approx[tt], reference[tt]).all()
    finally:
        os.unlink(tmp_path)
