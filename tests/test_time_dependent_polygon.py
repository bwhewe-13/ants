########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for two-dimensional time-dependent problems on polygon meshes.
# Includes Backward Euler, Crank-Nicolson, BDF2, TR-BDF2 and the hybrid
# methods.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import fixed2d, hybrid2d, timed2d, vhybrid2d
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    SpatialDiscretization,
    TemporalDiscretization,
    TimeDependentData,
)
from ants.utils import hybrid as hytools
from ants.utils import mesh2d
from tests import problems2d


def _slab_problem(temporal):
    steps = 4
    edges_t = np.linspace(0, 0.4, steps + 1)
    mat_data, sources, geometry, quadrature, solver, time_data = (
        problems2d.manufactured_td_01(8, 4, edges_t, 0.1, temporal=temporal)
    )
    geometry.space_disc = SpatialDiscretization.STEP
    return mat_data, sources, geometry, quadrature, solver, time_data


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.time_dependent
@pytest.mark.parametrize("temporal", [1, 3])
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_rectangle_cells(temporal, spatial):
    mat_data, sources, geometry, quadrature, solver, time_data = _slab_problem(temporal)
    geometry.space_disc = spatial
    reference = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    sources, geometry = problems2d.slab_to_polygon(sources, geometry)
    geometry.space_disc = spatial
    flux = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(flux.reshape(reference.shape), reference, atol=1e-12)


@pytest.mark.polygon2d
@pytest.mark.time_dependent
@pytest.mark.hybrid
@pytest.mark.parametrize("temporal", [1, 3])
def test_rectangle_cells_hybrid(temporal):
    mat_data, sources, geometry, quadrature, solver, time_data = _slab_problem(temporal)
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, 1, 1)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    groups_c = np.ones((time_data.steps,), dtype=np.int32)
    angles_c = 4 * np.ones((time_data.steps,), dtype=np.int32)
    reference = hybrid2d.time_dependent(
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
    reference_v = vhybrid2d.time_dependent(
        mat_data,
        groups_c,
        sources,
        geometry,
        quadrature,
        angles_c,
        solver,
        time_data,
        edges_g,
    )

    sources, geometry = problems2d.slab_to_polygon(sources, geometry)
    flux = hybrid2d.time_dependent(
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
    assert np.allclose(flux.reshape(reference.shape), reference, atol=1e-12)
    flux = vhybrid2d.time_dependent(
        mat_data,
        groups_c,
        sources,
        geometry,
        quadrature,
        angles_c,
        solver,
        time_data,
        edges_g,
    )
    assert np.allclose(flux.reshape(reference_v.shape), reference_v, atol=1e-12)


def _hexagon_problem(steps, time_disc):
    # Infinite medium: phi(t) = q / sa + (phi_0 - q / sa) exp(-v sa t)
    mesh = mesh2d.hexagon_mesh(0.5, mesh2d.hexagon_boundary((0.0, 0.0), 2.0, "pointy"))
    quadrature = ants.angular_xy(6)
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.4]]]),
        fission=np.zeros((1, 1, 1)),
        velocity=np.array([2.0]),
    )
    sources = SourceData(
        external=np.ones((1, mesh.cells, 1, 1, 1)),
        boundary_x=np.zeros((1, 1, 1, 1, 1)),
        boundary_y=np.zeros((1, 1, 1, 1, 1)),
        initial_flux=0.2 * np.ones((mesh.cells, 1, 36, 1)),
        initial_flux_x=0.2 * np.ones((mesh.faces, 1, 36, 1)),
        initial_flux_y=0.2 * np.ones((mesh.cells, 1, 36, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=[1] * mesh.sides,
    )
    time_data = TimeDependentData(steps=steps, dt=1.0 / steps, time_disc=time_disc)
    return mat_data, sources, geometry, quadrature, SolverData(), time_data


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.time_dependent
@pytest.mark.parametrize(
    ("time_disc", "order"),
    [
        (TemporalDiscretization.BDF1, 1),
        (TemporalDiscretization.CN, 2),
        (TemporalDiscretization.BDF2, 2),
        (TemporalDiscretization.TR_BDF2, 2),
    ],
)
def test_infinite_medium_order(time_disc, order):
    exact = 1.0 / 0.6 + (0.2 - 1.0 / 0.6) * np.exp(-2.0 * 0.6)
    errors = []
    for steps in [10, 20]:
        flux = timed2d.time_dependent(*_hexagon_problem(steps, time_disc))
        errors.append(np.max(np.fabs(flux - exact)))
    assert np.fabs(np.log2(errors[0] / errors[1]) - order) < 0.15


def _triangle_problem(time_disc, steps, dt):
    mesh = mesh2d.triangle_mesh(np.linspace(0, 2, 9), np.linspace(0, 1, 5), "crossed")
    quadrature = ants.angular_xy(4)
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.zeros((1, 1, 1)),
        velocity=np.array([10.0]),
    )
    external = (1 + mesh.cell_centroid[:, 0] ** 2).reshape(1, mesh.cells, 1, 1, 1)
    boundary = np.zeros((1, 1, mesh.boundary_faces.size, 1, 1))
    boundary[0, 0, mesh.face_side[mesh.boundary_faces] == 0] = 1.0
    sources = SourceData(
        external=external,
        boundary_x=boundary,
        boundary_y=np.zeros((1, 1, 1, 1, 1)),
        initial_flux_x=np.zeros((mesh.faces, 1, 16, 1)),
        initial_flux_y=np.zeros((mesh.cells, 1, 16, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=[0, 1, 0, 0],
    )
    time_data = TimeDependentData(steps=steps, dt=dt, time_disc=time_disc)
    return mat_data, sources, geometry, quadrature, SolverData(), time_data


@pytest.mark.polygon2d
@pytest.mark.time_dependent
@pytest.mark.parametrize(
    "time_disc", [TemporalDiscretization.CN, TemporalDiscretization.TR_BDF2]
)
@pytest.mark.parametrize(
    "spatial",
    [
        SpatialDiscretization.STEP,
        SpatialDiscretization.DIAMOND,
        SpatialDiscretization.STEP_CHARACTERISTIC,
    ],
)
def test_steady_state(time_disc, spatial):
    mat_data, sources, geometry, quadrature, solver, time_data = _triangle_problem(
        time_disc, 480, 0.05
    )
    geometry.space_disc = spatial
    steady_sources = SourceData(
        external=sources.external[0],
        boundary_x=sources.boundary_x[0],
        boundary_y=sources.boundary_y[0],
    )
    steady = fixed2d.fixed_source(
        mat_data, steady_sources, geometry, quadrature, solver
    )
    flux = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(flux, steady, rtol=1e-8)


@pytest.mark.polygon2d
@pytest.mark.time_dependent
@pytest.mark.hybrid
@pytest.mark.parametrize(
    "time_disc", [TemporalDiscretization.CN, TemporalDiscretization.TR_BDF2]
)
def test_hybrid_edges(time_disc):
    mat_data, sources, geometry, quadrature, solver, time_data = _triangle_problem(
        time_disc, 4, 0.1
    )
    rng = np.random.default_rng(0)
    sources.initial_flux_x = rng.random(sources.initial_flux_x.shape)
    sources.initial_flux_y = rng.random(sources.initial_flux_y.shape)
    reference = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )

    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, 1, 1)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    flux = hybrid2d.time_dependent(
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
    assert np.allclose(flux, reference, rtol=1e-10)
    flux = vhybrid2d.time_dependent(
        mat_data,
        np.ones((4,), dtype=np.int32),
        sources,
        geometry,
        quadrature,
        4 * np.ones((4,), dtype=np.int32),
        solver,
        time_data,
        edges_g,
    )
    assert np.allclose(flux, reference, rtol=1e-10)
