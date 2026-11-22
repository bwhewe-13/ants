########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Compact sources give the same flux as the dense sources
#
########################################################################

from dataclasses import replace

import numpy as np
import pytest

import ants
from ants.datatypes import (
    CompactSource,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.fixed1d import fixed_source as fixed_source_1d
from ants.fixed2d import fixed_source as fixed_source_2d
from ants.timed1d import time_dependent as time_dependent_1d
from ants.timed2d import time_dependent as time_dependent_2d


def _materials(groups, moments=0):
    # Two materials with downscattering
    total = np.array([[1.0, 1.5, 2.0], [0.5, 0.8, 1.2]])[:, :groups]
    scatter = np.array(
        [
            [[0.5, 0.0, 0.0], [0.3, 0.6, 0.0], [0.1, 0.2, 0.8]],
            [[0.2, 0.0, 0.0], [0.1, 0.4, 0.0], [0.0, 0.1, 0.6]],
        ]
    )[:, :groups, :groups]
    if moments > 0:
        # Linearly anisotropic scattering
        scatter = np.stack([scatter, 0.3 * scatter], axis=-1)
    return MaterialData(
        total=total,
        scatter=scatter,
        fission=np.zeros((2, groups, groups)),
        velocity=np.array([10.0, 5.0, 1.0])[:groups],
    )


def _problem_1d(geometry=1, bc_x=[0, 0], moments=0):
    cells_x, angles, groups = 20, 4, 3
    quadrature = ants.angular_x(angles, bc_x=bc_x)
    geometry = GeometryData(
        medium_map=np.repeat(np.array([0, 1], dtype=np.int32), cells_x // 2),
        delta_x=np.full(cells_x, 0.25),
        bc_x=bc_x,
        geometry=geometry,
    )
    # Angle dependent external source in the first group only
    external = np.zeros((cells_x, angles, groups))
    external[5:10, :, 0] = 1.0 + quadrature.angle_x[None, :]
    # Boundary source in the first group on the left side
    boundary_x = np.zeros((2, angles, groups))
    boundary_x[0, :, 0] = 1.0
    return _materials(groups, moments), external, boundary_x, geometry, quadrature


def _problem_2d(moments=0):
    cells_x, cells_y, angles, groups = 6, 5, 4, 2
    quadrature = ants.angular_xy(angles)
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    medium_map[3:] = 1
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.full(cells_x, 0.3),
        delta_y=np.full(cells_y, 0.4),
        geometry=3,
    )
    # Angle dependent external source in the second group only
    external = np.zeros((cells_x, cells_y, angles**2, groups))
    external[1:3, 2:4, :, 1] = 1.0 + quadrature.angle_x * quadrature.angle_y
    # Boundary source in the first group on the bottom side
    boundary_x = np.zeros((2, cells_y, 1, 1))
    boundary_y = np.zeros((2, cells_x, angles**2, groups))
    boundary_y[0, 2:5, :, 0] = 1.0
    materials = _materials(groups, moments)
    return materials, external, boundary_x, boundary_y, geometry, quadrature


########################################################################
# Data structure
########################################################################


def test_round_trip():
    _, external, boundary_x, _, _ = _problem_1d()
    for dense in (external, boundary_x):
        compact = CompactSource.from_dense(dense)
        assert compact.index.shape == dense.shape[-2:]
        assert np.array_equal(compact.to_dense(), dense)
    # One group with distinct angles is stored as one slab per angle
    compact = CompactSource.from_dense(external)
    assert compact.values.shape == (external.shape[0], external.shape[1])
    assert compact.nbytes < external.nbytes


def test_round_trip_time():
    dense = np.zeros((4, 2, 1, 3))
    dense[:, 0, 0, 1] = [1.0, 0.5, 0.25, 0.0]
    compact = CompactSource.from_dense(dense, time=True)
    assert compact.steps == 4
    assert np.array_equal(compact.to_dense(), dense)
    # Same profile with a time amplitude
    values = np.array([[1.0], [0.0]])
    index = np.array([[-1, 0, -1]])
    scaled = CompactSource(values, index, scale=[1.0, 0.5, 0.25, 0.0])
    assert np.array_equal(scaled.to_dense(), dense)


def test_steady_state_rejects_time_dependence():
    materials, external, boundary_x, geometry, quadrature = _problem_1d()
    compact = CompactSource.from_dense(external[None], time=True)
    compact.scale = np.ones(3)
    sources = SourceData(external=compact, boundary_x=boundary_x)
    with pytest.raises(ValueError, match="time dependence"):
        fixed_source_1d(materials, sources, geometry, quadrature, SolverData())


########################################################################
# One-dimensional problems
########################################################################


@pytest.mark.smoke
@pytest.mark.parametrize("geometry", [1, 2])
@pytest.mark.parametrize(
    "solver",
    [
        SolverData(),
        SolverData(angular=True),
        SolverData(flux_at_edges=1),
        SolverData(parallel=ants.datatypes.ParallelType.GROUP, num_threads=2),
        SolverData(mg_solver=ants.datatypes.MultigroupSolver.DMD),
    ],
)
def test_fixed1d(geometry, solver):
    bc_x = [1, 0] if geometry == 2 else [0, 0]
    materials, external, boundary_x, geometry, quadrature = _problem_1d(geometry, bc_x)
    dense = SourceData(external=external, boundary_x=boundary_x)
    compact = SourceData(
        external=CompactSource.from_dense(external),
        boundary_x=CompactSource.from_dense(boundary_x),
    )
    reference = fixed_source_1d(materials, dense, geometry, quadrature, solver)
    approx = fixed_source_1d(materials, compact, geometry, quadrature, solver)
    assert np.all(reference > 0.0) or solver.angular
    assert np.allclose(approx, reference, atol=1e-12)


@pytest.mark.parametrize("angular", [False, True])
def test_fixed1d_anisotropic(angular):
    materials, external, boundary_x, geometry, quadrature = _problem_1d(moments=1)
    solver = SolverData(angular=angular)
    dense = SourceData(external=external, boundary_x=boundary_x)
    compact = SourceData(
        external=CompactSource.from_dense(external),
        boundary_x=CompactSource.from_dense(boundary_x),
    )
    reference = fixed_source_1d(materials, dense, geometry, quadrature, solver)
    approx = fixed_source_1d(materials, compact, geometry, quadrature, solver)
    assert np.allclose(approx, reference, atol=1e-12)


def _time_sources_1d(time_disc, steps, external, angles):
    # Dense half angle boundary decaying in time and the compact version
    # with the boundary profile and a time amplitude
    bc_steps = 2 * steps if time_disc == 4 else steps
    q_steps = {1: steps, 2: steps + 1, 3: steps, 4: 2 * steps + 1}[time_disc]
    scale = np.exp(-np.arange(bc_steps, dtype=float))
    values = np.array([[1.0], [0.0]])
    index = np.array([[0, -1, -1]])
    compact_x = CompactSource(values, index, scale=scale)
    dense_x = compact_x.to_dense()
    # External source with a linear time dependence
    q_scale = 1.0 + 0.1 * np.arange(q_steps)
    dense_q = external[None] * q_scale[:, None, None, None]
    compact_q = CompactSource.from_dense(external)
    compact_q.scale = q_scale
    # CN and TR-BDF2 take the initial flux at the cell edges
    edges = 1 if time_disc in (2, 4) else 0
    initial = np.zeros((external.shape[0] + edges, angles, external.shape[2]))
    dense = SourceData(initial_flux=initial, external=dense_q, boundary_x=dense_x)
    compact = replace(dense, external=compact_q, boundary_x=compact_x)
    return dense, compact


@pytest.mark.smoke
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
def test_timed1d(time_disc):
    materials, external, boundary_x, geometry, quadrature = _problem_1d()
    steps = 3
    dense, compact = _time_sources_1d(
        time_disc, steps, external, quadrature.angle_x.size
    )
    time_data = TimeDependentData(steps=steps, dt=0.1, time_disc=time_disc)
    solver = SolverData()
    reference = time_dependent_1d(
        materials, dense, geometry, quadrature, solver, time_data
    )
    approx = time_dependent_1d(
        materials, compact, geometry, quadrature, solver, time_data
    )
    assert np.abs(reference).max() > 0.0
    assert np.allclose(approx, reference, atol=1e-12)


########################################################################
# Two-dimensional problems
########################################################################


@pytest.mark.smoke
@pytest.mark.parametrize(
    "solver",
    [
        SolverData(),
        SolverData(angular=True),
        SolverData(flux_at_edges=1),
        SolverData(parallel=ants.datatypes.ParallelType.GROUP, num_threads=2),
    ],
)
def test_fixed2d(solver):
    problem = _problem_2d()
    materials, external, boundary_x, boundary_y, geometry, quadrature = problem
    dense = SourceData(external=external, boundary_x=boundary_x, boundary_y=boundary_y)
    compact = SourceData(
        external=CompactSource.from_dense(external),
        boundary_x=boundary_x,
        boundary_y=CompactSource.from_dense(boundary_y),
    )
    reference = fixed_source_2d(materials, dense, geometry, quadrature, solver)
    approx = fixed_source_2d(materials, compact, geometry, quadrature, solver)
    if isinstance(reference, tuple):
        for ref, apx in zip(reference, approx):
            assert np.allclose(apx, ref, atol=1e-12)
    else:
        assert np.allclose(approx, reference, atol=1e-12)


def test_fixed2d_anisotropic():
    problem = _problem_2d(moments=1)
    materials, external, boundary_x, boundary_y, geometry, quadrature = problem
    solver = SolverData(angular=True)
    dense = SourceData(external=external, boundary_x=boundary_x, boundary_y=boundary_y)
    compact = SourceData(
        external=CompactSource.from_dense(external),
        boundary_x=boundary_x,
        boundary_y=CompactSource.from_dense(boundary_y),
    )
    reference = fixed_source_2d(materials, dense, geometry, quadrature, solver)
    approx = fixed_source_2d(materials, compact, geometry, quadrature, solver)
    assert np.allclose(approx, reference, atol=1e-12)


@pytest.mark.parametrize("time_disc", [1, 3])
def test_timed2d(time_disc):
    problem = _problem_2d()
    materials, external, boundary_x, boundary_y, geometry, quadrature = problem
    steps = 3
    # Bottom boundary in the first group, turned off after the first step
    values = np.zeros((2, external.shape[0], 1))
    values[0, 2:5, 0] = 1.0
    index = np.array([[[0, -1]], [[-1, -1]], [[-1, -1]]])
    compact_y = CompactSource(values, index)
    dense_y = compact_y.to_dense()
    initial = np.zeros(external.shape)
    dense = SourceData(
        initial_flux=initial,
        external=external[None],
        boundary_x=boundary_x[None],
        boundary_y=dense_y,
    )
    compact = replace(
        dense, external=CompactSource.from_dense(external), boundary_y=compact_y
    )
    time_data = TimeDependentData(steps=steps, dt=0.1, time_disc=time_disc)
    solver = SolverData()
    reference = time_dependent_2d(
        materials, dense, geometry, quadrature, solver, time_data
    )
    approx = time_dependent_2d(
        materials, compact, geometry, quadrature, solver, time_data
    )
    assert np.abs(reference).max() > 0.0
    assert np.allclose(approx, reference, atol=1e-12)
