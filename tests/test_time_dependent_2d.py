########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test two-dimensional time-dependent problems
#
########################################################################

import os
import tempfile

import numpy as np
import pytest

import ants
from ants import fixed2d, timed2d
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from tests import problems2d as prob


def _simple_problem(cells, angles, boundary_x, boundary_y, steps, dt):
    """Absorbing 2D slab, constant source, returns (mat_data, sources,
    geometry, quadrature, solver, time_data)."""
    length = 1.0
    delta_x = np.repeat(length / cells, cells)
    delta_y = np.repeat(length / cells, cells)

    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.ones((1,)),
    )

    external = 0.5 * np.ones((1, cells, cells, 1, 1))
    initial_flux = np.zeros((cells, cells, quadrature.angle_x.size, 1))

    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        initial_flux=initial_flux,
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=steps, dt=dt, time_disc=1)

    return mat_data, sources, geometry, quadrature, solver, time_data


@pytest.mark.slab2d
@pytest.mark.bdf1
def test_broadcast_boundary_bdf1():
    """Broadcast boundary (angle dim = 1) converges to fixed-source solution."""
    cells = 8
    angles = 4
    steps = 200
    dt = 1.0

    # Broadcast boundary: same value for all angles
    boundary_x = np.zeros((1, 2, cells, 1, 1))
    boundary_y = np.zeros((1, 2, cells, 1, 1))

    mat_data, sources, geometry, quadrature, solver, time_data = _simple_problem(
        cells, angles, boundary_x, boundary_y, steps, dt
    )

    # Fixed-source reference (steady state)
    fixed_sources = SourceData(
        external=sources.external[0],
        boundary_x=sources.boundary_x[0],
        boundary_y=sources.boundary_y[0],
    )
    fixed_flux = fixed2d.fixed_source(
        mat_data, fixed_sources, geometry, quadrature, solver
    )

    timed_flux = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )

    assert np.isclose(
        fixed_flux[:, :, 0], timed_flux[:, :, 0], atol=1e-3
    ).all(), (
        "Broadcast boundary: time-dependent did not converge to fixed-source solution"
    )


@pytest.mark.slab2d
@pytest.mark.parametrize("time_disc", [1, 2, 3, 4])
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_steady_state(time_disc, spatial):
    """Converges to the fixed-source solution, CN and TR-BDF2 carry the
    cell edges."""
    cells = 8
    boundary_x = np.zeros((1, 2, cells, 1, 1))
    boundary_x[0, 0] = 1.0
    boundary_y = np.zeros((1, 2, cells, 1, 1))

    mat_data, sources, geometry, quadrature, solver, time_data = _simple_problem(
        cells, 4, boundary_x, boundary_y, 400, 1.0
    )
    geometry.space_disc = spatial
    time_data.time_disc = time_disc
    if time_disc in (2, 4):
        sources.initial_flux_x = np.zeros((cells + 1, cells, 16, 1))
        sources.initial_flux_y = np.zeros((cells, cells + 1, 16, 1))

    fixed_sources = SourceData(
        external=sources.external[0],
        boundary_x=sources.boundary_x[0],
        boundary_y=sources.boundary_y[0],
    )
    fixed_flux = fixed2d.fixed_source(
        mat_data, fixed_sources, geometry, quadrature, solver
    )
    timed_flux = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    assert np.allclose(fixed_flux, timed_flux, rtol=1e-10)


@pytest.mark.slab2d
@pytest.mark.bdf1
def test_half_angle_boundary_bdf1():
    """Half-angle boundary (manufactured_td_02) runs without error and
    produces physically reasonable flux values."""
    cells = 8
    angles = 4
    steps = 5
    dt = 0.1

    T = steps * dt
    edges_t = np.linspace(0, T, steps + 1)

    mat_data, sources, geometry, quadrature, solver, time_data = (
        prob.manufactured_td_02(cells, angles, edges_t, dt, temporal=1)
    )

    flux = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )

    assert flux.shape == (cells, cells, 1), "Unexpected flux shape"
    assert np.isfinite(flux).all(), "Non-finite values in flux"
    assert (flux >= 0.0).all(), "Negative flux values"


@pytest.mark.slab2d
@pytest.mark.bdf1
def test_save_to_file_bdf1():
    """save_to_file produces a file whose contents match in-memory run."""
    cells = 6
    angles = 4
    steps = 10
    dt = 1.0

    boundary_x = np.zeros((1, 2, cells, 1, 1))
    boundary_y = np.zeros((1, 2, cells, 1, 1))

    mat_data, sources, geometry, quadrature, solver, _ = _simple_problem(
        cells, angles, boundary_x, boundary_y, steps, dt
    )

    # In-memory run
    time_data_mem = TimeDependentData(steps=steps, dt=dt, time_disc=1)
    flux_mem = timed2d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data_mem
    )

    # File-backed run
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        fname = f.name
    try:
        time_data_file = TimeDependentData(
            steps=steps, dt=dt, time_disc=1, save_to_file=fname
        )
        flux_file = timed2d.time_dependent(
            mat_data, sources, geometry, quadrature, solver, time_data_file
        )

        assert os.path.exists(fname), "Output file not created"
        assert (
            flux_file.shape == flux_mem.shape
        ), "Shape mismatch between file and memory runs"
        np.testing.assert_allclose(
            flux_file,
            flux_mem,
            rtol=1e-12,
            err_msg="File-backed flux differs from in-memory flux",
        )
    finally:
        os.unlink(fname)
