########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test one-dimensional time-dependent problems
#
########################################################################

import os

import numpy as np
import pytest

from ants import fixed1d, timed1d
from ants.datatypes import TimeDependentData
from tests import problems1d as prob


@pytest.mark.slab1d
@pytest.mark.bdf1
@pytest.mark.parametrize(("boundary"), [[0, 0], [1, 0], [0, 1]])
def test_reed_bdf1(boundary):
    mat_data, sources, geometry, quadrature, solver = prob.reeds(boundary)

    # Get Fixed Source
    fixed_flux = fixed1d.fixed_source(mat_data, sources, geometry, quadrature, solver)

    # Set time dependent variables
    time_data = TimeDependentData(steps=100, dt=1.0, time_disc=1)
    sources.external = sources.external[None, ...].copy()
    sources.boundary_x = sources.boundary_x[None, ...].copy()
    sources.initial_flux = np.zeros(
        (geometry.delta_x.size, quadrature.angle_x.size, mat_data.total.shape[0])
    )
    # Get Time Dependent
    timed_flux = timed1d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )

    assert np.isclose(fixed_flux[:, 0], timed_flux[-1, :, 0]).all(), "Incorrect Flux"


@pytest.mark.slab1d
@pytest.mark.bdf2
@pytest.mark.parametrize(("boundary"), [[0, 0], [1, 0], [0, 1]])
def test_reed_bdf2(boundary):
    mat_data, sources, geometry, quadrature, solver = prob.reeds(boundary)
    # Get Fixed Source
    fixed_flux = fixed1d.fixed_source(mat_data, sources, geometry, quadrature, solver)
    # Set time dependent variables
    time_data = TimeDependentData(steps=100, dt=1.0, time_disc=3)
    print(sources.external.shape, sources.boundary_x.shape)
    sources.external = sources.external[None, ...].copy()
    sources.boundary_x = sources.boundary_x[None, ...].copy()
    print(sources.external.shape, sources.boundary_x.shape)
    sources.initial_flux = np.zeros(
        (geometry.delta_x.size, quadrature.angle_x.size, mat_data.total.shape[1])
    )
    # Get Time Dependent
    timed_flux = timed1d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    assert np.isclose(fixed_flux[:, 0], timed_flux[-1, :, 0]).all(), "Incorrect Flux"


@pytest.mark.sphere1d
@pytest.mark.bdf1
@pytest.mark.multigroup1d
def test_sphere_01_bdf1():
    mat_data, sources, geometry, quadrature, solver, time_data = prob.sphere_01("timed")

    flux = timed1d.time_dependent(
        mat_data, sources, geometry, quadrature, solver, time_data
    )
    ref_file_name = "uranium_sphere_backward_euler_flux.npy"
    reference = np.load(os.path.join(prob.PATH, ref_file_name))
    for tt in range(time_data.steps):
        assert np.isclose(flux[tt], reference[tt]).all()
