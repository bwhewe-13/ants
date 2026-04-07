########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Default Problems for Testing
#
########################################################################

import numpy as np

import ants
from ants.datatypes import (
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import manufactured_1d as mms

# Path for reference solutions
PATH = "data/references_multigroup/"


def reeds(bc_x):
    cells_x = 160
    angles = 4

    # Spatial
    length_x = 8.0 if np.sum(bc_x) > 0.0 else 16.0
    edges_x = np.linspace(0, length_x, cells_x + 1)

    mat_data = MaterialData(
        total=np.array([[1.0], [0.0], [5.0], [50.0]]),
        scatter=np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]]),
        fission=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
        velocity=np.ones((1,)),
    )

    sources = SourceData(
        external=ants.external1d.reeds(edges_x, bc_x),
        boundary_x=np.zeros((2, 1, 1)),
    )

    # Layout and Materials
    if bc_x == [0, 0]:
        layout = [
            [0, "scattering", "0-4, 12-16"],
            [1, "vacuum", "4-5, 11-12"],
            [2, "absorber", "5-6, 10-11"],
            [3, "source", "6-10"],
        ]
    elif bc_x == [0, 1]:
        layout = [
            [0, "scattering", "0-4"],
            [1, "vacuum", "4-5"],
            [2, "absorber", "5-6"],
            [3, "source", "6-8"],
        ]
    elif bc_x == [1, 0]:
        layout = [
            [0, "scattering", "4-8"],
            [1, "vacuum", "3-4"],
            [2, "absorber", "2-3"],
            [3, "source", "0-2"],
        ]
    geometry = GeometryData(
        medium_map=ants.spatial1d(layout, edges_x),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        bc_x=bc_x,
        geometry=1,
    )
    quadrature = ants.angular_x(angles, bc_x=bc_x)
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver


def sphere_01(ptype):
    # ptype can be "timed", "fixed", or "critical"
    cells_x = 100
    angles = 4
    groups = 87

    # Spatial
    length_x = 10.0
    edges_x = np.linspace(0, length_x, cells_x + 1)

    # Energy Grid
    edges_g, edges_gidx = ants.energy_grid(groups, 87)

    # Layout and Materials
    layout = [
        [0, "uranium-%20%", "0-4"],
        [1, "uranium-%0%", "4-6"],
        [2, "stainless-steel-440", "6-10"],
    ]

    # Materials
    mat_data = ants.materials(groups, np.array(layout)[:, 1], datatype=True)
    mat_data.velocity = ants.energy_velocity(groups, edges_g)

    # Sources
    if ptype == "timed":
        sources = SourceData(
            initial_flux=np.zeros((cells_x, angles, groups)),
            external=np.zeros((1, cells_x, 1, 1)),
            boundary_x=ants.boundary1d.deuterium_tritium(1, edges_g)[None, ...],
        )
    else:
        sources = SourceData(
            external=np.zeros((cells_x, 1, 1)),
            boundary_x=ants.boundary1d.deuterium_tritium(1, edges_g),
        )

    geometry = GeometryData(
        medium_map=ants.spatial1d(layout, edges_x),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        bc_x=[1, 0],
        geometry=2,
    )
    quadrature = ants.angular_x(angles, bc_x=[1, 0])
    solver = SolverData()
    time_data = TimeDependentData(steps=5, dt=1e-8)

    return mat_data, sources, geometry, quadrature, solver, time_data


def manufactured_ss_01(cells_x, angles):
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )

    # Spatial
    length_x = 1.0

    # Sources
    sources = SourceData(
        external=np.ones((cells_x, 1, 1)),
        boundary_x=np.array([[[1.0]], [[0.0]]]),
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells_x), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
    )
    quadrature = ants.angular_x(angles)
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver


def manufactured_ss_02(cells_x, angles):
    mat_data, sources, geometry, quadrature, solver = manufactured_ss_01(
        cells_x, angles
    )
    sources.external *= 0.5
    return mat_data, sources, geometry, quadrature, solver


def manufactured_ss_03(cells_x, angles):
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.9]]]),
        fission=np.array([[[0.0]]]),
    )

    # Spatial
    length_x = 1.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    quadrature = ants.angular_x(angles)

    # Sources
    sources = SourceData(
        external=ants.external1d.manufactured_ss_03(centers_x, quadrature.angle_x),
        boundary_x=ants.boundary1d.manufactured_ss_03(quadrature.angle_x),
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells_x), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver


def manufactured_ss_04(cells_x, angles):
    mat_data = MaterialData(
        total=np.array([[1.0], [1.0]]),
        scatter=np.array([[[0.3]], [[0.9]]]),
        fission=np.array([[[0.0]], [[0.0]]]),
    )

    # Spatial
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    quadrature = ants.angular_x(angles)

    # Sources
    sources = SourceData(
        external=ants.external1d.manufactured_ss_04(centers_x, quadrature.angle_x),
        boundary_x=ants.boundary1d.manufactured_ss_04(),
    )

    layout = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    geometry = GeometryData(
        medium_map=ants.spatial1d(layout, edges_x),
        delta_x=np.repeat(length_x / cells_x, cells_x),
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver


def manufactured_ss_05(cells_x, angles):
    mat_data, _, geometry, quadrature, solver = manufactured_ss_04(cells_x, angles)

    # Spatial
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Sources
    sources = SourceData(
        external=ants.external1d.manufactured_ss_05(centers_x, quadrature.angle_x),
        boundary_x=ants.boundary1d.manufactured_ss_05(),
    )
    return mat_data, sources, geometry, quadrature, solver


def manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1):
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.ones((1,)),
    )

    # Spatial
    length_x = 2
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    quadrature = ants.angular_x(angles)
    mu = quadrature.angle_x

    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_01(centers_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, mu, edges_t)[1:]
    # Crank Nicolson
    elif temporal == 2:
        initial_flux = mms.solution_td_01(edges_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, mu, edges_t)
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_01(centers_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, mu, edges_t)[1:]
    # TR-BDF2
    elif temporal == 4:
        initial_flux = mms.solution_td_01(edges_x, mu, np.array([0.0]))[0]
        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external1d.manufactured_td_01(centers_x, mu, gamma_steps)

    sources = SourceData(
        initial_flux=initial_flux,
        external=external,
        boundary_x=2 * np.ones((1, 2, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells_x), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=edges_t.shape[0] - 1, dt=dt, time_disc=temporal)

    return mat_data, sources, geometry, quadrature, solver, time_data


def manufactured_td_02(cells_x, angles, edges_t, dt, temporal=1):
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.ones((1,)),
    )

    # Spatial
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    quadrature = ants.angular_x(angles)
    mu = quadrature.angle_x

    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_02(centers_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_02(centers_x, mu, edges_t)[1:]
        boundary_x = ants.boundary1d.manufactured_td_02(mu, edges_t)[1:]
    # Crank Nicolson
    elif temporal == 2:
        initial_flux = mms.solution_td_02(edges_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_02(centers_x, mu, edges_t)
        boundary_x = ants.boundary1d.manufactured_td_02(mu, edges_t)[1:]
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_02(centers_x, mu, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_02(centers_x, mu, edges_t)[1:]
        boundary_x = ants.boundary1d.manufactured_td_02(mu, edges_t)[1:]
    # TR-BDF2
    elif temporal == 4:
        initial_flux = mms.solution_td_02(edges_x, mu, np.array([0.0]))[0]
        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external1d.manufactured_td_02(centers_x, mu, gamma_steps)
        boundary_x = ants.boundary1d.manufactured_td_02(mu, gamma_steps)[1:]

    sources = SourceData(
        initial_flux=initial_flux, external=external, boundary_x=boundary_x
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells_x), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=edges_t.shape[0] - 1, dt=dt, time_disc=temporal)

    return mat_data, sources, geometry, quadrature, solver, time_data
