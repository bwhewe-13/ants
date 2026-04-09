########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Default Problems for Testing Two-dimensional problems
#
########################################################################

import numpy as np

import ants
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.utils import manufactured_2d as mms

# Path for reference solutions
PATH = "data/references_multigroup/"


def manufactured_ss_01(cells, angles):
    # Spatial dimension x
    length_x = 1.0
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Spatial dimension y
    length_y = 1.0
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )

    # Externals
    external = 0.5 * np.ones((cells, cells, 1, 1))
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )

    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver, edges_x, edges_y


def manufactured_ss_02(cells, angles):
    # Spatial dimension x
    length_x = 1.0
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Spatial dimension y
    length_y = 1.0
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )

    # Externals
    external = np.ones((cells, cells, 1, 1))
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )

    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver, edges_x, edges_y


def manufactured_ss_03(cells, angles):
    # Spatial dimension x
    length_x = 2.0
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Spatial dimension y
    length_y = 2.0
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.array([[[0.0]]]),
    )

    # Externals
    external = ants.external2d.manufactured_ss_03(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_03(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )

    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver, edges_x, edges_y


def manufactured_ss_04(cells, angles):
    # Spatial dimension x
    length_x = 2.0
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Spatial dimension y
    length_y = 2.0
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.array([[[0.0]]]),
    )

    # Externals
    external = ants.external2d.manufactured_ss_04(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_04(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )

    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver, edges_x, edges_y


def manufactured_td_01(cells, angles, edges_t, dt, temporal=1):
    # Spatial Dimensions
    length_x = 2.0
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    length_y = 2.0
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)
    angle_x = quadrature.angle_x
    angle_y = quadrature.angle_y

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.ones((1,)),
    )

    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_01(
            centers_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_01(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )[1:]
        sources = SourceData(
            initial_flux=initial_flux,
            external=external,
            boundary_x=2 * np.ones((1, 2, 1, 1, 1)),
            boundary_y=2 * np.ones((1, 2, 1, 1, 1)),
        )
    # Crank Nicolson
    elif temporal == 2:
        initial_flux_x = mms.solution_td_01(
            edges_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        initial_flux_y = mms.solution_td_01(
            centers_x, edges_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_01(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )
        sources = SourceData(
            initial_flux_x=initial_flux_x,
            initial_flux_y=initial_flux_y,
            external=external,
            boundary_x=2 * np.ones((1, 2, 1, 1, 1)),
            boundary_y=2 * np.ones((1, 2, 1, 1, 1)),
        )
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_01(
            centers_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_01(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )[1:]
        sources = SourceData(
            initial_flux=initial_flux,
            external=external,
            boundary_x=2 * np.ones((1, 2, 1, 1, 1)),
            boundary_y=2 * np.ones((1, 2, 1, 1, 1)),
        )
    # TR-BDF2
    elif temporal == 4:
        initial_flux_x = mms.solution_td_01(
            edges_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        initial_flux_y = mms.solution_td_01(
            centers_x, edges_y, angle_x, angle_y, np.array([0.0])
        )[0]
        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external2d.manufactured_td_01(
            centers_x, centers_y, angle_x, angle_y, gamma_steps
        )
        sources = SourceData(
            initial_flux_x=initial_flux_x,
            initial_flux_y=initial_flux_y,
            external=external,
            boundary_x=2 * np.ones((1, 2, 1, 1, 1)),
            boundary_y=2 * np.ones((1, 2, 1, 1, 1)),
        )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=edges_t.shape[0] - 1, dt=dt, time_disc=temporal)

    return mat_data, sources, geometry, quadrature, solver, time_data


def manufactured_td_02(cells, angles, edges_t, dt, temporal=1):
    # Spatial Dimensions
    length_x = np.pi
    delta_x = np.repeat(length_x / cells, cells)
    edges_x = np.linspace(0, length_x, cells + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    length_y = np.pi
    delta_y = np.repeat(length_y / cells, cells)
    edges_y = np.linspace(0, length_y, cells + 1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    quadrature = ants.angular_xy(angles)
    angle_x = quadrature.angle_x
    angle_y = quadrature.angle_y

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.25]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.ones((1,)),
    )

    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_02(
            centers_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )[1:]
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t[1:]
        )
        sources = SourceData(
            initial_flux=initial_flux,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
    # Crank Nicolson
    elif temporal == 2:
        initial_flux_x = mms.solution_td_02(
            edges_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        initial_flux_y = mms.solution_td_02(
            centers_x, edges_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t[1:]
        )
        sources = SourceData(
            initial_flux_x=initial_flux_x,
            initial_flux_y=initial_flux_y,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_02(
            centers_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        external = ants.external2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t
        )[1:]
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, edges_t[1:]
        )
        sources = SourceData(
            initial_flux=initial_flux,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
    # TR-BDF2
    elif temporal == 4:
        initial_flux_x = mms.solution_td_02(
            edges_x, centers_y, angle_x, angle_y, np.array([0.0])
        )[0]
        initial_flux_y = mms.solution_td_02(
            centers_x, edges_y, angle_x, angle_y, np.array([0.0])
        )[0]
        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, gamma_steps
        )
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(
            centers_x, centers_y, angle_x, angle_y, gamma_steps[1:]
        )
        sources = SourceData(
            initial_flux_x=initial_flux_x,
            initial_flux_y=initial_flux_y,
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=delta_x,
        delta_y=delta_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=edges_t.shape[0] - 1, dt=dt, time_disc=temporal)

    return mat_data, sources, geometry, quadrature, solver, time_data
