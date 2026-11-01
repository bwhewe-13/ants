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
    SpatialDiscretization,
    TimeDependentData,
)
from ants.utils import manufactured_2d as mms
from ants.utils import mesh2d

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


def manufactured_ss_01_polygon(mesh, angles):
    # Same as manufactured_ss_01 on a polygon mesh of the unit square
    quadrature = ants.angular_xy(angles)

    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )

    # Exact solution on the boundary faces
    centers = mesh.face_center[mesh.boundary_faces]
    boundary_x = mms.solution_ss_01_points(centers[:, 0], quadrature.angle_x)[
        None, :, :, None
    ]

    sources = SourceData(
        external=0.5 * np.ones((mesh.cells, 1, 1, 1)),
        boundary_x=boundary_x,
        boundary_y=np.zeros((1, 1, 1, 1)),
    )

    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
    )
    solver = SolverData()

    return mat_data, sources, geometry, quadrature, solver


def slab_to_polygon(sources, geometry):
    # Rectangular cells of a SLAB2D problem as a POLYGON2D problem, with
    # cell ii * cells_y + jj = (ii, jj). Time dependent sources have a
    # leading time axis.
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    cells = centers_x.size * centers_y.size
    mesh = mesh2d.rectangle_mesh(edges_x, edges_y)

    timed = sources.boundary_x.ndim == 5
    boundary_x = sources.boundary_x if timed else sources.boundary_x[None]
    boundary_y = sources.boundary_y if timed else sources.boundary_y[None]
    external = sources.external if timed else sources.external[None]

    # Boundary faces from the x and y boundaries
    sides = mesh.face_side[mesh.boundary_faces]
    centers = mesh.face_center[mesh.boundary_faces]
    boundary = np.zeros((boundary_x.shape[0], 1, sides.size) + boundary_x.shape[3:])
    for bb, (side, (x, y)) in enumerate(zip(sides, centers)):
        if side < 2:
            jj = 0 if boundary_x.shape[2] == 1 else np.argmin(np.fabs(centers_y - y))
            boundary[:, 0, bb] = boundary_x[:, side, jj]
        else:
            ii = 0 if boundary_y.shape[2] == 1 else np.argmin(np.fabs(centers_x - x))
            boundary[:, 0, bb] = boundary_y[:, side - 2, ii]

    if external.shape[1] > 1:
        external = external.reshape((external.shape[0], cells, 1) + external.shape[3:])

    initial_flux = sources.initial_flux
    if initial_flux is not None:
        initial_flux = initial_flux.reshape((cells, 1) + initial_flux.shape[2:])

    polygon_sources = SourceData(
        external=external if timed else external[0],
        boundary_x=boundary if timed else boundary[0],
        boundary_y=np.zeros((1, 1, 1, 1, 1) if timed else (1, 1, 1, 1)),
        initial_flux=initial_flux,
    )
    polygon_geometry = GeometryData(
        medium_map=geometry.medium_map.reshape(cells, 1),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=list(geometry.bc_x) + list(geometry.bc_y),
    )
    return polygon_sources, polygon_geometry
