########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Three-dimensional problems shared by the 3D tests
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
from ants.utils import hybrid as hytools


def zero_boundaries():
    # Vacuum boundary_x, boundary_y, and boundary_z
    return [np.zeros((2, 1, 1, 1, 1)) for _ in range(3)]


def two_materials(groups=1):
    # Scattering material (0) and a thin absorber (1)
    total = np.array([[1.0, 1.5], [0.3, 0.4]])[:, :groups]
    scatter = np.array([[[0.6, 0.0], [0.2, 1.0]], [[0.1, 0.0], [0.05, 0.2]]])
    return MaterialData(
        total=total,
        scatter=scatter[:, :groups, :groups],
        fission=np.zeros((2, groups, groups)),
    )


def coupled_materials(groups):
    # Within-group, down, and up scattering in both materials
    scatter = np.zeros((2, groups, groups))
    for gg in range(groups):
        scatter[:, gg, gg] = [0.4, 0.1]
        if gg + 1 < groups:
            scatter[:, gg + 1, gg] = [0.2, 0.05]
        if gg > 0:
            scatter[:, gg - 1, gg] = [0.05, 0.01]
    total = np.array([np.full(groups, 1.0), np.full(groups, 0.3)])
    return MaterialData(
        total=total, scatter=scatter, fission=np.zeros((2, groups, groups))
    )


def axis_problem(spatial, groups=2):
    # Problem that changes along every axis, with a different number of
    # cells, cell widths, and boundary conditions on each
    rng = np.random.default_rng(13)
    cells = (6, 5, 4)
    medium_map = rng.integers(0, 2, size=cells).astype(np.int32)
    external = np.zeros(cells + (1, groups))
    external[..., 0, 0] = rng.random(cells) * (medium_map == 0)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=rng.uniform(0.2, 0.6, cells[0]),
        delta_y=rng.uniform(0.2, 0.6, cells[1]),
        delta_z=rng.uniform(0.2, 0.6, cells[2]),
        bc_x=[1, 0],
        bc_y=[0, 0],
        bc_z=[0, 1],
        geometry=Geometry.SLAB3D,
        space_disc=spatial,
    )
    # Incoming flux on the lower y face only
    boundary_y = np.array([1.0, 0.0]).reshape(2, 1, 1, 1, 1)
    sources = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1, 1)),
        boundary_y=boundary_y,
        boundary_z=np.zeros((2, 1, 1, 1, 1)),
    )
    quadrature = ants.angular_xyz(4, bc_x=[1, 0], bc_z=[0, 1])
    return geometry, sources, quadrature


def permute_axes(geometry, sources, quadrature):
    # Relabel the axes so the new x is the old z, the new y is the old x,
    # and the new z is the old y. The boundary sources must be uniform on
    # each face. The flux of the new problem is flux.transpose(2, 0, 1, ...).
    geometry_p = GeometryData(
        medium_map=np.ascontiguousarray(geometry.medium_map.transpose(2, 0, 1)),
        delta_x=geometry.delta_z,
        delta_y=geometry.delta_x,
        delta_z=geometry.delta_y,
        bc_x=geometry.bc_z,
        bc_y=geometry.bc_x,
        bc_z=geometry.bc_y,
        geometry=Geometry.SLAB3D,
        space_disc=geometry.space_disc,
    )
    sources_p = SourceData(
        external=np.ascontiguousarray(sources.external.transpose(2, 0, 1, 3, 4)),
        boundary_x=sources.boundary_z,
        boundary_y=sources.boundary_x,
        boundary_z=sources.boundary_y,
    )
    quadrature_p = ants.QuadratureData(
        angle_x=quadrature.angle_z,
        angle_y=quadrature.angle_x,
        angle_z=quadrature.angle_y,
        angle_w=quadrature.angle_w,
    )
    return geometry_p, sources_p, quadrature_p


def time_axis_problem(spatial, time_disc, materials):
    # axis_problem with the sources of a time-dependent problem, constant in
    # time with a zero initial flux
    groups = materials.total.shape[1]
    geometry, sources, quadrature = axis_problem(spatial, groups)
    cells = geometry.medium_map.shape
    shape = (quadrature.angle_w.size, groups)
    kwargs = dict(
        external=sources.external[None],
        boundary_x=sources.boundary_x[None],
        boundary_y=sources.boundary_y[None],
        boundary_z=sources.boundary_z[None],
    )
    if time_disc in (1, 3):
        kwargs["initial_flux"] = np.zeros(cells + shape)
    else:
        kwargs["initial_flux_x"] = np.zeros((cells[0] + 1,) + cells[1:] + shape)
        kwargs["initial_flux_y"] = np.zeros((cells[0], cells[1] + 1, cells[2]) + shape)
        kwargs["initial_flux_z"] = np.zeros(cells[:2] + (cells[2] + 1,) + shape)
    return sources, SourceData(**kwargs), geometry, quadrature


def coupled_time_problem(time_disc, groups=4, ratios=None):
    # time_axis_problem with coupled groups and a velocity for each
    materials = coupled_materials(groups)
    materials.velocity = np.linspace(4.0, 1.0, groups)
    if ratios is not None:
        materials.scatter = np.stack(
            [ratio * materials.scatter for ratio in ratios], -1
        )
    _, sources, geometry, quadrature = time_axis_problem(2, time_disc, materials)
    return materials, sources, geometry, quadrature


def collided_data(materials, groups_c):
    # Collided materials and the hybrid indexing on a uniform energy grid
    groups_u = materials.total.shape[1]
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(None, groups_u, groups_c)
    hybrid_data = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    if groups_u == groups_c:
        return materials, hybrid_data, edges_g
    xs_total, xs_scatter, xs_fission = hytools.coarsen_materials(
        materials.total,
        materials.scatter,
        materials.fission,
        edges_g[edges_gidx_u],
        edges_gidx_c,
    )
    materials_c = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
        velocity=hytools.coarsen_velocity(materials.velocity, edges_gidx_c),
    )
    return materials_c, hybrid_data, edges_g


def solution_td_01(x, y, z, angle_x, edges_t):
    # Linear in space so diamond difference has no spatial error, shape
    # (T, I, J, K, N, 1)
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")
    spatial = 1 + mesh_x + 0.5 * mesh_y + 0.25 * mesh_z
    flux = spatial[None] * (1 + np.sin(edges_t))[:, None, None, None]
    return np.repeat(flux[..., None, None], angle_x.size, axis=4)


def external_td_01(x, y, z, quadrature, edges_t, xs_total, xs_scatter, velocity):
    # Source of solution_td_01, shape (T, I, J, K, N, 1)
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")
    spatial = 1 + mesh_x + 0.5 * mesh_y + 0.25 * mesh_z
    gradient = quadrature.angle_x + 0.5 * quadrature.angle_y + 0.25 * quadrature.angle_z
    removal = xs_total - xs_scatter * np.sum(quadrature.angle_w)
    external = np.zeros((edges_t.size,) + spatial.shape + (gradient.size, 1))
    for tt, time in enumerate(edges_t):
        amplitude = 1 + np.sin(time)
        external[tt, ..., 0] = (
            spatial[..., None] * (np.cos(time) / velocity + removal * amplitude)
            + gradient * amplitude
        )
    return external


def boundary_td_01(edges, centers, quadrature, edges_t):
    # Incoming angles of solution_td_01 on each face, boundary_x, boundary_y,
    # and boundary_z with shape (T, 2, face, face, N/2, 1)
    cosines = (quadrature.angle_x, quadrature.angle_y, quadrature.angle_z)
    boundaries = []
    for axis, angle in enumerate(cosines):
        sides = []
        for side, incoming in enumerate((angle > 0, angle < 0)):
            points = [centers, centers, centers]
            points[axis] = edges[[0, -1]][[side]]
            flux = solution_td_01(*points, angle, edges_t)
            flux = np.squeeze(flux, axis=axis + 1)
            sides.append(flux[..., incoming, :])
        boundaries.append(np.stack(sides, axis=1))
    return boundaries


def manufactured_td_01(cells, angles, edges_t, dt, temporal=1):
    # Time-dependent manufactured solution with isotropic scattering
    length = 1.0
    delta = np.repeat(length / cells, cells)
    edges = np.linspace(0, length, cells + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Angular
    quadrature = ants.angular_xyz(angles)
    angle_x = quadrature.angle_x

    xs_total = 1.0
    xs_scatter = 0.5
    velocity = 2.0
    mat_data = MaterialData(
        total=np.array([[xs_total]]),
        scatter=np.array([[[xs_scatter]]]),
        fission=np.array([[[0.0]]]),
        velocity=np.array([velocity]),
    )

    # External source at edges_t for CN, at the gamma steps for TR-BDF2
    if temporal == 4:
        times = ants.gamma_time_steps(edges_t)
    else:
        times = edges_t
    external = external_td_01(
        centers, centers, centers, quadrature, times, xs_total, xs_scatter, velocity
    )
    boundary_x, boundary_y, boundary_z = boundary_td_01(
        edges, centers, quadrature, times[1:]
    )

    # Backward Euler and BDF2
    if temporal in (1, 3):
        initial_flux = solution_td_01(
            centers, centers, centers, angle_x, np.array([0.0])
        )[0]
        sources = SourceData(
            initial_flux=initial_flux,
            external=external[1:],
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            boundary_z=boundary_z,
        )
    # Crank Nicolson and TR-BDF2
    else:
        zero = np.array([0.0])
        sources = SourceData(
            initial_flux_x=solution_td_01(edges, centers, centers, angle_x, zero)[0],
            initial_flux_y=solution_td_01(centers, edges, centers, angle_x, zero)[0],
            initial_flux_z=solution_td_01(centers, centers, edges, angle_x, zero)[0],
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            boundary_z=boundary_z,
        )

    geometry = GeometryData(
        medium_map=np.zeros((cells, cells, cells), dtype=np.int32),
        delta_x=delta,
        delta_y=delta,
        delta_z=delta,
        bc_x=[0, 0],
        bc_y=[0, 0],
        bc_z=[0, 0],
        geometry=Geometry.SLAB3D,
        space_disc=SpatialDiscretization.DIAMOND,
    )

    solver = SolverData(tol_angular=1e-13, tol_energy=1e-13, max_iter_angular=500)
    time_data = TimeDependentData(steps=edges_t.size - 1, dt=dt, time_disc=temporal)

    return mat_data, sources, geometry, quadrature, solver, time_data
