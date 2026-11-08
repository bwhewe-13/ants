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
from ants.datatypes import Geometry, GeometryData, MaterialData, SourceData


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
