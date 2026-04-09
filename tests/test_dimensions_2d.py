########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Dimensionality tests for two dimensional slabs
#
########################################################################

import os

import numpy as np
import pytest

import ants
from ants.datatypes import Geometry, GeometryData, MaterialData, SolverData, SourceData
from ants.fixed2d import fixed_source

# Path for reference solutions
PATH = os.path.join("data", "weight_matrix_2d")


def cylinder_01():
    # Problem parameters
    cells_x = cells_y = 50
    angles = 4

    # Spatial Layout
    radius = 4.279960
    delta_x = np.repeat(radius * 2 / cells_x, cells_x)
    delta_y = np.repeat(radius * 2 / cells_y, cells_y)

    # Boundary conditions
    bc_x = [0, 0]
    bc_y = [0, 0]

    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84 * 0.0816]], [[0.0]]])

    # Update cross sections for cylinder
    # coordinates = [(radius, radius), [radius]]
    # weight_matrix = ants.weight_cylinder2d(coordinates, edges_x, \
    #                                          edges_y, N=250_000)
    weight_matrix = np.load(os.path.join(PATH, "cylinder_two_material.npy"))
    medium_map, xs_total, xs_scatter, xs_fission = ants.weight_spatial2d(
        weight_matrix, xs_total, xs_scatter, xs_fission
    )

    mat_data = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
    )

    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=delta_x,
        delta_y=delta_y,
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=Geometry.SLAB2D,
    )
    solver = SolverData()

    return mat_data, geometry, quadrature, solver


@pytest.mark.smoke
@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_x():
    mat_data, geometry, quadrature, solver = cylinder_01()

    cells_x, cells_y = geometry.medium_map.shape
    groups = mat_data.total.shape[1]
    n_angles = quadrature.angle_x.size

    # Set boundary_y and external sources
    external = np.zeros((cells_x, cells_y, 1, 1))
    boundary_y = np.zeros((2, 1, 1, 1))

    # Different boundary_x dimensions
    bounds = [
        np.zeros((2, 1, 1, 1)),
        np.zeros((2, cells_y, 1, 1)),
        np.zeros((2, cells_y, 1, groups)),
        np.zeros((2, cells_y, n_angles, groups)),
    ]
    flux = []
    for loc in [0, 1]:
        for boundary_x in bounds:
            boundary_x[loc] = 1.0
            sources = SourceData(
                external=external,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
            result = fixed_source(mat_data, sources, geometry, quadrature, solver)
            boundary_x *= 0.0
            if loc == 1:
                flux.append(result[::-1, :, 0].flatten())
            else:
                flux.append(result[:, :, 0].flatten())
    for ii in range(len(flux) - 1):
        assert np.isclose(flux[ii], flux[ii + 1], atol=1e-10).all()


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_y():
    mat_data, geometry, quadrature, solver = cylinder_01()

    cells_x, cells_y = geometry.medium_map.shape
    groups = mat_data.total.shape[1]
    n_angles = quadrature.angle_x.size

    # Set boundary_x and external sources
    external = np.zeros((cells_x, cells_y, 1, 1))
    boundary_x = np.zeros((2, 1, 1, 1))

    # Different boundary_y dimensions
    bounds = [
        np.zeros((2, 1, 1, 1)),
        np.zeros((2, cells_x, 1, 1)),
        np.zeros((2, cells_x, 1, groups)),
        np.zeros((2, cells_x, n_angles, groups)),
    ]
    flux = []
    for loc in [0, 1]:
        for boundary_y in bounds:
            boundary_y[loc] = 1
            sources = SourceData(
                external=external,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
            result = fixed_source(mat_data, sources, geometry, quadrature, solver)
            boundary_y *= 0.0
            if loc == 1:
                flux.append(result[:, ::-1, 0].flatten())
            else:
                flux.append(result[:, :, 0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii + 1], atol=1e-10))


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_external_unit():
    mat_data, geometry, quadrature, solver = cylinder_01()

    cells_x, cells_y = geometry.medium_map.shape
    groups = mat_data.total.shape[1]
    n_angles = quadrature.angle_x.size

    # Set boundary_x and boundary_y
    boundary_x = np.zeros((2, 1, 1, 1))
    boundary_y = np.zeros((2, 1, 1, 1))

    # Different external dimensions
    externals = [
        np.ones((cells_x, cells_y, 1, 1)),
        np.ones((cells_x, cells_y, 1, groups)),
        np.ones((cells_x, cells_y, n_angles, groups)),
    ]
    flux = []
    for external in externals:
        sources = SourceData(
            external=external,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
        result = fixed_source(mat_data, sources, geometry, quadrature, solver)
        flux.append(result[:, :, 0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii + 1], atol=1e-10))
