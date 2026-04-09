########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Criticality tests for two dimensional slabs
#
########################################################################


import os

import numpy as np
import pytest

import ants
from ants.critical2d import k_criticality
from ants.datatypes import GeometryData, MaterialData, SolverData
from tests import criticality_benchmarks as benchmarks

PATH = "data/weight_matrix_2d/"


@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_pua_1_0_infinite_y(bc_x):
    cells_x = 100
    length_x = 1.853722 * 2 if bc_x == [0, 0] else 1.853722
    cells_y = 10
    length_y = 2000
    bc_y = [0, 0]
    angles = 12
    solver = SolverData()
    quadrature = ants.angular_xy(angles=angles, bc_x=bc_x)
    mat_data, _ = benchmarks.PUa_1_0(cells_x, bc_x)
    geometry = GeometryData(
        medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_y"), [[0, 0], [0, 1], [1, 0]])
def test_pua_1_0_infinite_x(bc_y):
    cells_x = 10
    length_x = 2000
    bc_x = [0, 0]
    cells_y = 100
    length_y = 1.853722 * 2 if bc_y == [0, 0] else 1.853722
    angles = 12
    solver = SolverData()
    quadrature = ants.angular_xy(angles=angles, bc_x=bc_x)
    mat_data, _ = benchmarks.PUa_1_0(cells_x, bc_x)
    geometry = GeometryData(
        medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_pu_2_0_infinite_y(bc_x):
    # Material Parameters
    cells_x = 100
    length_x = 1.795602 * 2 if bc_x == [0, 0] else 1.795602
    cells_y = 10
    length_y = 2000
    bc_y = [0, 0]
    angles = 10
    solver = SolverData()
    quadrature = ants.angular_xy(angles=angles, bc_x=bc_x)
    mat_data, _ = benchmarks.PU_2_0(cells_x, bc_x, 1)
    geometry = GeometryData(
        medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 5e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_y"), [[0, 0], [0, 1], [1, 0]])
def test_pu_2_0_infinite_x(bc_y):
    cells_x = 10
    length_x = 2000
    cells_y = 100
    length_y = 1.795602 * 2 if bc_y == [0, 0] else 1.795602
    bc_x = [0, 0]
    angles = 10
    solver = SolverData()
    quadrature = ants.angular_xy(angles=angles, bc_x=bc_x)
    mat_data, _ = benchmarks.PU_2_0(cells_x, bc_x, 1)
    geometry = GeometryData(
        medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 5e-3, "k-effective: " + str(keff)


@pytest.mark.slab2d
@pytest.mark.power_iteration
def test_two_group_twigl():
    # Material Properties
    cells_x = 80
    cells_y = 80
    angles = 4

    # Spatial Dimensions
    length_x = 80.0
    edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 12)

    length_y = 80.0
    edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 12)

    # Medium Map
    medium_map = 2 * np.ones((cells_x, cells_y), dtype=np.int32)
    coords_mat0 = [[(24, 24), 32, 32]]
    coords_mat1 = [[(24, 0), 32, 24], [(0, 24), 24, 32]]
    medium_map = ants.spatial2d(medium_map, 0, coords_mat0, edges_x, edges_y)
    medium_map = ants.spatial2d(medium_map, 1, coords_mat1, edges_x, edges_y)

    # Boundary conditions
    bc_x = [1, 0]
    bc_y = [1, 0]

    # Total Cross Section
    mat1_total = np.array([0.238095, 0.83333])
    mat2_total = np.array([0.238095, 0.83333])
    mat3_total = np.array([0.25641, 0.666667])
    xs_total = np.array([mat1_total, mat2_total, mat3_total])

    # Scatter Cross Section
    mat1_scatter = np.array([[0.218095, 0.01], [0.0, 0.68333]])
    mat2_scatter = np.array([[0.218095, 0.01], [0.0, 0.68333]])
    mat3_scatter = np.array([[0.23841, 0.01], [0.0, 0.616667]])
    xs_scatter = np.array([mat1_scatter, mat2_scatter, mat3_scatter])

    # Fission Cross Section
    chi = np.array([1, 0])  # All fast
    mat1_nufission = np.array([0.007, 0.2])
    mat1_fission = chi[:, None] @ mat1_nufission[None, :]
    mat2_nufission = np.array([0.007, 0.2])
    mat2_fission = chi[:, None] @ mat2_nufission[None, :]
    mat3_nufission = np.array([0.003, 0.06])
    mat3_fission = chi[:, None] @ mat3_nufission[None, :]
    xs_fission = np.array([mat1_fission.T, mat2_fission.T, mat3_fission.T])

    mat_data = MaterialData(total=xs_total, scatter=xs_scatter, fission=xs_fission)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    solver = SolverData()

    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    reference_keff = 0.917507
    assert abs(keff - reference_keff) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
def test_cylinder_two_material():
    # Material Parameters
    cells_x = 50
    cells_y = 50
    angles = 6

    # Spatial layout
    radius = 4.279960

    # Inscribed inside circle
    length_x = length_y = 2 * radius

    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84 * 0.0816]], [[0.0]]])

    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]

    # coords = [[(radius, radius), (0.0, radius)]]
    # edges_x = np.linspace(0, length_x, cells_x + 1)
    # edges_y = np.linspace(0, length_y, cells_y + 1)
    # Update cross sections for cylinder
    # weight_matrix = ants.weight_matrix2d(edges_x, edges_y, materials=2, \
    #                     N_particles=cells_x * 50_000, circles=coords, \
    #                     circle_index=[0])
    # np.save(PATH + "cylinder_two_material", weight_matrix)

    weight_matrix = np.load(os.path.join(PATH, "cylinder_two_material.npy"))
    weighted = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    medium_map, xs_total, xs_scatter, xs_fission = weighted

    mat_data = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
    )
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    solver = SolverData()

    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.cylinder2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(
    ("bc_x", "bc_y"),
    [([0, 0], [0, 1]), ([0, 0], [1, 0]), ([0, 1], [0, 0]), ([1, 0], [0, 0])],
)
def test_cylinder_two_material_half(bc_x, bc_y):
    # Material Parameters
    cells_x = 50 if bc_x == [0, 0] else 25
    cells_y = 50 if bc_y == [0, 0] else 25
    angles = 6

    # Spatial layout
    radius = 4.279960

    # Inscribed inside circle
    length_x = 2 * radius if bc_x == [0, 0] else radius
    length_y = 2 * radius if bc_y == [0, 0] else radius

    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84 * 0.0816]], [[0.0]]])

    # Update cross sections for cylinder
    # coords = [[(radius, radius), (0.0, radius)]]
    # edges_x = np.linspace(0, length_x, cells_x + 1)
    # edges_y = np.linspace(0, length_y, cells_y + 1)
    # weight_matrix = ants.weight_matrix2d(edges_x, edges_y, materials=2, \
    #                     N_particles=cells_x * 50_000, circles=coords, \
    #                     circle_index=[0])
    # np.save(PATH + "cylinder_two_material", weight_matrix)

    weight_matrix = np.load(os.path.join(PATH, "cylinder_two_material.npy"))
    if bc_x == [0, 1]:
        weight_matrix = weight_matrix[:25, :, :].copy()
    elif bc_x == [1, 0]:
        weight_matrix = weight_matrix[25:, :, :].copy()
    elif bc_y == [0, 1]:
        weight_matrix = weight_matrix[:, :25, :].copy()
    elif bc_y == [1, 0]:
        weight_matrix = weight_matrix[:, 25:, :].copy()

    weighted = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    medium_map, xs_total, xs_scatter, xs_fission = weighted

    mat_data = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
    )
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    solver = SolverData()

    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.cylinder2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(
    ("bc_x", "bc_y"),
    [([0, 1], [0, 1]), ([1, 0], [0, 1]), ([0, 1], [1, 0]), ([1, 0], [1, 0])],
)
def test_cylinder_two_material_quarter(bc_x, bc_y):
    # Material Parameters
    cells_x = 25
    cells_y = 25
    angles = 6

    # Spatial layout
    radius = 4.279960

    # Inscribed inside circle
    length_x = radius
    length_y = radius

    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84 * 0.0816]], [[0.0]]])

    # Update cross sections for cylinder
    # coords = [[(radius, radius), (0.0, radius)]]
    # edges_x = np.linspace(0, length_x, cells_x + 1)
    # edges_y = np.linspace(0, length_y, cells_y + 1)
    # weight_matrix = ants.weight_matrix2d(edges_x, edges_y, materials=2, \
    #                     N_particles=cells_x * 50_000, circles=coords, \
    #                     circle_index=[0])
    # np.save(PATH + "cylinder_two_material", weight_matrix)

    weight_matrix = np.load(os.path.join(PATH, "cylinder_two_material.npy"))
    if bc_x == [0, 1]:
        weight_matrix = weight_matrix[:25, :, :].copy()
    elif bc_x == [1, 0]:
        weight_matrix = weight_matrix[25:, :, :].copy()
    if bc_y == [0, 1]:
        weight_matrix = weight_matrix[:, :25, :].copy()
    if bc_y == [1, 0]:
        weight_matrix = weight_matrix[:, 25:, :].copy()

    weighted = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    medium_map, xs_total, xs_scatter, xs_fission = weighted

    mat_data = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
    )
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    solver = SolverData()

    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("layer"), ["small", "large"])
def test_cylinder_three_material(layer):
    # Radii
    radius01 = 15.396916 if layer == "small" else 14.606658
    radius02 = radius01 + 1.830563 if layer == "small" else radius01 + 18.30563

    # Material Parameters
    cells_x = cells_y = 100
    angles = 4

    # Inscribed inside circle
    length_x = length_y = 2 * radius02

    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    xs_total = np.array([[0.54628], [0.54628], [0.0]])
    xs_scatter = np.array([[[0.464338]], [[0.491652]], [[0.0]]])
    xs_fission = np.array([[[1.70 * 0.054628]], [[0.0]], [[0.0]]])

    # Update cross sections for cylinder
    # coords = [
    #     [(radius02, radius02), (0.0, radius01)],
    #     [(radius02, radius02), (radius01, radius02)],
    # ]
    # edges_x = np.linspace(0, length_x, cells_x + 1)
    # edges_y = np.linspace(0, length_y, cells_y + 1)
    # weight_matrix = ants.weight_matrix2d(edges_x, edges_y, materials=3, \
    #                         N_particles=cells_x * 50_000, circles=coords, \
    #                         circle_index=[0,1])
    # np.save(os.path.join(PATH, f"cylinder_three_material_{layer}"), weight_matrix)
    weight_matrix = np.load(os.path.join(PATH, f"cylinder_three_material_{layer}.npy"))

    weighted = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    medium_map, xs_total, xs_scatter, xs_fission = weighted

    mat_data = MaterialData(
        total=xs_total,
        scatter=xs_scatter,
        fission=xs_fission,
    )
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    solver = SolverData()

    _, keff = k_criticality(mat_data, geometry, quadrature, solver)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)
