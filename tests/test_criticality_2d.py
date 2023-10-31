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


import pytest
import numpy as np

import ants
from ants.critical2d import power_iteration

PATH = "data/weight_matrix_2d/"

@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("finite", "spatial"), [("x", "step"), \
                         ("x", "diamond"), ("y", "step"), ("y", "diamond")])
def test_one_group_infinite(finite, spatial):
    # Material Parameters
    if finite == "x":
        cells_x = 100; length_x = 1.853722 * 2
        cells_y = 10; length_y = 1000 * 2 
    else:
        cells_y = 100; length_y = 1.853722 * 2
        cells_x = 10; length_x = 1000 * 2
    angles = 12
    groups = 1
    # Spatial discretization
    ss = 1 if spatial == "step" else 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[3.24*0.0816]]])
    # Collect problem dictionary
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 1, "geometry": 1, "spatial": ss, \
            "bc_x": bc_x, "bc_y": bc_y}
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("finite", "spatial"), [("x", "step"), \
                         ("x", "diamond"), ("y", "step"), ("y", "diamond")])
def test_two_group_infinite(finite, spatial):
    # Material Parameters
    if finite == "x":
        cells_x = 100; length_x = 1.795602 * 2
        cells_y = 10; length_y = 2000
    else:
        cells_y = 100; length_y = 1.795602 * 2
        cells_x = 10; length_x = 2000
    groups = 2
    angles = 10        
    # Spatial discretization
    ss = 1 if spatial == "step" else 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    fission = np.array(chi @ (nu * sigmaf))
    total = np.array([0.3360,0.2208])
    scatter = np.array([[0.23616, 0.0],[0.0432, 0.0792]])
    xs_total = np.array([total])
    xs_scatter = np.array([scatter.T])
    xs_fission = np.array([fission])
    # Collect problem dictionary
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 1, "geometry": 1, "spatial": ss, \
            "bc_x": bc_x, "bc_y": bc_y}
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 5e-3, "k-effective: " + str(keff)


@pytest.mark.slab2d
@pytest.mark.power_iteration
def test_two_group_twigl():
    # Material Properties
    cells_x = 160; length_x = 160
    cells_y = 160; length_y = 160
    groups = 2
    angles = 4
    # Spatial Dimensions
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    # Medium Map
    medium_map = 2 * np.ones((cells_x, cells_y), dtype=np.int32)
    coords_mat0 = [[(24, 24), 32, 32], [(24, 104), 32, 32], 
                   [(104, 24), 32, 32], [(104, 104), 32, 32]]
    coords_mat1 = [[(24, 56), 32, 48], [(56, 24), 48, 32], 
                   [(104, 56), 32, 48], [(56, 104), 48, 32]]
    medium_map = ants.spatial2d(medium_map, 0, coords_mat0, edges_x, edges_y)
    medium_map = ants.spatial2d(medium_map, 1, coords_mat1, edges_x, edges_y)
    # Boundary conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Total Cross Section
    mat1_total = np.array([0.238095,0.83333])
    mat2_total = np.array([0.238095,0.83333])
    mat3_total = np.array([0.25641,0.666667])
    xs_total = np.array([mat1_total, mat2_total, mat3_total])
    # Scatter Cross Section
    mat1_scatter = np.array([[0.218095, 0.01], [0.0, 0.68333]])
    mat2_scatter = np.array([[0.218095, 0.01], [0.0, 0.68333]])
    mat3_scatter = np.array([[0.23841, 0.01], [0.0,0.616667]])
    xs_scatter = np.array([mat1_scatter, mat2_scatter, mat3_scatter])
    # Fission Cross Section
    chi = np.array([1,0]) # All fast
    mat1_nufission = np.array([0.007,0.2])
    mat1_fission = chi[:,None] @ mat1_nufission[None,:]
    mat2_nufission = np.array([0.007,0.2])
    mat2_fission = chi[:,None] @ mat2_nufission[None,:]
    mat3_nufission = np.array([0.003,0.06])
    mat3_fission = chi[:,None] @ mat3_nufission[None,:]
    xs_fission = np.array([mat1_fission.T, mat2_fission.T, mat3_fission.T])
    # Collect problem dictionary
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 3, "geometry": 1, "spatial": 2, \
            "qdim": 2, "bc_x": bc_x, "bcdim_x": 1, "bc_y": bc_y, "bcdim_y": 1}
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    reference_keff = 0.917507
    assert abs(keff - reference_keff) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
def test_cylinder_two_material():
    # Material Parameters
    cells_x = cells_y = 50
    angles = 6
    groups = 1
    # Spatial layout
    radius = 4.279960
    coordinates = [(radius, radius), [radius]]
    # Inscribed inside circle
    length_x = length_y = 2 * radius
    # Spatial Dimensions
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84*0.0816]], [[0.0]]])
    # Update cross sections for cylinder
    # weight_matrix = ants.weight_cylinder2d(coordinates, edges_x, edges_y, N=250_000)
    # np.save(PATH + "cylinder_two_material", weight_matrix)
    weight_matrix = np.load(PATH + "cylinder_two_material.npy")
    medium_map, xs_total, xs_scatter, xs_fission \
        = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    # Collect problem dictionary
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": len(xs_total), "geometry": 1, \
            "spatial": 2, "qdim": 2, "bc_x": bc_x, "bcdim_x": 1, \
            "bc_y": bc_y, "bcdim_y": 1}
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info) 
    flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("layer"), ["small", "large"])
def test_cylinder_three_material(layer):
    # Radii
    radius01 = 15.396916 if layer == "small" else 14.606658
    radius02 = radius01 + 1.830563 if layer == "small" else radius01 + 18.30563
    coordinates = [(radius02, radius02), [radius01, radius02]]
    # Material Parameters
    cells_x = cells_y = 100
    angles = 4
    groups = 1
    # Inscribed inside circle
    length_x = length_y = 2 * radius02
    # Spatial Dimensions
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    # Boundary Conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    xs_total = np.array([[0.54628], [0.54628], [0.0]])
    xs_scatter = np.array([[[0.464338]], [[0.491652]], [[0.0]]])
    xs_fission = np.array([[[1.70*0.054628]], [[0.0]], [[0.0]]])
    # Update cross sections for cylinder
    # weight_matrix = ants.weight_cylinder2d(coordinates, edges_x, edges_y, N=500_000)
    # np.save(PATH + f"cylinder_three_material_{layer}", weight_matrix)
    weight_matrix = np.load(PATH + f"cylinder_three_material_{layer}.npy")
    medium_map, xs_total, xs_scatter, xs_fission \
        = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    # Collect problem dictionary
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": len(xs_total), "geometry": 1, \
            "spatial": 2, "qdim": 2, "bc_x": bc_x, "bcdim_x": 1, \
            "bc_y": bc_y, "bcdim_y": 1}
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info)    
    flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)
