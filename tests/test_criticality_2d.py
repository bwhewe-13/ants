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
from ants.critical2d import power_iteration as power
from ants.utils import dimensions


@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("finite", "spatial"), [("x", "step"), \
                         ("x", "diamond"), ("y", "step"), ("y", "diamond")])
def test_one_group_infinite(finite, spatial):
    if finite == "x":
        cells_x = 100; length_x = 1.853722 * 2
        cells_y = 10; length_y = 1000 * 2 
    else:
        cells_y = 100; length_y = 1.853722 * 2
        cells_x = 10; length_x = 1000 * 2
    # Spatial discretization
    ss = 1 if spatial == "step" else 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    # Cross Sections
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[3.24*0.0816]]])
    # Material Parameters
    groups = 1
    angles = 12
    bc = [0, 0]
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 1, "geometry": 1, "spatial": ss, \
            "qdim": 2, "bc_x": bc, "bcdim_x": 1, "bc_y": bc, "bcdim_y": 1}
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("finite", "spatial"), [("x", "step"), \
                         ("x", "diamond"), ("y", "step"), ("y", "diamond")])
def test_two_group_infinite(finite, spatial):
    if finite == "x":
        cells_x = 100; length_x = 1.795602 * 2
        cells_y = 10; length_y = 2000
    else:
        cells_y = 100; length_y = 1.795602 * 2
        cells_x = 10; length_x = 2000
    # Spatial discretization
    ss = 1 if spatial == "step" else 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
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
    # Material Parameters
    groups = 2
    angles = 10
    bc = [0, 0]
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 1, "geometry": 1, "spatial": ss, \
            "qdim": 2, "bc_x": bc, "bcdim_x": 1, "bc_y": bc, "bcdim_y": 1}    
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 5e-3, "k-effective: " + str(keff)


@pytest.mark.slab2d
@pytest.mark.power_iteration
def test_two_group_twigl():
    cells_x = 80; length_x = 80
    cells_y = 80; length_y = 80
    medium_map = np.ones((cells_x, cells_y), dtype=np.int32) * 2
    xx = int(cells_x / length_x)
    yy = int(cells_y / length_y)
    medium_map[24*yy:56*yy, 0:24*xx] = 1
    medium_map[0:24*yy, 24*xx:56*xx] = 1
    medium_map[24*yy:56*yy, 24*xx:56*xx] = 0
    # Orient the medium map correctly
    medium_map = np.block([[np.flip(medium_map, axis=(1,0)), \
                            np.flip(medium_map, axis=0)], \
                    [np.flip(medium_map, axis=1), medium_map]])
    cells_x *= 2; length_x = 80 * 2
    cells_y *= 2; length_y = 80 * 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    delta_y = np.repeat(length_y / cells_y, cells_y)
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
    # Material Properties
    groups = 2
    angles = 4
    bc = [0, 0]
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": 3, "geometry": 1, "spatial": 2, \
            "qdim": 2, "bc_x": bc, "bcdim_x": 1, "bc_y": bc, "bcdim_y": 1}
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    reference_keff = 0.917507
    assert abs(keff - reference_keff) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
def test_cylinder_two_material():
    radii = [(0.0, 4.279960)]
    radius = max(radii)[1]
    center = (radius, radius)
    cells_x = 100
    cells_y = 100
    delta_x = np.repeat(radius * 2 / cells_x, cells_x)
    delta_y = np.repeat(radius * 2 / cells_y, cells_y)
    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84*0.0816]], [[0.0]]])
    # Update cross sections for cylinder
    # weight_map = math.monte_carlo_weight_matrix(delta_x, delta_y, \
    #                                             center, radii)
    weight_map = np.load("data/weight_maps_2d/cylinder_two_material.npy")
    data = dimensions.cylinder_cross_sections(weight_map, xs_total, \
                                              xs_scatter, xs_fission, \
                                              cells_x, cells_y)
    medium_map, xs_total, xs_scatter, xs_fission = data
    medium_map = dimensions.expand_cylinder_medium_map(medium_map, [1,2,3,4])
    # Material Parameters
    groups = 1
    angles = 6
    bc = [0, 0]
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": len(xs_total), "geometry": 1, \
            "spatial": 2, "qdim": 2, "bc_x": bc, "bcdim_x": 1, "bc_y": bc, \
            "bcdim_y": 1}
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)


@pytest.mark.cylinder2d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("layer"), ["small"]) #, "large"])
def test_cylinder_three_material(layer):
    if layer == "small":
        radii = [(0, 15.396916), (15.396916, 15.396916 + 1.830563)]
    elif layer == "large":
        radii = [(0, 14.606658), (14.606658, 14.606658 + 18.30563)]
    radius = max(radii)[1]
    center = (radius, radius)
    cells_x = 100
    cells_y = 100
    delta_x = np.repeat(radius * 2 / cells_x, cells_x)
    delta_y = np.repeat(radius * 2 / cells_y, cells_y)
    # Cross Sections
    xs_total = np.array([[0.54628], [0.54628], [0.0]])
    xs_scatter = np.array([[[0.464338]], [[0.491652]], [[0.0]]])
    xs_fission = np.array([[[1.70*0.054628]], [[0.0]], [[0.0]]])
    # Update cross sections for cylinder
    # from ants.utils import math
    # weight_map = math.monte_carlo_weight_matrix(delta_x, delta_y, \
    #                                             center, radii)
    weight_map = np.load("data/weight_maps_2d/cylinder_three_material_{}.npy".format(layer))
    data = dimensions.cylinder_cross_sections(weight_map, xs_total, \
                                              xs_scatter, xs_fission, \
                                              cells_x, cells_y)
    medium_map, xs_total, xs_scatter, xs_fission = data
    medium_map = dimensions.expand_cylinder_medium_map(medium_map, [1,2,3,4])
    # Material Parameters
    groups = 1
    angles = 4
    bc = [0, 0]
    info = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
            "groups": groups, "materials": len(xs_total), "geometry": 1, "spatial": 2, \
            "qdim": 2, "bc_x": bc, "bcdim_x": 1, "bc_y": bc, "bcdim_y": 1}
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    flux, keff = power(xs_total, xs_scatter, xs_fission, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, angle_w, info)
    assert abs(keff - 1) < 2e-3, "k-effective: " + str(keff)

# if __name__ == "__main__":
#     test_cylinder_three_material("small")
#     print()
#     test_cylinder_three_material("large")