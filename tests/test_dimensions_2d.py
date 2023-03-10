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

import ants
from ants.fixed2d import source_iteration as iteration
from ants.utils import dimensions
# from tests import tools

import pytest
import numpy as np


@pytest.mark.smoke
@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_x():
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
    weight_map = np.load("data/weight_map_cylinder_two_material.npy")
    data = dimensions.cylinder_cross_sections(weight_map, xs_total, \
                                              xs_scatter, xs_fission, \
                                              cells_x, cells_y)
    medium_map, xs_total, xs_scatter, xs_fission = data
    medium_map = dimensions.expand_cylinder_medium_map(medium_map, [1,2,3,4])
    # Material Parameters
    groups = 1
    angles = 4
    bc = [0, 0]
    angles, angle_x, angle_y, angle_w = ants.calculate_xy_angles(angles, [bc,bc])
    velocity = np.ones((groups))
    external = np.zeros((cells_x, cells_y)).flatten()
    # Set boundary terms
    boundary_y = np.zeros((2,))
    bounds = [np.zeros((2,)), np.zeros((2, cells_y)), \
              np.zeros((2, cells_y, groups)), \
              np.zeros((2, cells_y, angles, groups))]
    params = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
             "groups": groups, "materials": len(xs_total), "geometry": 1, \
             "spatial": 2, "qdim": 1, "bc_x": bc, "bcdim_x": 0, \
             "bc_y": bc, "bcdim_y": 0, "steps": 0, "dt": 0, \
             "adjoint": False, "angular": False}
    flux = []
    for loc in [0, 1]:
        for bcdim_x, boundary_x in enumerate(bounds):
            boundary_x[loc] = 1
            params["bcdim_x"] = bcdim_x
            result = iteration(xs_total, xs_scatter, xs_fission, external, \
                               boundary_x.flatten(), boundary_y, \
                               medium_map.flatten(), delta_x, delta_y, \
                               angle_x, angle_y, angle_w, params)
            boundary_x *= 0.0
            if loc == 1:
                flux.append(result[::-1,:,0].flatten())
            else:
                flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii+1], atol=1e-10))


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_y():
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
    weight_map = np.load("data/weight_map_cylinder_two_material.npy")
    data = dimensions.cylinder_cross_sections(weight_map, xs_total, \
                                              xs_scatter, xs_fission, \
                                              cells_x, cells_y)
    medium_map, xs_total, xs_scatter, xs_fission = data
    medium_map = dimensions.expand_cylinder_medium_map(medium_map, [1,2,3,4])
    # Material Parameters
    groups = 1
    angles = 4
    bc = [0, 0]
    angles, angle_x, angle_y, angle_w = ants.calculate_xy_angles(angles, [bc,bc])
    velocity = np.ones((groups))
    external = np.zeros((cells_x, cells_y)).flatten()
    # Set boundary terms
    boundary_x = np.zeros((2,))
    bounds = [np.zeros((2,)), np.zeros((2, cells_y)), \
              np.zeros((2, cells_y, groups)), \
              np.zeros((2, cells_y, angles, groups))]
    params = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
             "groups": groups, "materials": len(xs_total), "geometry": 1, \
             "spatial": 2, "qdim": 1, "bc_x": bc, "bcdim_x": 0, \
             "bc_y": bc, "bcdim_y": 0, "steps": 0, "dt": 0, \
             "adjoint": False, "angular": False}
    flux = []
    for loc in [0, 1]:
        for bcdim_y, boundary_y in enumerate(bounds):
            boundary_y[loc] = 1
            params["bcdim_y"] = bcdim_y
            result = iteration(xs_total, xs_scatter, xs_fission, external, \
                               boundary_x, boundary_y.flatten(), \
                               medium_map.flatten(), delta_x, delta_y, \
                               angle_x, angle_y, angle_w, params)
            boundary_y *= 0.0
            if loc == 1:
                flux.append(result[:,::-1,0].flatten())
            else:
                flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii+1], atol=1e-10))


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_external_zero():
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
    weight_map = np.load("data/weight_map_cylinder_two_material.npy")
    data = dimensions.cylinder_cross_sections(weight_map, xs_total, \
                                              xs_scatter, xs_fission, \
                                              cells_x, cells_y)
    medium_map, xs_total, xs_scatter, xs_fission = data
    medium_map = dimensions.expand_cylinder_medium_map(medium_map, [1,2,3,4])
    # Material Parameters
    groups = 1
    angles = 4
    bc = [0, 0]
    angles, angle_x, angle_y, angle_w = ants.calculate_xy_angles(angles, [bc,bc])
    velocity = np.ones((groups))
    external = np.zeros((cells_x, cells_y)).flatten()
    # Set boundary terms
    boundary_x = np.zeros((2,))
    boundary_x[0] = 1.0
    boundary_y = np.zeros((2,))
    externals = [np.zeros((cells_x, cells_y)), \
                 np.zeros((cells_x, cells_y, groups)), \
                 np.zeros((cells_x, cells_y, angles, groups))]
    params = {"cells_x": cells_x, "cells_y": cells_y, "angles": angles, \
             "groups": groups, "materials": len(xs_total), "geometry": 1, \
             "spatial": 2, "qdim": 1, "bc_x": bc, "bcdim_x": 0, \
             "bc_y": bc, "bcdim_y": 0, "steps": 0, "dt": 0, \
             "adjoint": False, "angular": False}
    flux = []
    for qdim, external in enumerate(externals):
        params["qdim"] = qdim + 1
        result = iteration(xs_total, xs_scatter, xs_fission, \
                           external.flatten(), boundary_x, boundary_y, \
                           medium_map.flatten(), delta_x, delta_y, \
                           angle_x, angle_y, angle_w, params)
        flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii+1], atol=1e-10))

# if __name__ == "__main__":
#     test_dimensions_boundary_x()
