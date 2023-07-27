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

import pytest
import numpy as np

import ants
from ants.fixed2d import source_iteration

# Path for reference solutions
PATH = "data/weight_matrix_2d/"

def cylinder_01():
    # Problem parameters
    cells_x = cells_y = 50
    angles = 4
    groups = 1

    # Spatial Layout
    radius = 4.279960
    coordinates = [(radius, radius), [radius]]

    delta_x = np.repeat(radius * 2 / cells_x, cells_x)
    edges_x = np.linspace(0, radius * 2, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    delta_y = np.repeat(radius * 2 / cells_y, cells_y)
    edges_y = np.linspace(0, radius * 2, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    # Boundary conditions
    bc_x = [0, 0]
    bc_y = [0, 0]
    # Cross Sections
    xs_total = np.array([[0.32640], [0.0]])
    xs_scatter = np.array([[[0.225216]], [[0.0]]])
    xs_fission = np.array([[[2.84*0.0816]], [[0.0]]])
    # Update cross sections for cylinder
    # weight_matrix = ants.weight_cylinder2d(coordinates, edges_x, \
    #                                          edges_y, N=250_000)
    weight_matrix = np.load(PATH + "cylinder_two_material.npy")
    medium_map, xs_total, xs_scatter, xs_fission \
        = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
    # Collect problem dictionary
    info = {
                "cells_x": cells_x,
                "cells_y": cells_y,
                "angles": angles, 
                "groups": groups, 
                "materials": xs_total.shape[0],
                "geometry": 1, 
                "spatial": 2, 
                "qdim": 1,
                "bc_x": bc_x,
                "bcdim_x": 1,
                "bc_y": bc_y,
                "bcdim_y": 1
            }
    # Collect angles
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    # Return pertinent information
    return xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        delta_y, angle_x, angle_y, angle_w, info


@pytest.mark.smoke
@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_x():
    # Call original problem
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, delta_y, \
        angle_x, angle_y, angle_w, info = cylinder_01()
    # Set boundary_y and external sources
    external = np.zeros((info["cells_x"] * info["cells_y"]))
    info["qdim"] = 1
    boundary_y = np.zeros((2))
    info["bcdim_y"] = 1
    # Different boundary_x dimensions
    bounds = [np.zeros((2,)), np.zeros((2, info["cells_y"])), \
              np.zeros((2, info["cells_y"], info["groups"])), \
              np.zeros((2, info["cells_y"], info["angles"]**2, info["groups"]))]
    flux = []
    for loc in [0, 1]:
        for bcdim_x, boundary_x in enumerate(bounds):
            boundary_x[loc] = 1.0
            info["bcdim_x"] = bcdim_x + 1
            result = source_iteration(xs_total, xs_scatter, xs_fission, \
                        external, boundary_x.flatten(), boundary_y, medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info)
            boundary_x *= 0.0
            if loc == 1:
                flux.append(result[::-1,:,0].flatten())
            else:
                flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.isclose(flux[ii], flux[ii+1], atol=1e-10).all()


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_boundary_y():
    # Call original problem
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, delta_y, \
        angle_x, angle_y, angle_w, info = cylinder_01()
    # Set boundary_x and external sources
    external = np.zeros((info["cells_x"] * info["cells_y"]))
    info["qdim"] = 1
    boundary_x = np.zeros((2))
    info["bcdim_x"] = 1
    # Different boundary_x dimensions
    bounds = [np.zeros((2,)), np.zeros((2, info["cells_x"])), \
              np.zeros((2, info["cells_x"], info["groups"])), \
              np.zeros((2, info["cells_x"], info["angles"]**2, info["groups"]))]
    flux = []
    for loc in [0, 1]:
        for bcdim_y, boundary_y in enumerate(bounds):
            boundary_y[loc] = 1
            info["bcdim_y"] = bcdim_y + 1
            result = source_iteration(xs_total, xs_scatter, xs_fission, \
                        external, boundary_x, boundary_y.flatten(), medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info)
            boundary_y *= 0.0
            if loc == 1:
                flux.append(result[:,::-1,0].flatten())
            else:
                flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii+1], atol=1e-10))


@pytest.mark.cylinder2d
@pytest.mark.dimensions
def test_dimensions_external_unit():
    # Call original problem
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, delta_y, \
        angle_x, angle_y, angle_w, info = cylinder_01()
    # Set boundary_x and boundary_y
    boundary_x = np.zeros((2))
    info["bcdim_x"] = 1
    boundary_y = np.zeros((2))
    info["bcdim_y"] = 1
    # Different boundary_x dimensions
    externals = [np.ones((info["cells_x"] * info["cells_y"])), \
                 np.ones((info["cells_x"] * info["cells_y"] * info["groups"])), \
                 np.ones((info["cells_x"] * info["cells_y"] * info["angles"]**2 * info["groups"]))]
    flux = []
    for qdim, external in enumerate(externals):
        info["qdim"] = qdim + 1
        result = source_iteration(xs_total, xs_scatter, xs_fission, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)
        flux.append(result[:,:,0].flatten())
    for ii in range(len(flux) - 1):
        assert np.all(np.isclose(flux[ii], flux[ii+1], atol=1e-10))
