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
from ants.utils import manufactured_2d as mms

# Path for reference solutions
PATH = "data/references_multigroup/"


def manufactured_ss_01(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1, "spatial": 2, 
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False}
    
    # Spatial dimension x
    length_x = 1.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    
    # Spatial dimension y
    length_y = 1.
    delta_y = np.repeat(length_y / info["cells_y"], info["cells_y"])
    edges_y = np.linspace(0, length_y, info["cells_y"]+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    # Externals
    external = 0.5 * np.ones((info["cells_x"], info["cells_y"], 1, 1)) 
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_01(centers_x, \
                                            centers_y, angle_x, angle_y)

    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, edges_x, edges_y


def manufactured_ss_02(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1, "spatial": 2, \
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False}
    
    # Spatial dimension x
    length_x = 1.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    
    # Spatial dimension y
    length_y = 1.
    delta_y = np.repeat(length_y / info["cells_y"], info["cells_y"])
    edges_y = np.linspace(0, length_y, info["cells_y"]+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    
    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    
    # Externals
    external = np.ones((info["cells_x"], info["cells_y"], 1, 1))
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_02(centers_x, \
                                            centers_y, angle_x, angle_y)

    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, edges_x, edges_y


def manufactured_ss_03(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1, "spatial": 2, 
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False}
    
    # Spatial dimension x
    length_x = 2.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    
    # Spatial dimension y
    length_y = 2.
    delta_y = np.repeat(length_y / info["cells_y"], info["cells_y"])
    edges_y = np.linspace(0, length_y, info["cells_y"]+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.5]]])
    xs_fission = np.array([[[0.0]]])
    
    # Externals
    external = ants.external2d.manufactured_ss_03(centers_x, centers_y, \
                                                    angle_x, angle_y)
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_03(centers_x, \
                                            centers_y, angle_x, angle_y)

    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, edges_x, edges_y


def manufactured_ss_04(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1, "spatial": 2, 
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False}
    
    # Spatial dimension x
    length_x = 2.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    
    # Spatial dimension y
    length_y = 2.
    delta_y = np.repeat(length_y / info["cells_y"], info["cells_y"])
    edges_y = np.linspace(0, length_y, info["cells_y"]+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    
    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.5]]])
    xs_fission = np.array([[[0.0]]])
    
    # Externals
    external = ants.external2d.manufactured_ss_04(centers_x, centers_y, \
                                                    angle_x, angle_y)
    boundary_x, boundary_y = ants.boundary2d.manufactured_ss_04(centers_x, \
                                            centers_y, angle_x, angle_y)

    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, edges_x, edges_y


def manufactured_td_01(cells, angles, edges_t, dt, temporal=1):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1,  "spatial": 2, \
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False,  \
            "steps": edges_t.shape[0] - 1, "dt": dt}

    # Spatial Dimensions
    cells_x = cells
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = cells
    length_y = 2.
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    velocity = np.ones((info["groups"],))

    # Sources
    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_01(centers_x, centers_y, angle_x, \
                                          angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux, )

        external = ants.external2d.manufactured_td_01(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)[1:]
    # Crank Nicolson
    elif temporal == 2:
        initial_flux_x = mms.solution_td_01(edges_x, centers_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux_y = mms.solution_td_01(centers_x, edges_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux_x, initial_flux_y)

        external = ants.external2d.manufactured_td_01(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_01(centers_x, centers_y, angle_x, \
                                          angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux, )

        external = ants.external2d.manufactured_td_01(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)[1:]
    # TR-BDF2
    elif temporal == 4:
        initial_flux_x = mms.solution_td_01(edges_x, centers_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux_y = mms.solution_td_01(centers_x, edges_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux_x, initial_flux_y)

        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external2d.manufactured_td_01(centers_x, centers_y, \
                                          angle_x, angle_y, gamma_steps)

    boundary_x = 2 * np.ones((1, 2, 1, 1, 1))
    boundary_y = 2 * np.ones((1, 2, 1, 1, 1))
    # Layout
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)

    return *initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        angle_x, angle_y, angle_w, info


def manufactured_td_02(cells, angles, edges_t, dt, temporal=1):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, \
            "groups": 1, "materials": 1, "geometry": 1,  "spatial": 2, \
            "bc_x": [0, 0], "bc_y": [0, 0], "angular": False,  \
            "steps": edges_t.shape[0] - 1, "dt": dt}

    # Spatial Dimensions
    cells_x = cells
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    cells_y = cells
    length_y = np.pi
    delta_y = np.repeat(length_y / cells_y, cells_y)
    edges_y = np.linspace(0, length_y, cells_y+1)
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

    # Angular
    angle_x, angle_y, angle_w = ants.angular_xy(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.25]]])
    xs_fission = np.array([[[0.0]]])
    velocity = np.ones((info["groups"],))

    # Sources
    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_02(centers_x, centers_y, angle_x, \
                                          angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux, )

        external = ants.external2d.manufactured_td_02(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)[1:]
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(centers_x, \
                                    centers_y, angle_x, angle_y, edges_t[1:])
    # Crank Nicolson
    elif temporal == 2:
        initial_flux_x = mms.solution_td_02(edges_x, centers_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux_y = mms.solution_td_02(centers_x, edges_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux_x, initial_flux_y)

        external = ants.external2d.manufactured_td_02(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(centers_x, \
                                    centers_y, angle_x, angle_y, edges_t[1:])
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_02(centers_x, centers_y, angle_x, \
                                          angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux, )

        external = ants.external2d.manufactured_td_02(centers_x, centers_y, \
                                          angle_x, angle_y, edges_t)[1:]
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(centers_x, \
                                    centers_y, angle_x, angle_y, edges_t[1:])
    # TR-BDF2
    elif temporal == 4:
        initial_flux_x = mms.solution_td_02(edges_x, centers_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux_y = mms.solution_td_02(centers_x, edges_y, angle_x, angle_y, np.array([0.0]))[0]
        initial_flux = (initial_flux_x, initial_flux_y)

        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external2d.manufactured_td_02(centers_x, centers_y, \
                                          angle_x, angle_y, gamma_steps)
        boundary_x, boundary_y = ants.boundary2d.manufactured_td_02(centers_x, \
                                    centers_y, angle_x, angle_y, gamma_steps[1:])

    # Layout
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)   

    return *initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        angle_x, angle_y, angle_w, info
