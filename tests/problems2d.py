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

# Path for reference solutions
PATH = "data/references_multigroup/"


def manufactured_01(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, "groups": 1, 
            "materials": 1, "geometry": 1, "spatial": 2, "qdim": 3, 
            "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
            "angular": False}
    
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
    
    # Cross sections
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    
    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    
    # External Source
    external = 0.5 * np.ones((info["cells_x"], info["cells_y"], 1, 1)) 
    
    # Boundary sources
    shape_x = (2, info["cells_y"], info["angles"]**2, info["groups"])
    shape_y = (2, info["cells_x"], info["angles"]**2, info["groups"])
    boundary_x, boundary_y = ants.boundaries2d("mms-01", shape_x, shape_y, \
                                        angle_x=angle_x, angle_y=angle_y, \
                                        centers_x=centers_x)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, centers_x, centers_y


def manufactured_02(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, "groups": 1, 
            "materials": 1, "geometry": 1, "spatial": 2, "qdim": 3, 
            "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
            "angular": False}
    
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
    
    # Cross sections
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    
    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    # External Source
    external = np.ones((info["cells_x"], info["cells_y"], 1, 1))
    
    # Boundary sources
    shape_x = (2, info["cells_y"], info["angles"]**2, info["groups"])
    shape_y = (2, info["cells_x"], info["angles"]**2, info["groups"])
    boundary_x, boundary_y = ants.boundaries2d("mms-02", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y, \
                                    centers_x=centers_x, centers_y=centers_y)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, centers_x, centers_y


def manufactured_03(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, "groups": 1, 
        "materials": 1, "geometry": 1, "spatial": 2, "qdim": 3, 
        "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
        "angular": False}
    
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
    
    # Cross sections
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.5]]])
    xs_fission = np.array([[[0.0]]])
    
    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    
    # External Source
    shape_q = (info["cells_x"], info["cells_y"], info["angles"]**2, info["groups"])
    external = ants.externals2d("mms-03", shape_q, angle_x=angle_x, angle_y=angle_y)
    
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-03", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, centers_x, centers_y


def manufactured_04(cells, angles):
    info = {"cells_x": cells, "cells_y": cells, "angles": angles, "groups": 1, 
        "materials": 1, "geometry": 1, "spatial": 2, "qdim": 3, 
        "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
        "angular": False}
    
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
    
    # Cross sections
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.5]]])
    xs_fission = np.array([[[0.0]]])
    
    # Layout
    medium_map = np.zeros((info["cells_x"], info["cells_y"]), dtype=np.int32)
    angle_x, angle_y, angle_w = ants.angular_xy(info)
    # External Source
    shape_q = (info["cells_x"], info["cells_y"], info["angles"]**2, info["groups"])
    external = ants.externals2d("mms-04", shape_q, angle_x=angle_x, angle_y=angle_y, \
                                centers_x=centers_x, centers_y=centers_y)
    external = np.transpose(external, axes=(1, 0, 2, 3))
    
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-04", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y, \
                                    centers_x=centers_x, centers_y=centers_y)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, angle_x, angle_y, \
        angle_w, info, centers_x, centers_y