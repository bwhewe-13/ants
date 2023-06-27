########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Method of Manufactured Solutions for Two Dimensional Slabs
#
########################################################################

import pytest
import numpy as np

import ants
from ants.fixed2d import source_iteration
from ants.utils import manufactured_2d as mms


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_01(angular, spatial):
    info = {"cells_x": 200, "cells_y": 200, "angles": 2, "groups": 1, 
            "materials": 1, "geometry": 1, "spatial": spatial, "qdim": 3, 
            "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
            "angular": angular}
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
    shape_q = (info["cells_x"], info["cells_y"], info["angles"]**2, info["groups"])
    external = ants.externals2d(0.5, shape_q)
    external = external.flatten()
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-01", shape_x, shape_y, \
                                        angle_x=angle_x, angle_y=angle_y, \
                                        centers_x=centers_x)
    boundary_x = boundary_x.flatten()
    boundary_y = boundary_y.flatten()
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_01(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        flux = np.transpose(flux, axes=(1, 0, 2))
    else:
        flux = np.transpose(flux, axes=(1, 0, 2, 3))
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux" 


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_02(angular, spatial):
    info = {"cells_x": 200, "cells_y": 200, "angles": 2, "groups": 1, 
            "materials": 1, "geometry": 1, "spatial": spatial, "qdim": 3, 
            "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
            "angular": angular}
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
    shape_q = (info["cells_x"], info["cells_y"], info["angles"]**2, info["groups"])
    external = ants.externals2d(1.0, shape_q)
    external = external.flatten()
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-02", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y, \
                                    centers_x=centers_x, centers_y=centers_y)
    boundary_x = boundary_x.flatten()
    boundary_y = boundary_y.flatten()
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_02(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        flux = np.transpose(flux, axes=(1, 0, 2))
    else:
        flux = np.transpose(flux, axes=(1, 0, 2, 3))
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_03(angular, spatial):
    info = {"cells_x": 200, "cells_y": 200, "angles": 4, "groups": 1, 
        "materials": 1, "geometry": 1, "spatial": spatial, "qdim": 3, 
        "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
        "angular": angular}
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
    external = external.flatten()
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-03", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y)
    boundary_x = boundary_x.flatten()
    boundary_y = boundary_y.flatten()
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)
    exact = mms.solution_mms_03(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        flux = np.transpose(flux, axes=(1, 0, 2))
    else:
        flux = np.transpose(flux, axes=(1, 0, 2, 3))
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), [(True, 1), (True, 2), \
                         (False, 1), (False, 2)])
def test_manufactured_04(angular, spatial):
    info = {"cells_x": 200, "cells_y": 200, "angles": 4, "groups": 1, 
        "materials": 1, "geometry": 1, "spatial": spatial, "qdim": 3, 
        "bc_x": [0, 0], "bcdim_x": 4, "bc_y": [0, 0], "bcdim_y": 4,
        "angular": angular}
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
    external = external.flatten()
    # Boundary sources
    shape_x = (2, info["cells_y"]) + shape_q[2:]
    shape_y = (2, info["cells_x"]) + shape_q[2:]
    boundary_x, boundary_y = ants.boundaries2d("mms-04", shape_x, shape_y, \
                                    angle_x=angle_x, angle_y=angle_y, \
                                    centers_x=centers_x, centers_y=centers_y)
    boundary_x = boundary_x.flatten()
    boundary_y = boundary_y.flatten()
    # Run Source Iteration 
    flux = source_iteration(xs_total, xs_scatter, xs_fission, external, \
                        boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, info)

    exact = mms.solution_mms_04(centers_x, centers_y, angle_x, angle_y)
    # Rearrange dimensions
    if not angular:
        exact = np.sum(exact * angle_w[None,None,:,None], axis=2)
        flux = np.transpose(flux, axes=(1, 0, 2))
    else:
        flux = np.transpose(flux, axes=(1, 0, 2, 3))
    # Evaluate
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact[(..., 0)], atol=atol).all(), \
            "Incorrect flux"