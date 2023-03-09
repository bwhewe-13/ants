########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Method of Manufactured Solutions for One Dimensional Slabs
#
########################################################################

from ants.fixed1d import source_iteration as si
from ants.utils import manufactured_solutions as mms
from tests import tools

import pytest
import numpy as np


@pytest.mark.smoke
@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_one_material01(angular):
    angular = True if angular == "True" else False
    info = {"cells": 400, "angles": 4, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 1, "bc": [0, 0], "bcdim": 0, 
             "steps": 0, "dt": 0, "angular": angular, "adjoint": False}
    # mu, w = tools._x_angles(info["angles"], info["bc"])
    info, mu, w = tools._x_angles(info)
    xs_total = np.array([1.])[:,None]
    xs_scatter = np.array([[0.]])[:,None]
    xs_fission = np.array([[0.]])[:,None]
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    width = 1
    edges = np.linspace(0, width, info["cells"]+1)
    source = np.ones((info["cells"]))
    boundary = np.zeros((2))
    boundary[0] = 1
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux = si(xs_total, xs_scatter, xs_fission, source, boundary, \
                medium_map, cell_width, mu, w, info)
    xspace = 0.5 * (edges[1:] + edges[:-1])
    ref_flux = mms.solution_one_material_01(xspace, mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * w, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_one_material02(angular):
    angular = True if angular == "True" else False
    info = {"cells": 400, "angles": 4, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 1, "bc": [0, 0], "bcdim": 0, 
             "steps": 0, "dt": 0, "angular": angular, "adjoint": False}
    # mu, w = tools._x_angles(info["angles"], info["bc"])
    info, mu, w = tools._x_angles(info)
    xs_total = np.array([1.])[:,None]
    xs_scatter = np.array([[0.]])[:,None]
    xs_fission = np.array([[0.]])[:,None]
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    width = 1
    edges = np.linspace(0, width, info["cells"]+1)
    source = np.ones((info["cells"])) * 0.5
    boundary = np.zeros((2))
    boundary[0] = 1
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux = si(xs_total, xs_scatter, xs_fission, source, boundary, \
                medium_map, cell_width, mu, w, info)
    xspace = 0.5 * (edges[1:] + edges[:-1])
    ref_flux = mms.solution_one_material_02(xspace, mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * w, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_two_material01(angular):
    angular = True if angular == "True" else False
    info = {"cells": 400, "angles": 4, "groups": 1, "materials": 2,
             "geometry": 1, "spatial": 2, "qdim": 3, "bc": [0, 0], "bcdim": 2,
             "steps": 0, "dt": 0, "angular": angular, "adjoint": False}
    # mu, w = tools._x_angles(info["angles"], info["bc"])
    info, mu, w = tools._x_angles(info)
    xs_total = np.array([[1.],[1.]])
    xs_scatter = np.array([[[0.3]],[[0.9]]])
    xs_fission = np.array([[[0.0]],[[0.0]]])
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    medium_map[200:] = 1
    width = 2
    edges = np.linspace(0, width, info["cells"]+1)
    source = tools._mms_two_material(edges, mu) 
    boundary = tools._mms_boundary("mms-03", mu)
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux = si(xs_total, xs_scatter, xs_fission, source, boundary, \
                medium_map, cell_width, mu, w, info)
    xspace = 0.5 * (edges[1:] + edges[:-1])
    ref_flux = mms.solution_two_material_01(xspace, mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * w, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_two_material02(angular):
    angular = True if angular == "True" else False
    info = {"cells": 400, "angles": 4, "groups": 1, "materials": 2,
             "geometry": 1, "spatial": 2, "qdim": 3, "bc": [0, 0], "bcdim": 2,
             "steps": 0, "dt": 0, "angular": angular, "adjoint": False}
    # mu, w = tools._x_angles(info["angles"], info["bc"])
    info, mu, w = tools._x_angles(info)
    xs_total = np.array([[1.],[1.]])
    xs_scatter = np.array([[[0.3]],[[0.9]]])
    xs_fission = np.array([[[0.0]],[[0.0]]])
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    medium_map[200:] = 1
    width = 2
    edges = np.linspace(0, width, info["cells"]+1)
    source = tools._mms_two_material_angle(edges, mu) 
    boundary = tools._mms_boundary("mms-04", mu)
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux = si(xs_total, xs_scatter, xs_fission, source, boundary, \
                medium_map, cell_width, mu, w, info)
    xspace = 0.5 * (edges[1:] + edges[:-1])
    ref_flux = mms.solution_two_material_02(xspace, mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * w, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)
