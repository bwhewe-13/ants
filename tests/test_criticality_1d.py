########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Benchmarks from "Analytical Benchmark Test Set for Criticality Code 
# Verification" from Los Alamos National Lab
#
########################################################################

import ants
from ants.critical1d import power_iteration as power

import pytest
import numpy as np

def normalize(flux, boundary):
    cells = len(flux)
    if boundary == [0, 0]:
        flux /= flux[int(cells*0.5)]
        idx = [int(cells*0.375), int(cells*0.25), int(cells*0.125), 0]
    elif boundary == [0, 1]:
        flux /= flux[-1]
        idx = [int(cells*0.75), int(cells*0.5), int(cells*0.25), 0]
    else: 
        flux /= flux[0]
        idx = [int(cells*0.25), int(cells*0.5), int(cells*0.75), -1]
    nflux = np.array([flux[ii] for ii in idx])
    return nflux

@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_slab_plutonium_01a(boundary):
    params = {"cells": 50, "angles": 16, "groups": 1, "materials": 1,
            "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
            "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[3.24*0.0816]]])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    length = 1.853722 * 2 if np.sum(boundary) == 0 else 1.853722
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_slab_plutonium_01b(boundary):
    params = {"cells": 75, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[2.84*0.0816]]])
    params["cells"] = 150 if np.sum(params["bc"]) == 0 else 75
    length = 2.256751 * 2 if np.sum(params["bc"]) == 0 else 2.256751
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.power_iteration
def test_one_group_sphere_plutonium_01b():
    params = {"cells": 150, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[2.84*0.0816]]])
    length = 6.082547
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.93538006, 0.75575352, 0.49884364, 0.19222603])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_one_group_slab_plutonium_02a():
    params = {"cells": 500, "angles": 16, "groups": 1, "materials": 2,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": [0, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640], [0.32640]])
    xs_scatter = np.array([[[0.225216]], [[0.293760]]])
    xs_fission = np.array([[[3.24*0.0816]], [[0.0]]])
    length = 1.478401 * 2 + 3.063725
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    materials = [[0, "fuel", "0 - 2.956802"], \
                [1, "moderator", "2.956802 - 6.020527"]]
    medium_map = ants._medium_map(materials, edges_x)    
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_one_group_slab_plutonium_02b():
    params = {"cells": 502, "angles": 16, "groups": 1, "materials": 2,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": [0, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640], [0.32640]])
    xs_scatter = np.array([[[0.225216]], [[0.293760]]])
    xs_fission = np.array([[[3.24*0.0816]], [[0.0]]])
    length = np.sum([1.531863, 1.317831*2, 1.531863])
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    materials = [[0, "fuel", "1.531863 - 4.167525"], \
                [1, "moderator", "0 - 1.531863, 4.167525 - 5.699388"]]
    medium_map = ants._medium_map(materials, edges_x)    
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_slab_uranium_01a(boundary):
    params = {"cells": 75, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.248064]]])
    xs_fission = np.array([[[2.70*0.065280]]])
    params["cells"] = 150 if np.sum(params["bc"]) == 0 else 75
    length = 2.872934 * 2 if np.sum(params["bc"]) == 0 else 2.872934
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.9669506, 0.8686259, 0.7055218, 0.4461912])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_one_group_sphere_uranium_01a():
    params = {"cells": 150, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.248064]]])
    xs_fission = np.array([[[2.70*0.065280]]])
    length = 7.428998
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.93244907, 0.74553332, 0.48095413, 0.17177706])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_slab_heavy_water_01a(boundary):
    params = {"cells": 600, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[0.464338]]])
    xs_fission = np.array([[[1.70*0.054628]]])
    params["cells"] = 600 if np.sum(params["bc"]) == 0 else 300
    length = 10.371065 * 2 if np.sum(params["bc"]) == 0 else 10.371065
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.93945236, 0.76504084, 0.49690627, 0.13893858])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_one_group_sphere_heavy_water_01a():
    params = {"cells": 300, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[0.464338]]])
    xs_fission = np.array([[[1.70*0.054628]]])
    length = 22.017156
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    ref_flux = np.array([0.91063756, 0.67099621, 0.35561622, 0.04678614])
    flux = normalize(flux.flatten(), params["bc"])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_one_group_slab_uranium_reactor_01a():
    params = {"cells": 200, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": [0, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.407407]])
    xs_scatter = np.array([[[0.328042]]])
    xs_fission = np.array([[[2.50*0.06922744]]])
    length = 200 * 250
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    kinfinite = 2.1806667
    assert abs(keff - kinfinite) < 2e-3, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_slab_plutonium_01(boundary):
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.3360,0.2208]])
    xs_scatter = np.array([np.array([[0.23616, 0.0],[0.0432, 0.0792]]).T])
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    params["cells"] = 200 if np.sum(params["bc"]) == 0 else 100
    length = 1.795602 * 2 if np.sum(params["bc"]) == 0 else 1.795602
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.power_iteration
def test_two_group_sphere_plutonium_01():
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.3360,0.2208]])
    xs_scatter = np.array([np.array([[0.23616, 0.0],[0.0432, 0.0792]]).T])
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    length = 5.231567
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.issue
# @pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_slab_uranium_01(boundary):
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.3456, 0.2160]])
    xs_scatter = np.array([np.array([[0.26304, 0.0],[0.0720, 0.078240]]).T])
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.50, 2.70]])
    sigmaf = np.array([[0.06912, 0.06912]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    params["cells"] = 200 if np.sum(params["bc"]) == 0 else 100
    length = 3.006375 * 2 if np.sum(params["bc"]) == 0 else 3.006375
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.issue
# @pytest.mark.power_iteration
def test_two_group_sphere_uranium_01():
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[0.3456, 0.2160]])
    xs_scatter = np.array([np.array([[0.26304, 0.0],[0.0720, 0.078240]]).T])
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.50, 2.70]])
    sigmaf = np.array([[0.06912, 0.06912]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    length = 7.909444
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_slab_uranium_aluminum(boundary):
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[1.27698, 0.26817]])
    xs_scatter = np.array([np.array([[1.21313, 0.0],[0.020432, 0.247516]]).T])
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.83, 0.0]])
    sigmaf = np.array([[0.06070636042, 0.0]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    params["cells"] = 200 if np.sum(params["bc"]) == 0 else 100
    length = 7.830630 * 2 if np.sum(params["bc"]) == 0 else 7.830630
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_two_group_sphere_uranium_aluminum():
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[1.27698, 0.26817]])
    xs_scatter = np.array([np.array([[1.21313, 0.0],[0.020432, 0.247516]]).T])
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.83, 0.0]])
    sigmaf = np.array([[0.06070636042, 0.0]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    length = 17.66738
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_slab_uranium_reactor_01(boundary):
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[2.52025, 0.65696]])
    xs_scatter = np.array([np.array([[2.44383, 0.0],[0.029227, 0.62568]]).T])
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    params["cells"] = 200 if np.sum(params["bc"]) == 0 else 100
    length = 7.566853 * 2 if np.sum(params["bc"]) == 0 else 7.566853
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"


@pytest.mark.power_iteration
def test_two_group_sphere_uranium_reactor_01():
    params = {"cells": 200, "angles": 20, "groups": 2, "materials": 1,
             "geometry": 2, "spatial": 2, "qdim": 2, "bc": [1, 0], 
             "bcdim": 0, "angular": False}
    angle_x, angle_w = ants._angle_x(params)
    xs_total = np.array([[2.52025, 0.65696]])
    xs_scatter = np.array([np.array([[2.44383, 0.0],[0.029227, 0.62568]]).T])
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    length = 16.049836
    edges_x = np.linspace(0, length, params["cells"]+1)
    delta_x = np.repeat(length / params["cells"], params["cells"])
    medium_map = np.zeros((params["cells"]), dtype=np.int32)
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        delta_x, angle_x, angle_w, params)
    assert abs(keff - 1.) < 2e-3, str(keff) + " not critical"
