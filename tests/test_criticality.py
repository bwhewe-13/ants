########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Criticality tests
#
########################################################################

# from ants import x_multi_group
from ants.critical1d import power_iteration as power
from tests import tools

import pytest
import numpy as np

@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_plutonium_01a(boundary):
    info = {"cells": 50, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, "bcdim": 0}
    mu, w = tools._x_angles(info["angles"], info["bc"])
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[3.24*0.0816]]])
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    width = 1.853722 * 2 if np.sum(boundary) == 0 else 1.853722
    edges = np.linspace(0, width, info["cells"]+1)
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        cell_width, mu, w, info)
    assert abs(keff - 1) < 2e-3

@pytest.mark.smoke
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_plutonium_01b(boundary):
    info = {"cells": 75, "angles": 16, "groups": 1, "materials": 1,
             "geometry": 1, "spatial": 2, "qdim": 2, "bc": boundary, "bcdim": 0}
    mu, w = tools._x_angles(info["angles"], info["bc"])
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[2.84*0.0816]]])
    info["cells"] = 150 if np.sum(info["bc"]) == 0 else 75
    width = 2.256751 * 2 if np.sum(info["bc"]) == 0 else 2.256751
    medium_map = np.zeros((info["cells"]), dtype=np.int32)
    edges = np.linspace(0, width, info["cells"]+1)
    cell_width = np.repeat(width / info["cells"], info["cells"])
    flux, keff = power(xs_total, xs_scatter, xs_fission,  medium_map, \
                        cell_width, mu, w, info)
    ref_flux = np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592])
    flux = tools.normalize(flux.flatten(), info["bc"])
    assert np.all(abs(flux - ref_flux) < 1e-2)
    assert abs(keff - 1) < 2e-3


# @pytest.mark.smoke
# @pytest.mark.crit_keff
# def test_one_group_plutonium_02a():
#     mu, angle_weight = benchmarks.angles("vacuum")
#     problem = benchmarks.OneGroup("slab", "vacuum")
#     problem.plutonium_02a()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.smoke
# @pytest.mark.crit_keff
# def test_one_group_plutonium_02b():
#     mu, angle_weight = benchmarks.angles("vacuum")
#     problem = benchmarks.OneGroup("slab", "vacuum")
#     problem.plutonium_02b()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3    


# @pytest.mark.crit_keff
# @pytest.mark.crit_flux
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_one_group_uranium_01a(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.OneGroup(geometry, boundary)
#     problem.uranium_01a()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     phi = benchmarks.normalize_phi(phi, geometry, boundary)
#     assert np.all(abs(phi - problem.flux_scale) < 5e-3)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.crit_keff
# @pytest.mark.crit_flux
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_one_group_heavy_water_01a(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.OneGroup(geometry, boundary)
#     problem.heavy_water_01a()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     phi = benchmarks.normalize_phi(phi, geometry, boundary)
#     assert np.all(abs(phi - problem.flux_scale) < 5e-3)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.smoke
# @pytest.mark.crit_infinite
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "reflected")])
# def test_one_group_uranium_reactor_01a(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.OneGroup(geometry, boundary)
#     problem.uranium_reactor_01a()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - problem.k_infinite) < 2e-3


# @pytest.mark.smoke
# @pytest.mark.crit_keff
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_two_group_plutonium_01(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.TwoGroup(geometry, boundary)
#     problem.plutonium_01()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.issue
# @pytest.mark.crit_keff
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_two_group_uranium_01(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.TwoGroup(geometry, boundary)
#     problem.uranium_01()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.crit_keff
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_two_group_uranium_aluminum(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.TwoGroup(geometry, boundary)
#     problem.uranium_aluminum()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3


# @pytest.mark.crit_keff
# @pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
#                             ("slab", "reflected"), ("sphere", "vacuum")])
# def test_two_group_uranium_reactor_01(geometry, boundary):
#     mu, angle_weight = benchmarks.angles(boundary)
#     problem = benchmarks.TwoGroup(geometry, boundary)
#     problem.uranium_reactor_01()
#     phi, keff = x_multi_group.criticality(problem.medium_map, problem.xs_total, \
#                     problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
#                     problem.params, problem.cell_width)
#     assert abs(keff - 1) < 2e-3
