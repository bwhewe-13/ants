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

from ants.cyants import multi_group
import tests.criticality_benchmarks as benchmarks

import pytest
import numpy as np


@pytest.mark.crit_keff
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
                            ("slab", "reflected")])
def test_one_group_plutonium_01a(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.OneGroup(geometry, boundary)
    problem.plutonium_01a()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    assert abs(keff - 1) < 2e-3


@pytest.mark.smoke
@pytest.mark.crit_keff
@pytest.mark.crit_flux
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
                            ("slab", "reflected"), ("sphere", "vacuum")])
def test_one_group_plutonium_01b(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.OneGroup(geometry, boundary)
    problem.plutonium_01b()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    phi = benchmarks.normalize_phi(phi, geometry, boundary)
    assert np.all(abs(phi - problem.flux_scale) < 5e-3)
    assert abs(keff - 1) < 2e-3


@pytest.mark.crit_keff
@pytest.mark.crit_flux
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
                            ("slab", "reflected"), ("sphere", "vacuum")])
def test_one_group_uranium_01a(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.OneGroup(geometry, boundary)
    problem.uranium_01a()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    phi = benchmarks.normalize_phi(phi, geometry, boundary)
    assert np.all(abs(phi - problem.flux_scale) < 5e-3)
    assert abs(keff - 1) < 2e-3


@pytest.mark.crit_keff
@pytest.mark.crit_flux
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
                            ("slab", "reflected"), ("sphere", "vacuum")])
def test_one_group_heavy_water_01a(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.OneGroup(geometry, boundary)
    problem.heavy_water_01a()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    phi = benchmarks.normalize_phi(phi, geometry, boundary)
    assert np.all(abs(phi - problem.flux_scale) < 5e-3)
    assert abs(keff - 1) < 2e-3


@pytest.mark.smoke
@pytest.mark.crit_infinite
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "reflected")])
def test_one_group_uranium_reactor_01a(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.OneGroup(geometry, boundary)
    problem.uranium_reactor_01a()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    assert abs(keff - problem.k_infinite) < 2e-3


@pytest.mark.smoke
@pytest.mark.crit_keff
@pytest.mark.parametrize(("geometry", "boundary"), [("slab", "vacuum"), \
                            ("slab", "reflected"), ("sphere", "vacuum")])
def test_two_group_plutonium_01a(geometry, boundary):
    mu, angle_weight = benchmarks.angles(boundary)
    problem = benchmarks.TwoGroup(geometry, boundary)
    problem.plutonium_01a()
    phi, keff = multi_group.criticality(problem.medium_map, problem.xs_total, \
                    problem.xs_scatter, problem.xs_fission, mu, angle_weight, \
                    problem.params, problem.cell_width)
    assert abs(keff - 1) < 2e-3