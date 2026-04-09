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

import os

import numpy as np
import pytest

import ants
from ants.critical1d import k_criticality
from ants.datatypes import SolverData
from tests import criticality_benchmarks as benchmarks
from tests import problems1d


def normalize(flux, boundary):
    cells_x = len(flux)
    if boundary == [0, 0]:
        flux /= flux[int(cells_x * 0.5)]
        idx = [int(cells_x * 0.375), int(cells_x * 0.25), int(cells_x * 0.125), 0]
    elif boundary == [0, 1]:
        flux /= flux[-1]
        idx = [int(cells_x * 0.75), int(cells_x * 0.5), int(cells_x * 0.25), 0]
    else:
        flux /= flux[0]
        idx = [int(cells_x * 0.25), int(cells_x * 0.5), int(cells_x * 0.75), -1]
    nflux = np.array([flux[ii] for ii in idx])
    return nflux


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_pua_1_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 100 if np.sum(boundary) == 0 else 50
    materials, geometry = benchmarks.PUa_1_0(cells_x, boundary)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_pub_1_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 150 if np.sum(boundary) == 0 else 75
    materials, geometry = benchmarks.PUb_1_0(cells_x, boundary, 1)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592])
    flux = normalize(flux.flatten(), boundary)
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_pub_1_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.PUb_1_0(150, [1, 0], 2)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.93538006, 0.75575352, 0.49884364, 0.19222603])
    flux = normalize(flux.flatten(), [1, 0])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
def test_pua_h20_1_0_nonsymmetric_slab():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[0, 0])
    materials, geometry = benchmarks.PUa_H20_1_0(500, nonsymmetric=True)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
def test_pua_h20_1_0_symmetric_slab():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[0, 0])
    materials, geometry = benchmarks.PUa_H20_1_0(502, nonsymmetric=False)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_ua_1_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 150 if np.sum(boundary) == 0 else 75
    materials, geometry = benchmarks.Ua_1_0(cells_x, boundary, geometry_type=1)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.9669506, 0.8686259, 0.7055218, 0.4461912])
    flux = normalize(flux.flatten(), boundary)
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_ua_1_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.Ua_1_0(150, [1, 0], geometry_type=2)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.93244907, 0.74553332, 0.48095413, 0.17177706])
    flux = normalize(flux.flatten(), [1, 0])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_ud2o_1_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    materials, geometry = benchmarks.UD2O_1_0(600, boundary, geometry_type=1)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.93945236, 0.76504084, 0.49690627, 0.13893858])
    flux = normalize(flux.flatten(), boundary)
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_ud2o_1_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.UD2O_1_0(300, [1, 0], geometry_type=2)
    flux, keff = k_criticality(materials, geometry, quadrature, solver)
    ref_flux = np.array([0.91063756, 0.67099621, 0.35561622, 0.04678614])
    flux = normalize(flux.flatten(), [1, 0])
    assert np.all(np.isclose(flux, ref_flux, atol=1e-2)), "flux not accurate"
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_pu_2_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=20, bc_x=boundary)
    cells_x = 200 if np.sum(boundary) == 0 else 100
    materials, geometry = benchmarks.PU_2_0(cells_x, boundary, 1)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_pu_2_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.PU_2_0(200, [1, 0], 2)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_u_2_0_slab(boundary):
    # Note: the fast-group fission xs is 0.06192, not 0.06912 (slow group).
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 200 if np.sum(boundary) == 0 else 100
    materials, geometry = benchmarks.U_2_0(cells_x, boundary, 1)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_u_2_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.U_2_0(200, [1, 0], 2)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_ual_2_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 200 if np.sum(boundary) == 0 else 100
    materials, geometry = benchmarks.UAL_2_0(cells_x, boundary, 1)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_ual_2_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.UAL_2_0(200, [1, 0], 2)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.slab1d
@pytest.mark.power_iteration
@pytest.mark.parametrize(("boundary"), [[0, 0], [0, 1], [1, 0]])
def test_urra_2_0_slab(boundary):
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=boundary)
    cells_x = 200 if np.sum(boundary) == 0 else 100
    materials, geometry = benchmarks.URRa_2_0(cells_x, boundary, 1)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_urra_2_0_sphere():
    solver = SolverData()
    quadrature = ants.angular_x(angles=16, bc_x=[1, 0])
    materials, geometry = benchmarks.URRa_2_0(200, [1, 0], 2)
    _, keff = k_criticality(materials, geometry, quadrature, solver)
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.sphere1d
@pytest.mark.power_iteration
@pytest.mark.multigroup1d
def test_sphere_01_power_iteration():
    mat_data, _, geometry, quadrature, solver, _ = problems1d.sphere_01("critical")
    flux, keff = k_criticality(mat_data, geometry, quadrature, solver)
    path = os.path.join(problems1d.PATH, "uranium_sphere_power_iteration_flux.npy")
    ref_flux = np.load(path)
    path = os.path.join(problems1d.PATH, "uranium_sphere_power_iteration_keff.npy")
    ref_keff = np.load(path)
    assert np.isclose(flux, ref_flux).all()
    assert np.fabs(keff - ref_keff) < 1e-05
