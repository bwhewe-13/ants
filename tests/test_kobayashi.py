########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for the Kobayashi 3D benchmark problems and reference fluxes
#
########################################################################

import numpy as np
import pytest

from ants.benchmarks import kobayashi
from ants.datatypes import SolverData
from ants.fixed3d import fixed_source

# Void volume of each problem (cm^3), not counting the source region
VOID_VOLUMES = {1: 50.0**3 - 10.0**3, 2: 10.0 * 90.0 * 10.0, 3: 15000.0}


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("problem", [1, 2, 3])
def test_region_volumes(problem):
    materials, sources, geometry, quadrature, solver = kobayashi.build_problem(
        problem, cell_width=10 / 3, angles=2
    )
    volume = geometry.delta_x[0] * geometry.delta_y[0] * geometry.delta_z[0]
    regions = np.bincount(geometry.medium_map.ravel(), minlength=3) * volume
    total = np.prod(kobayashi.LENGTHS[problem])
    assert np.isclose(regions[0], 1000.0)
    assert np.isclose(regions[1], VOID_VOLUMES[problem])
    assert np.isclose(regions[2], total - 1000.0 - VOID_VOLUMES[problem])
    # Unit source only in the source region
    assert np.isclose(sources.external.sum() * volume, 1000.0)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("cell_width", [1.0, 2.5, 3.0])
def test_cell_width(cell_width):
    with pytest.raises(ValueError, match="odd number"):
        kobayashi.build_problem(1, cell_width=cell_width)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("scattering", [False, True])
def test_reference_shared_points(scattering):
    # Points in more than one table of a problem have the same flux
    for problem in "123":
        values = {}
        for case in kobayashi.REFERENCE:
            if case[0] != problem:
                continue
            points, flux, _ = kobayashi.reference(case, scattering)
            for point, value in zip(map(tuple, points), flux):
                assert values.setdefault(point, value) == value


@pytest.mark.smoke
@pytest.mark.slab3d
def test_reference_points_at_centers():
    # Every reference point is a cell center for the allowed widths
    for cell_width in [10.0, 10 / 3, 2.0]:
        for case in kobayashi.REFERENCE:
            points, _, _ = kobayashi.reference(case)
            assert np.allclose((points / cell_width - 0.5) % 1.0, 0.0)


@pytest.mark.slab3d
@pytest.mark.parametrize("problem", [1, 2, 3])
@pytest.mark.parametrize(
    "scattering", [pytest.param(False, marks=pytest.mark.smoke), True]
)
def test_near_source(problem, scattering):
    # Close to the source the ray effects are small, so S8 with 2 cm cells
    # is within a few percent of the reference flux
    inputs = kobayashi.build_problem(
        problem, scattering, cell_width=2.0, angles=8, solver=SolverData(num_threads=4)
    )
    flux = fixed_source(*inputs)[..., 0]
    points, reference, _ = kobayashi.reference(f"{problem}A", scattering)
    approx = kobayashi.flux_at(flux, points[:2], inputs[2])
    assert np.allclose(approx, reference[:2], rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_flux_at_faces():
    # Points on the domain faces take the cell inside the domain
    geometry = kobayashi.build_problem(1, cell_width=10 / 3, angles=2)[2]
    cells = geometry.medium_map.shape
    flux = np.arange(np.prod(cells), dtype=float).reshape(cells)
    points = np.array([[0.0, 5.0, 5.0], [100.0, 100.0, 100.0], [5.0, 0.0, 100.0]])
    approx = kobayashi.flux_at(flux, points, geometry)
    assert np.array_equal(approx, flux[[0, -1, 1], [1, -1, 0], [1, -1, -1]])
    with pytest.raises(ValueError, match="inside the domain"):
        kobayashi.flux_at(flux, np.array([[-1.0, 5.0, 5.0]]), geometry)
