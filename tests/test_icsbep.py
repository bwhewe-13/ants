########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for the ICSBEP benchmark specifications and problem builder.
# Set ANTS_XS_LIBRARY to an HDF5 cross section library to run the benchmarks.
#
########################################################################

import os
import re

import numpy as np
import pytest

import ants
from ants.benchmarks.icsbep import VOID, _region_cells

LIBRARY_PATH = os.environ.get("ANTS_XS_LIBRARY")


@pytest.mark.smoke
@pytest.mark.parametrize("name", ants.list_benchmarks())
def test_benchmark_specification(name):
    benchmark = ants.get_benchmark(name)
    assert len(benchmark.radii) == len(benchmark.regions)
    assert np.all(np.diff(benchmark.radii) > 0)
    compositions = benchmark.compositions()
    for region in benchmark.regions:
        assert region in compositions
        assert (region == VOID) or (len(compositions[region]) > 0)
    for composition in compositions.values():
        for nuclide, density in composition.items():
            assert re.fullmatch(r"[A-Z][a-z]?-\d{1,3}m?", nuclide)
            assert 0 < density < 0.2
    assert 0.9 < benchmark.keff < 1.1
    assert 0 < benchmark.keff_unc < 0.01
    assert 250 < benchmark.temperature < 400


@pytest.mark.parametrize("cells", [10, 37, 100, 1000])
@pytest.mark.parametrize("name", ants.list_benchmarks())
def test_region_cells(name, cells):
    benchmark = ants.get_benchmark(name)
    if cells < len(benchmark.radii):
        return
    region_cells = _region_cells(benchmark.radii, cells)
    assert region_cells.sum() == cells
    assert np.all(region_cells >= 1)


def test_benchmark_lookup():
    assert ants.get_benchmark("pu-met-fast-001").name == "PU-MET-FAST-001"
    assert "HEU-MET-FAST-001" in ants.list_benchmarks("heu")
    with pytest.raises(KeyError):
        ants.get_benchmark("PU-MET-FAST-999")


@pytest.mark.skipif(LIBRARY_PATH is None, reason="Set ANTS_XS_LIBRARY")
@pytest.mark.parametrize("name", ants.list_benchmarks())
def test_build_problem(name):
    benchmark = ants.get_benchmark(name)
    materials, geometry, quadrature, _ = ants.build_problem(
        benchmark, LIBRARY_PATH, cells=50, angles=4
    )
    # Region boundaries fall on cell edges
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    for radius in benchmark.radii:
        assert np.min(np.abs(edges_x - radius)) < 1e-10
    assert geometry.medium_map.shape == (50,)
    assert materials.total.shape[0] == len(set(benchmark.regions))
    assert quadrature.angle_x.shape == (4,)


@pytest.mark.skipif(LIBRARY_PATH is None, reason="Set ANTS_XS_LIBRARY")
@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_run_suite():
    names = ["HEU-MET-FAST-001", "PU-MET-FAST-001"]
    results = ants.run_suite(names, LIBRARY_PATH, cells=[50], angles=[4, 8])
    assert len(results) == 4
    for row in results:
        # P0 libraries without a transport correction overestimate keff
        assert 0.95 < row["keff"] < 1.2


@pytest.mark.skipif(LIBRARY_PATH is None, reason="Set ANTS_XS_LIBRARY")
@pytest.mark.sphere1d
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_run_suite_legendre():
    # P3 scattering moves keff closer to the benchmark than P0 scattering
    names = ["HEU-MET-FAST-001", "PU-MET-FAST-001"]
    try:
        p3 = ants.run_suite(names, LIBRARY_PATH, cells=[50], angles=[8], legendre=3)
    except ValueError as error:
        if "Legendre moments" not in str(error):
            raise
        pytest.skip(f"Library has too few Legendre moments: {error}")
    p0 = ants.run_suite(names, LIBRARY_PATH, cells=[50], angles=[8])
    for row_p0, row_p3 in zip(p0, p3):
        assert row_p3["legendre"] == 3
        assert abs(row_p3["diff_pcm"]) < abs(row_p0["diff_pcm"])
