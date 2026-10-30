########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# ICSBEP criticality benchmarks
#
########################################################################

"""ICSBEP criticality benchmark specifications and problem builders.

Each ``Benchmark`` holds the benchmark-model specification from the
International Handbook of Evaluated Criticality Safety Benchmark
Experiments: region radii, the material in each region, nuclide number
densities (atoms/barn-cm), temperature, and the benchmark k-effective.
``build_problem`` turns a benchmark into ANTS inputs for any number of
cells, angles, and energy groups, and ``run_suite`` runs a set of them.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

import ants
from ants.datatypes import GeometryData, SolverData
from ants.materials import _load_library, parse_composition

# Region material without any nuclides
VOID = "void"


@dataclass
class Benchmark:
    """Specification of an ICSBEP benchmark model.

    Attributes
    ----------
    name : str
        ICSBEP identifier, e.g. ``"PU-MET-FAST-001"``.
    radii : list[float]
        Outer radius of each region (cm), increasing.
    regions : list[str]
        Material name of each region (``"void"`` for empty regions).
    materials : dict[str, str or dict]
        Nuclide number densities (atoms/barn-cm) of each material, in any
        form accepted by ``ants.materials.parse_composition``.
    temperature : float
        Benchmark temperature (K).
    keff : float
        Benchmark-model k-effective.
    keff_unc : float
        Benchmark-model k-effective uncertainty.
    reference : str
        Handbook revision and input deck the data were taken from.
    geometry : str
        Benchmark geometry (only ``"sphere"`` is supported).
    """

    name: str
    radii: List[float]
    regions: List[str]
    materials: Dict[str, Union[str, dict]]
    temperature: float
    keff: float
    keff_unc: float
    reference: str = ""
    geometry: str = "sphere"

    def compositions(self):
        """Return the normalized number densities of each material."""
        compositions = {VOID: {}}
        for name, composition in self.materials.items():
            compositions[name] = parse_composition(composition)
        return compositions


BENCHMARKS = {}


def register(benchmark):
    """Add a benchmark to the ``BENCHMARKS`` registry."""
    BENCHMARKS[benchmark.name.upper()] = benchmark
    return benchmark


def get_benchmark(name):
    """Return a registered ICSBEP benchmark by name (case-insensitive)."""
    _load_data()
    key = str(name).upper()
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}, available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[key]


def list_benchmarks(family=None):
    """List registered benchmarks, optionally containing ``family``."""
    _load_data()
    names = sorted(BENCHMARKS)
    if family is not None:
        names = [nn for nn in names if family.upper() in nn]
    return names


def _load_data():
    # Benchmark data modules register themselves on import
    from ants.benchmarks import icsbep_data  # noqa: F401


########################################################################
# Problem Setup
########################################################################


def _region_cells(radii, cells):
    # Cells per region proportional to thickness (at least one each)
    thickness = np.diff(np.concatenate(([0.0], radii)))
    if np.ndim(cells) > 0:
        assert len(cells) == len(radii), "Need the number of cells per region"
        return np.asarray(cells, dtype=np.int32)
    assert cells >= len(radii), f"Need at least {len(radii)} cells"
    exact = (cells - len(radii)) * thickness / thickness.sum()
    region_cells = np.floor(exact).astype(np.int32) + 1
    # Largest remainder for the leftover cells
    leftover = cells - region_cells.sum()
    region_cells[np.argsort(exact - np.floor(exact))[::-1][:leftover]] += 1
    return region_cells


def build_problem(benchmark, library, cells, angles, groups=None, temperature=None):
    """Create the ANTS k-eigenvalue problem for an ICSBEP benchmark.

    Parameters
    ----------
    benchmark : str or Benchmark
        Benchmark name or specification.
    library : str or Library
        HDF5 cross section library (or path to it).
    cells : int or list[int]
        Total number of spatial cells, divided between the regions in
        proportion to their thickness, or the number of cells per region.
        Region boundaries always fall on cell edges.
    angles : int
        Number of discrete ordinates.
    groups : int, optional
        Collapse the library to fewer energy groups (energy bin width
        weighting). Defaults to the library groups.
    temperature : float, optional
        Library temperature (K); defaults to the benchmark temperature.

    Returns
    -------
    materials : MaterialData
    geometry : GeometryData
    quadrature : QuadratureData
    solver : SolverData
    """
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    assert benchmark.geometry == "sphere", "Only spherical benchmarks are supported"
    library = _load_library(library)
    temperature = benchmark.temperature if temperature is None else temperature

    # Materials in order of first appearance
    material_names = list(dict.fromkeys(benchmark.regions))
    compositions = benchmark.compositions()
    mat_data = library.materials(
        [compositions[name] for name in material_names], temperature, groups
    )

    # Spatial
    region_cells = _region_cells(benchmark.radii, cells)
    edges_x = [0.0]
    medium_map = []
    for radius, region_cell, region in zip(
        benchmark.radii, region_cells, benchmark.regions
    ):
        edges_x.extend(np.linspace(edges_x[-1], radius, region_cell + 1)[1:])
        medium_map.extend([material_names.index(region)] * region_cell)

    geometry = GeometryData(
        medium_map=np.array(medium_map, dtype=np.int32),
        delta_x=np.diff(edges_x),
        bc_x=[1, 0],
        geometry=2,
    )
    quadrature = ants.angular_x(angles, bc_x=[1, 0])
    solver = SolverData()
    return mat_data, geometry, quadrature, solver


def run_suite(
    names, library, cells=(100,), angles=(8,), groups=(None,), temperature=None
):
    """Run k-eigenvalue calculations over a suite of ICSBEP benchmarks.

    Parameters
    ----------
    names : list[str]
        Benchmark names (see ``list_benchmarks``).
    library : str or Library
        HDF5 cross section library (or path to it).
    cells, angles, groups : list[int]
        Values to run for every benchmark (all combinations). ``None``
        in ``groups`` uses the library groups.
    temperature : float, optional
        Library temperature (K); defaults to each benchmark temperature.

    Returns
    -------
    list[dict]
        One row per run with keys ``name``, ``groups``, ``angles``,
        ``cells``, ``keff``, ``reference``, ``uncertainty``, and
        ``diff_pcm`` (calculated minus benchmark k, in pcm).
    """
    from ants.critical1d import k_criticality

    library = _load_library(library)
    results = []
    for name, ngroups, nangles, ncells in itertools.product(
        names, groups, angles, cells
    ):
        benchmark = get_benchmark(name)
        materials, geometry, quadrature, solver = build_problem(
            benchmark, library, ncells, nangles, ngroups, temperature
        )
        _, keff = k_criticality(materials, geometry, quadrature, solver)
        results.append(
            {
                "name": benchmark.name,
                "groups": materials.total.shape[1],
                "angles": nangles,
                "cells": int(np.sum(_region_cells(benchmark.radii, ncells))),
                "keff": keff,
                "reference": benchmark.keff,
                "uncertainty": benchmark.keff_unc,
                "diff_pcm": (keff - benchmark.keff) * 1e5,
            }
        )
    return results
