<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bwhewe-13/ants/main/docs/_static/logo-dark.svg">
  <img alt="ants" src="https://raw.githubusercontent.com/bwhewe-13/ants/main/docs/_static/logo.svg" width="202">
</picture>

[![PyPI](https://img.shields.io/pypi/v/ants-sn)](https://pypi.org/project/ants-sn/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ants-sn)](https://pypi.org/project/ants-sn/)
[![Style](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/ci.yml?label=Style)](https://github.com/bwhewe-13/ants/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/tests.yml?label=Tests)](https://github.com/bwhewe-13/ants/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/bwhewe-13/ants/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/ants)
[![Docs](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/docs.yml?label=Docs)](https://www.ben-whewell.com/ants/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/465198427.svg)](https://doi.org/10.5281/zenodo.24066032)

A Neutron Transport Solution (ANTS) solves the neutron transport equation with the discrete ordinates (S<sub>N</sub>) method. It is written in Cython and parallelized over angles and energy groups with OpenMP. The solutions are checked with manufactured solutions, benchmark problems, and the method of nearby problems.

📖 **Documentation:** [www.ben-whewell.com/ants](https://www.ben-whewell.com/ants/)

&nbsp;

## Features

|              | 1D | 2D | 3D |
|--------------|----|----|----|
| Geometry     | Slab, sphere | Slab | Slab |
| Problems     | Fixed source, criticality, time-dependent | Fixed source, criticality, time-dependent | Fixed source |
| Spatial      | Step, diamond difference, step characteristic | Step, diamond difference, step characteristic | Step, diamond difference, step characteristic |
| Time         | Backward Euler, Crank-Nicolson, BDF2, TR-BDF2 | Backward Euler, Crank-Nicolson, BDF2, TR-BDF2 | |
| Scattering   | Isotropic, anisotropic | Isotropic, anisotropic | Isotropic, anisotropic |
| Acceleration | DMD, hybrid | DMD, hybrid, as-S<sub>N</sub> | DMD, as-S<sub>N</sub> |
| Verification | Manufactured solutions, Sood benchmarks, ICSBEP, nearby problems | Manufactured solutions, Sood benchmarks, nearby problems | Manufactured solutions, Kobayashi benchmarks |

DMD is dynamic mode decomposition, hybrid is the collision-based hybrid method, and
as-S<sub>N</sub> is artificial scattering to reduce ray effects. Planned features are in
[TODO.md](TODO.md).

&nbsp;

## Installation

### Requirements
- Python 3.11+
- For source builds, a C/C++ compiler and OpenMP (included on Linux, `brew install libomp`
  on macOS)

### Install from PyPI
```bash
python -m pip install ants-sn
```
The package is imported as `ants`. Wheels are built for Linux, macOS 15 or newer, and
Windows, so no compiler is needed.

### Install from source
```bash
python -m pip install --upgrade pip
python -m pip install .
```

### Install for development (tests + style)
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

### Cross section libraries
Cross sections are built from HDF5 multigroup libraries of microscopic cross sections
(layout in `ants/library.py`), which requires the `library` extra:
```bash
python -m pip install "ants-sn[library]"
```
The standard NJOY and Serpent energy group structures (`ants.list_structures()`) and
ICSBEP benchmark specifications (`ants.list_benchmarks()`) are included with ANTS.

## Running ANTS

A one group fixed source problem in a 10 cm slab with vacuum boundaries:

```python
import numpy as np

import ants
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
from ants.fixed1d import fixed_source

cells_x = 100
edges_x = np.linspace(0.0, 10.0, cells_x + 1)

materials = MaterialData(
    total=np.array([[1.0]]),
    scatter=np.array([[[0.5]]]),
    fission=np.zeros((1, 1, 1)),
)
sources = SourceData(
    external=np.ones((cells_x, 1, 1)),
    boundary_x=np.zeros((2, 1, 1)),
)
geometry = GeometryData(
    medium_map=np.zeros(cells_x, dtype=np.int32),
    delta_x=np.diff(edges_x),
)
quadrature = ants.angular_x(8)

flux = fixed_source(materials, sources, geometry, quadrature, SolverData())
print(flux.shape)  # (cells_x, groups)
```

The [`examples/`](examples) folder of the repository has more problems, including
criticality, time-dependent, hybrid, and 2D and 3D problems. Run them from the
repository root:

```bash
python examples/fixed_source_1d_reeds.py
python examples/critical_1d_uranium_slab.py
python examples/icsbep_suite.py path/to/library.h5
```

## Testing, Coverage, and Docs

### Run tests
```bash
pytest tests
```
or, in parallel with `make`:
```bash
make test
```

### Run smoke tests (quick check)
```bash
pytest tests -m smoke
```

### Generate coverage locally
```bash
pytest tests --cov=ants --cov-report=term --cov-report=xml
```

### Run formatting and lint checks locally
```bash
pre-commit run --all-files
```

### Documentation
Building the [documentation](https://www.ben-whewell.com/ants/) locally needs [pandoc](https://pandoc.org/) and a LaTeX install with TikZ:
```bash
python -m pip install -e ".[docs]"
make docs
```

&nbsp;

## Citing ANTS
If you use ANTS in your work, please cite it with the DOI
[10.5281/zenodo.24066032](https://doi.org/10.5281/zenodo.24066032), which always points
to the latest version. Each release also has its own DOI on Zenodo. The **Cite this
repository** button on GitHub gives the citation in APA and BibTeX.

The changes in each version are in the [changelog](CHANGELOG.md).
