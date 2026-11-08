<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bwhewe-13/ants/main/docs/_static/logo-dark.svg">
  <img alt="ants" src="https://raw.githubusercontent.com/bwhewe-13/ants/main/docs/_static/logo.svg" width="202">
</picture>

[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/ants-sn/)
[![Style](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/ci.yml?label=Style)](https://github.com/bwhewe-13/ants/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/tests.yml?label=Tests)](https://github.com/bwhewe-13/ants/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/bwhewe-13/ants/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/ants)
[![Docs](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/docs.yml?label=Docs)](https://bwhewe-13.github.io/ants/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs, and fixed source problems of three dimensional slabs, using the discrete ordinates method and written in Cython.

There are a number of different acceleration methods used including a collision-based hybrid method, machine learning models to predict matrix-vector multiplication, dynamic mode decomposition, and synthetic diffusion acceleration (DSA).

There are also verification procedures to ensure both the code and solutions are correct. For code verification, manufactured solutions are used for one-, two-, and three-dimensional slab problems to ensure proper discretization. Solution verification uses the method of nearby problems, which uses one spatial grid.

📖 **Documentation:** The full ANTS documentation is hosted online at [bwhewe-13.github.io/ants](https://bwhewe-13.github.io/ants/).

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
The package is imported as `ants`. Wheels are built for Linux, macOS, and Windows, so
no compiler is needed.

### Install from source
```bash
python -m pip install --upgrade pip
python -m pip install .
```

### Install for development (tests + style)
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Cross section libraries
Cross sections are built from HDF5 multigroup libraries of microscopic cross sections
(layout in `ants/library.py`), which requires the `library` extra:
```bash
python -m pip install "ants-sn[library]"
```
The standard NJOY and Serpent energy group structures (`ants.list_structures()`) and
ICSBEP benchmark specifications (`ants.list_benchmarks()`) are included with ANTS.

### Enable pre-commit hooks
```bash
pre-commit install
```

## Running ANTS

Run one of the provided examples from the repository root:

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
The full documentation is hosted online at [bwhewe-13.github.io/ants](https://bwhewe-13.github.io/ants/). To build it locally:
```bash
python -m pip install sphinx sphinxcontrib-bibtex sphinxcontrib-tikz
make -C docs html
```

&nbsp;

## One Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9744; DJINN           |
| &#9745; Step Characteristic   | &#9745; Crank-Nicolson | &#9745; DMD               | &#9745; DMD             |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             | &#9744; Davidson Method |

&nbsp;

## Two Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9745; DMD             |
| &#9745; Step Characteristic   | &#9745; Crank-Nicolson | &#9745; DMD               | &#9744; Davidson Method |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             |                         |

&nbsp;

## Three Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9744; Backward Euler     | &#9745; Source Iteration  | &#9744; Power Iteration |
| &#9745; Diamond Difference    | &#9744; BDF2           | &#9744; DSA               | &#9744; DMD             |
| &#9745; Step Characteristic   | &#9744; Crank-Nicolson | &#9745; DMD               | &#9744; Davidson Method |
| &#9744; Discontinuous Galerkin| &#9744; TR - BDF2      | &#9744; GMRES             |                         |

&nbsp;

## Code and Solution Verification
- &#9744; Spatial Method of Manufactured Solutions (1D/2D/3D)
    - &#9745; &#9745; &#9745; Step Method
    - &#9745; &#9745; &#9745; Diamond Difference
    - &#9745; &#9745; &#9745; Step Characteristics
    - &#9744; &#9744; &#9744; Discontinuous Galerkin
- &#9745; Temporal Method of Manufactured Solutions
    - &#9745; &#9745; Backward Euler
    - &#9745; &#9745; BDF2
    - &#9745; &#9745; Crank-Nicolson
    - &#9745; &#9745; TR - BDF2
- &#9745; &#9745; Criticality Benchmarks \
    (Analytical Benchmark Test Set for Criticality Code Verification, Sood et al)
- &#9745; ICSBEP Fast Metal Benchmarks (1D)
- &#9745; Kobayashi Benchmarks (3D) \
    (3-D Radiation Transport Benchmark Problems and Results for Simple Geometries with Void Regions, Kobayashi et al)
- &#9745; &#9745; Method of Nearby Problems (MNP)

&nbsp;
