
# A Neutron Transport Solution (ANTS)

[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/ants/)
[![Tests](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/ci.yml?label=Tests)](https://github.com/bwhewe-13/ants/actions/workflows/ci.yml?query=job%3Atests)
[![Coverage](https://codecov.io/gh/bwhewe-13/ants/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/ants)
[![Docs](https://img.shields.io/github/actions/workflow/status/bwhewe-13/ants/ci.yml?label=Docs)](https://github.com/bwhewe-13/ants/actions/workflows/ci.yml?query=job%3Adocs)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs using the discrete ordinates method and written in Cython.

There are a number of different acceleration methods used including a collision-based hybrid method, machine learning models to predict matrix-vector multiplication, dynamic mode decomposition, and synthetic diffusion acceleration (DSA).

There are also verification procedures to ensure both the code and solutions are correct. For code verification, manufactured solutions are used for one- and two-dimenisonal slab problems to ensure proper discretization. Solution verification uses the method of nearby problems, which uses one spatial grid.

&nbsp;

## Installation

### Requirements
- Python 3.9+
- A C/C++ compiler
- OpenMP runtime support (Linux typically available by default)

### Install for use
```bash
python -m pip install --upgrade pip
python -m pip install .
```

### Install for development (tests + style)
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Enable pre-commit hooks
```bash
pre-commit install
```

## Running ANTS

Run one of the provided examples from the repository root:

```bash
python examples/fixed_source_1d_reeds.py
python examples/critical_1d_uranium_slab.py
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

### Build docs locally
```bash
python -m pip install sphinx sphinxcontrib-bibtex
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
| &#9744; Step Characteristic   | &#9745; Crank-Nicolson | &#9745; DMD               | &#9744; Davidson Method |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             |                         |

&nbsp;

## Code and Solution Verification
- &#9744; Spatial Method of Manufactured Solutions (1D/2D)
    - &#9745; &#9745; Step Method
    - &#9745; &#9745; Diamond Difference
    - &#9745; &#9744; Step Characteristics
    - &#9744; &#9744; Discontinuous Galerkin
- &#9745; Temporal Method of Manufactured Solutions
    - &#9745; &#9745; Backward Euler
    - &#9745; &#9745; BDF2
    - &#9745; &#9745; Crank-Nicolson
    - &#9745; &#9745; TR - BDF2
- &#9745; &#9745; Criticality Benchmarks \
    (Analytical Benchmark Test Set for Criticality Code Verification, Sood et al)
- &#9745; &#9745; Method of Nearby Problems (MNP)

&nbsp;

## Features To Add
- &#9744; Ray Effect Corrections (2D)
- &#9744; Adjoint Equations (1D/2D)
- &#9744; Acceleration Techniques (DSA, GMRES, CMFD)
- &#9744; Optimize DMD Implementation
- &#9744; Banded Triangular Meshes (2D)
