Getting Started
===============================================

The ANTS package solves the discrete ordinates neutron transport equation in
one-dimensional (slab and sphere), two-dimensional (rectangular), and
three-dimensional (rectangular) geometry. Three-dimensional problems are currently
limited to fixed source problems.

**Problem Types Supported:**

* **Fixed source problems**: Given an external source and/or boundary source, compute the steady-state flux
* **Time-dependent problems**: Solve transient problems with temporal evolution
* **Criticality (k-eigenvalue) problems**: Find the effective multiplication factor and fundamental eigenmode

**Acceleration Methods:**

* **Hybrid methods**: Collision-based decomposition for time-dependent problems
* **DMD (Dynamic Mode Decomposition)**: Reduced-order acceleration for multigroup problems
* **Artificial scattering (as-SN)**: Ray effect mitigation in two and three dimensions
* **OpenMP**: Parallel sweeps over angles, energy groups, or both

**Discretizations:**

* **Spatial**: Step method, diamond difference, or step characteristic
* **Angular**: Discrete ordinates (product quadrature sets)
* **Energy**: Multigroup formulation
* **Time**: BDF1 (backward Euler), BDF2, Crank-Nicolson (CN), and TR-BDF2


Installation
------------

ANTS is on PyPI as ``ants-sn`` and is imported as ``ants``:

.. code-block:: bash

   python -m pip install ants-sn

Wheels are built for CPython 3.11 to 3.14 on Linux, macOS 15 or newer, and
Windows, so no compiler is needed. Cross sections from HDF5 libraries need the
``library`` extra:

.. code-block:: bash

   python -m pip install "ants-sn[library]"

To build from source, which needs a C++ compiler and OpenMP (``brew install
libomp`` on macOS), clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/bwhewe-13/ants.git
   cd ants
   python -m pip install -e ".[dev]"


Required Input Data
-------------------

To solve a neutron transport problem, you need:

1. **Cross-sectional data** for each material:
   - Total cross section (:math:`\sigma^{\mathrm{t}}`)
   - Scattering matrix (:math:`\sigma^{\mathrm{s}}`)
   - Fission matrix (:math:`\chi \, \nu \, \sigma^{\mathrm{f}}`)

2. **Geometry specification**: spatial cells, material locations, dimensions

3. **Angular quadrature**: discrete directions and weights

4. **Source terms**: external source and/or boundary conditions

5. **Parameters**: problem configuration (geometry type, discretization, time stepping, etc.)


Current Limitations
-------------------

* No beta decay or delayed neutrons
* No temperature feedback or Doppler effects


Notation
--------

Throughout the documentation, the following notation is used for problem dimensions:

* :math:`I` : Number of spatial cells in the x direction
* :math:`J` : Number of spatial cells in the y direction
* :math:`K` : Number of spatial cells in the z direction (3D)
* :math:`N` : Number of discrete angles (1D), :math:`N^2` total angles (2D), or the
  total number of ordinates (3D, :math:`2 N^2` for ``angular_xyz(N)``)
* :math:`G` : Number of energy groups
* :math:`T` : Total elapsed time
* :math:`M` : Number of different materials
