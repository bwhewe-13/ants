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

**Discretizations:**

* **Spatial**: Step characteristic method or diamond difference
* **Angular**: Discrete ordinates (product quadrature sets)
* **Energy**: Multigroup formulation
* **Time**: BDF1 (backward Euler), BDF2, Crank-Nicolson (CN), and TR-BDF2


Installation
------------



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
