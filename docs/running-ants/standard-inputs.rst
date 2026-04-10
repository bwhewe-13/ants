Standard Inputs
===============================================

All problem types require standard input data: cross-sectional data, material locations,
spatial discretization, angular quadrature, and source/boundary conditions.
These are passed to solver functions as typed dataclass objects — see :doc:`extra`
for a complete reference and worked examples.


Cross-Sectional Data
---------------------

The neutron cross-sectional data required are:

* **Total cross section**: :math:`\sigma^{\mathrm{t}}` — shape :math:`(M \times G)`
* **Scattering matrix**: :math:`\sigma^{\mathrm{s}}` — shape :math:`(M \times G \times G)`
* **Fission matrix**: :math:`\chi \, \nu \, \sigma^{\mathrm{f}}` — shape :math:`(M \times G \times G)`

where :math:`M` is the number of materials and :math:`G` is the number of energy groups.

The ANTS package provides a utility function to retrieve real cross-sectional data:

.. autofunction:: ants.materials


Material Locations
-------------------

Material locations are specified using a ``medium_map`` array that identifies which
material occupies each spatial cell:

* **1D problems**: ``medium_map`` is shape :math:`(I,)` with integer values in [0, M-1]
* **2D problems**: ``medium_map`` is shape :math:`(I \times J,)` with integer values in [0, M-1]

For structured spatial layouts, use these utility functions:

.. autofunction:: ants.spatial1d

.. autofunction:: ants.spatial2d


Spatial Data
------------

Spatial discretization requires cell widths in each direction:

* **1D problems**: ``delta_x`` — array of cell widths, shape :math:`(I,)`
* **2D problems**: ``delta_x`` and ``delta_y`` — shapes :math:`(I,)` and :math:`(J,)` respectively


Angular Data
------------

Angular quadrature data (discrete directions and weights) is generated automatically
and returned as a ``QuadratureData`` object:

.. autofunction:: ants.angular_x

.. autofunction:: ants.angular_xy


Source and Boundary Data
------------------------

External sources and boundary conditions are passed inside a ``SourceData`` object.
Arrays are broadcast automatically along dimensions of size 1, so it is not necessary
to match the full ``(I, N, G)`` shape — a shape of ``(I, 1, 1)`` is sufficient for
a spatially varying, angle- and energy-independent source.

**One-Dimensional shapes:**

.. list-table::
   :header-rows: 1

   * - Field
     - Minimum broadcast shape
     - Full shape
   * - ``external`` (fixed/critical)
     - ``(I, 1, 1)``
     - ``(I, N, G)``
   * - ``external`` (time-dependent)
     - ``(1, I, 1, 1)``
     - ``(T, I, N, G)``
   * - ``boundary_x`` (fixed/critical)
     - ``(2, 1, 1)``
     - ``(2, N, G)``
   * - ``boundary_x`` (time-dependent)
     - ``(T, 2, 1, 1)``
     - ``(T, 2, N, G)``

**Two-Dimensional shapes:**

.. list-table::
   :header-rows: 1

   * - Field
     - Minimum broadcast shape
     - Full shape
   * - ``external`` (fixed/critical)
     - ``(I, J, 1, 1)``
     - ``(I, J, N^2, G)``
   * - ``external`` (time-dependent)
     - ``(1, I, J, 1, 1)``
     - ``(T, I, J, N^2, G)``
   * - ``boundary_x``
     - ``(T, 2, 1, 1, 1)``
     - ``(T, 2, J, N^2, G)``
   * - ``boundary_y``
     - ``(T, 2, 1, 1, 1)``
     - ``(T, 2, I, N^2, G)``

For initial flux shapes (time-dependent problems), see :doc:`time-dependent-inputs`.
