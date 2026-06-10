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


External Source Data
--------------------

External sources (``external``) are passed inside a ``SourceData`` object, alongside the
boundary sources described below. The external source can be angular, energy, and time
dependent. Arrays are broadcast automatically along dimensions of size 1, so it is not
necessary to match the full shape if it is an angle-, energy-, or time-dependent source.
The minimum broadcasting shapes for 1D and 2D problems are provided below; these tables
are the authoritative reference for source array shapes.

**External Source Shapes:**

.. list-table::
   :widths: 15 25 25 35
   :header-rows: 1

   * - Ndim
     - Problem Type
     - Minimum broadcast shape
     - Full shape
   * - 1
     - Fixed/Critical
     - ``(I, 1, 1)``
     - ``(I, N, G)``
   * - 1
     - Time-Dependent
     - ``(1, I, 1, 1)``
     - ``(T, I, N, G)``
   * - 2
     - Fixed/Critical
     - ``(I, J, 1, 1)``
     - ``(I, J, N**2, G)``
   * - 2
     - Time-Dependent
     - ``(1, I, J, 1, 1)``
     - ``(T, I, J, N**2, G)``


Boundary Source Data
--------------------

Like the external source, the boundary source terms ``boundary_x`` and ``boundary_y``,
are passed inside the ``SourceData`` object. The boundary source requires at least 2 values
for each boundary term (e.g. [x(0), x(X)]). The boundary sources are able to be space-
(for 2D), angle-, energy-, and time-dependent. The minimum broadcasting shapes are shown
in the table below. Figures showing the locations of the boundary sources are also included.


**Boundary Source Shapes:**

.. list-table::
   :widths: 20 22 25 33
   :header-rows: 1

   * - Ndim/Variable
     - Problem Type
     - Minimum broadcast shape
     - Full shape
   * - 1 / ``boundary_x``
     - Fixed/Critical
     - ``(2, 1, 1)``
     - ``(2, N, G)``
   * - 1 / ``boundary_x``
     - Time-Dependent
     - ``(1, 2, 1, 1)``
     - ``(T, 2, N, G)``
   * - 2 / ``boundary_x``
     - Fixed/Critical
     - ``(2, 1, 1, 1)``
     - ``(2, J, N**2, G)``
   * - 2 / ``boundary_x``
     - Time-Dependent
     - ``(1, 2, 1, 1, 1)``
     - ``(T, 2, J, N**2, G)``
   * - 2 / ``boundary_y``
     - Fixed/Critical
     - ``(2, 1, 1, 1)``
     - ``(2, I, N**2, G)``
   * - 2 / ``boundary_y``
     - Time-Dependent
     - ``(1, 2, 1, 1, 1)``
     - ``(T, 2, I, N**2, G)``


The time axis of a time-dependent source rarely needs to be built by hand. ANTS provides
helpers that expand a steady source over the time grid:

* ``ants.external1d.time_dependence_constant`` — repeat a steady external source across
  all steps (adds the leading ``T`` dimension).
* ``ants.boundary1d.time_dependence_constant``,
  ``ants.boundary1d.time_dependence_decay_01`` / ``_02`` — constant or decaying 1D
  boundary sources.
* ``ants.boundary2d.time_dependence_decay_01`` / ``_02`` / ``_03`` — decaying 2D
  boundary sources.

See :doc:`time-dependent-inputs` for time-stepping details and the initial-flux shapes.


**One-Dimensional Boundary Source**

.. container:: clickable-tikz

    .. tikz::
        :libs: positioning,arrows.meta
        :include: tikz/boundary-1d.tikz
        :xscale: 100

**Two-Dimensional Boundary Source**

.. container:: clickable-tikz

    .. tikz::
        :libs: positioning,arrows.meta
        :include: tikz/boundary-2d.tikz
        :xscale: 100

**Three-Dimensional Boundary Source**

.. container:: clickable-tikz

    .. tikz::
        :libs: positioning,arrows.meta
        :include: tikz/boundary-3d.tikz
        :xscale: 100


Time-Dependent Initial Flux
---------------------------

For initial flux shapes (time-dependent problems), see :doc:`time-dependent-inputs`.


.. raw:: html
    :file: ../functions/tikz-click-handler.html
