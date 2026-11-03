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
Energy groups are ordered from fastest to slowest (group 0 is the highest energy), and the
scattering and fission matrices are indexed ``[material, group out, group in]``.

For anisotropic scattering, the scattering matrix has the
Legendre moments :math:`\ell = 0, \ldots, L` on a last axis, shape
:math:`(M \times G \times G \times (L + 1))`, indexed
``[material, group out, group in, moment]``. The moments do not include the
:math:`(2\ell + 1)` factor (see :ref:`anisotropic scattering <sec-nte-anisotropic>`).
All one- and two-dimensional solvers select anisotropic scattering from the shape
of the scattering matrix. Two-dimensional problems use the same Legendre moments,
which are expanded into spherical harmonics internally.

.. code-block:: python

    # One group P2 scattering
    scatter = np.array([[[[0.733333, 0.2, 0.075]]]])


Energy Grids
~~~~~~~~~~~~

The NJOY, Serpent, CASMO, SCALE, and other standard group structures are included with ANTS.
All energy bounds are in eV and descending order.

.. code-block:: python

    edges_g = ants.get_structure("casmo_70")
    ants.list_structures("njoy")

    # Bounds and coarsening indices (here 70 fine groups to 10 coarse groups)
    edges_g, edges_gidx_fine, edges_gidx_coarse = ants.energy_grid("casmo_70", 70, 10)

.. autofunction:: ants.get_structure

.. autofunction:: ants.energy_grid


Cross Section Libraries
~~~~~~~~~~~~~~~~~~~~~~~

Macroscopic cross sections are built from number densities (atoms/barn-cm) and a
HDF5 multigroup library (requires ``pip install ants[library]``). Compositions
can be copied directly from MCNP or PARTISN input decks, and the common materials
(enriched uranium, uranium hydride, enriched plutonium, ...) are available by name.

.. code-block:: python

    library = ants.Library("endf_xmas172.h5")

    # MCNP material card or PARTISN material block
    fuel = ants.create_material("m1 94239.80c 3.7047E-02 31069.80c 8.3603E-04", library)

    # Named materials and compositions
    mat_data = ants.materials(172, ["uranium-%20%", "stainless-steel-440",
                                    {"H-1": 0.0668, "O-16": 0.0334}], library=library)

    # P3 anisotropic scattering, shape (materials, groups, groups, 4)
    fuel = ants.create_material("m1 94239.80c 3.7047E-02", library, legendre=3)

.. autofunction:: ants.materials

.. autofunction:: ants.create_material

.. autofunction:: ants.parse_composition

.. autoclass:: ants.library.Library
    :members: microscopic, macroscopic, materials


ICSBEP Benchmarks
~~~~~~~~~~~~~~~~~

The geometry, material layout, and number densities of ICSBEP criticality benchmarks
(``ants.list_benchmarks()``) are included, so the same benchmark can be run with any
number of cells, angles, and energy groups. The data are from the *International
Handbook of Evaluated Criticality Safety Benchmark Experiments*, NEA/NSC/DOC(95)03
(OECD-NEA, September 2015); the full evaluations are available from the NEA, and
publications using these benchmarks should cite the Handbook.

.. code-block:: python

    materials, geometry, quadrature, solver = ants.build_problem(
        "PU-MET-FAST-001", library, cells=200, angles=8
    )
    results = ants.run_suite(ants.list_benchmarks("MET-FAST"), library,
                             cells=[100, 400], angles=[4, 8, 16])

.. autofunction:: ants.build_problem

.. autofunction:: ants.run_suite


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


**Compact Sources:**

Broadcasting handles a dimension where the source is the same everywhere, but not
one where it is mostly zero. A source in one energy group of an 87-group problem
stores 86 groups of zeros. ``ants.datatypes.CompactSource`` stores each distinct
spatial profile once (``values``, shape ``(*spatial, S)``) and an integer
``index`` of shape ``(N, G)`` (``(N**2, G)`` in 2D) giving the profile of each
angle and group, with ``-1`` for a zero source. The spatial shape is ``(I,)`` or
``(I, J)`` for external sources and ``(2,)``, ``(2, J)``, or ``(2, I)`` for
boundary sources. Index dimensions of size 1 are broadcast. Any of
``external``, ``boundary_x``, and ``boundary_y`` can be a ``CompactSource``.

.. code-block:: python

    from ants.datatypes import CompactSource

    # Isotropic source in cells 10-20 of group 0, zero in the other groups
    values = np.zeros((cells_x, 1))
    values[10:20, 0] = 1.0
    index = np.full((1, groups), -1)
    index[0, 0] = 0
    external = CompactSource(values, index)

    # Or from an existing dense array
    boundary_x = CompactSource.from_dense(ants.boundary1d.deuterium_tritium(0, edges_g))

The steady-state sweeps read a compact external source through the index, so it is
never expanded. Compact boundary sources are expanded to the (surface sized) dense
array, and the nearby problems use dense sources.


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
