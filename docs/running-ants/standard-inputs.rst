Standard Inputs
===============================================

All problem types require standard input data: cross-sectional data, material locations,
spatial discretization, angular quadrature, and source/boundary conditions.


Cross-Sectional Data
---------------------

The neutron cross-sectional data required are:

* **Total cross section**: :math:`\sigma^{\mathrm{t}}` - vector of size :math:`(M \times G)`
* **Scattering matrix**: :math:`\sigma^{\mathrm{s}}` - matrix of size :math:`(M \times G \times G)`
* **Fission matrix**: :math:`\chi \, \nu \, \sigma^{\mathrm{f}}` - matrix of size :math:`(M \times G \times G)`

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

* **1D problems**: ``delta_x`` - array of cell widths, shape :math:`(I,)`
* **2D problems**: ``delta_x`` and ``delta_y`` - arrays of cell widths, shapes :math:`(I,)` and :math:`(J,)` respectively


Angular Data
------------

Angular quadrature data (discrete directions and weights) is generated automatically:

.. autofunction:: ants.angular_x

.. autofunction:: ants.angular_xy


Source and Boundary Data
------------------------

External source and boundary conditions can have various shapes depending on problem type.
The dimensions are flattened to allow flexible representations. Dimension specifiers
(``qdim``, ``bcdim_x``, ``bcdim_y``) indicate the actual shape when needed.

**One-Dimensional Source Dimensions:**

.. table::

    +------------------------+----------+------------------------+----------+
    | ``external`` shape     | ``qdim`` | ``boundary_x`` shape   | ``bcdim``|
    +========================+==========+========================+==========+
    | :math:`(I,)`           | 1        | :math:`(2,)`           | 1        |
    +------------------------+----------+------------------------+----------+
    | :math:`(I, G)`         | 2        | :math:`(2, G)`         | 2        |
    +------------------------+----------+------------------------+----------+
    | :math:`(I, N, G)`      | 3        | :math:`(2, N, G)`      | 3        |
    +------------------------+----------+------------------------+----------+
    | :math:`(I, N, G, T)`   | 4        | :math:`(2, N, G, T)`   | 4        |
    +------------------------+----------+------------------------+----------+


**Two-Dimensional Source Dimensions:**

.. list-table::
     :header-rows: 1

     * - ``external`` shape
         - ``qdim``
         - ``boundary_x`` shape
         - ``bcdim_x``
         - ``boundary_y`` shape
         - ``bcdim_y``
     * - :math:`(I, J)`
         - 1
         - :math:`(2,)`
         - 1
         - :math:`(2,)`
         - 1
     * - :math:`(I, J, G)`
         - 2
         - :math:`(2, J)`
         - 2
         - :math:`(2, I)`
         - 2
     * - :math:`(I, J, N^2, G)`
         - 3
         - :math:`(2, J, G)`
         - 3
         - :math:`(2, I, G)`
         - 3
     * - :math:`(I, J, N^2, G, T)`
         - 4
         - :math:`(2, J, N^2, G)`
         - 4
         - :math:`(2, I, N^2, G)`
         - 4
     * -
         -
         - :math:`(2, J, N^2, G, T)`
         - 5
         - :math:`(2, I, N^2, G, T)`
         - 5


General Parameters
-------------------

All problems require a parameters dictionary specifying problem configuration.
For detailed parameter documentation and working examples, see :doc:`extra`.
