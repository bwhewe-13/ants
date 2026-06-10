.. _sec-homogenized-zones:

Homogenized Zones with Multiple Materials
=========================================

In this analysis, the two-dimensional space for the deterministic neutron
transport equation is represented as a series of quadrilaterals. A triangular
mesh has been shown to be effective :cite:`lewis1993`, but it would alter the
discretization schemes presented in
:ref:`the spatial discretization section <sec-nte-spatial>`. While this does not
present an issue with media that use quadrilateral shaped materials, it severely
limits the types of two-dimensional problems that can be simulated. To allow for
additional material shapes in two-dimensional spaces, namely cylinders and
triangles, a homogenization process is used for each spatial cell.

In this regime, either a cylinder or triangle is overlaid on the Cartesian grid,
as shown in the figures below. For each spatial cell which contains regions of
both materials, a composite material is created using

.. math::

   \widehat{\sig{t}} = \rho \, \sig[1]{t} + (1 - \rho) \, \sig[2]{t}

where :math:`\sig[1]{t}` and :math:`\sig[2]{t}` are the original materials and
:math:`\rho` is the percentage of the first material present in the cell. This
allows for preservation of the cross-sectional information of the spatial cell
without requiring the exact shape. To prevent having an excessive number of
materials through this homogenization process, the :math:`\rho` value was limited
to four decimal places.

**Homogenized Zones — Cylinder**

.. container:: clickable-tikz

    .. tikz::
        :include: tikz/circle-mesh.tikz
        :xscale: 100

**Homogenized Zones — Triangle**

.. container:: clickable-tikz

    .. tikz::
        :include: tikz/triangle-mesh.tikz
        :xscale: 100

A Monte Carlo approach was taken to determine the :math:`\rho` values for the
cylindrical shape. A quarter cylinder, as seen in the cylinder figure, is used and
propagated over the four quadrants for the full cylinder. A uniform distribution
is sampled across the medium in both the :math:`x` and :math:`y` directions where
the specific spatial cell is determined :math:`(i, j)` for each particle
:math:`(x_{n}, y_{n})`. The sample radius :math:`r_{n}` is calculated for each
particle which determined the material the sample point is occupying. This is
added to the tally of the corresponding material for each spatial cell which can
be normalized to calculate the :math:`\rho` values. A quarter cylinder was used
for symmetry across each quadrant and a sufficient number of samples were used for
an accurate approximation of the zone homogenization.

The :math:`\rho` values for the triangle shape were calculated using finite
elements. The general structure of the triangle used with the Cartesian grid is
shown in the triangle figure, where the vertices of the triangle are represented
as :math:`(x_{1}, y_{1})`, :math:`(x_{2}, y_{2})`, and :math:`(x_{3}, y_{3})`. As
with the cylinder shape, a uniform distribution is sampled across the medium to
calculate the spatial cell :math:`(i, j)` for the particle. To determine the
material each particle is currently in, a function can be created using the
triangle shape functions from finite elements :cite:`lacerda2010`. The global
coordinates of the triangle vertices can be transformed onto the local coordinate
system represented as :math:`(\xi, \eta)`. The three shape functions are
calculated as :math:`\xi`, :math:`\eta`, and :math:`1 - \xi - \eta` in which the
function to determine the correct medium is

.. math::

   f(\xi, \eta) = \min (\xi, \eta, 1 - \xi - \eta)

in which :math:`f(\xi, \eta) \geq 0` is inside or on the edge of the triangle and
:math:`f(\xi, \eta) < 0` is outside the triangle. The transformation is carried
out as

.. math::

   \begin{aligned}
   \xi &= \begin{bmatrix}
       x - x_{1} & x_{3} - x_{1} \\
       y - y_{1} & y_{3} - y_{1}
   \end{bmatrix} / \Delta \\
   \eta &= \begin{bmatrix}
       x_{2} - x_{1} & x - x_{1} \\
       y_{2} - y_{1} & y - y_{1}
   \end{bmatrix} / \Delta \\
   \Delta &= \begin{bmatrix}
       x_{2} - x_{1} & x_{3} - x_{1} \\
       y_{2} - y_{1} & y_{3} - y_{1}
   \end{bmatrix}
   \end{aligned}

where :math:`(x_{1}, y_{1})`, :math:`(x_{2}, y_{2})`, and :math:`(x_{3}, y_{3})`
are the vertices of the triangle and :math:`(x_{n}, y_{n})` is the sample point. A
sufficient number of samples and normalization is used to determine the material
percentages :math:`\rho`.


Preset Two-Dimensional Layouts
==============================

The homogenization above is applied in practice by
``ants.main.weight_matrix2d`` (Monte Carlo zone fractions for curved/triangular
shapes) and ``ants.main.spatial2d`` (exact integer maps for rectangular regions).
The :mod:`ants.utils.mesh2d` module bundles these into ready-made benchmark
layouts. Each helper returns either a **weight matrix** of shape
``(cells_x, cells_y, materials)`` (homogenized fractions per cell) or an integer
**medium map** of shape ``(cells_x, cells_y)`` — both consumable as the
``medium_map`` input described in :doc:`standard-inputs`.

.. list-table:: Available layouts
   :widths: 20 15 65
   :header-rows: 1

   * - Layout
     - Output
     - Description
   * - ``cylinder-1mat``
     - weight matrix
     - Single fissile cylinder embedded in void.
   * - ``cylinder-2mat``
     - weight matrix
     - Fuel core + reflector annulus embedded in void.
   * - ``double-chevron``
     - weight matrix
     - Four triangular fuel regions in an HDPE moderator (9 × 9 cm benchmark).
   * - ``c5g7``
     - medium map
     - 2D C5G7 MOX fuel assembly benchmark (7 materials, rectangular regions).

Layout factory functions
-------------------------

.. autofunction:: ants.utils.mesh2d.cylinder_1mat

.. autofunction:: ants.utils.mesh2d.cylinder_2mat

.. autofunction:: ants.utils.mesh2d.double_chevron

.. autofunction:: ants.utils.mesh2d.c5g7

.. autofunction:: ants.utils.mesh2d.resize_weight_matrix

Command-line interface
----------------------

The same layouts can be generated (and optionally plotted or saved) from the
command line:

.. code-block:: bash

    # Generate a layout
    python -m ants.utils.mesh2d <layout> --cells-x N --cells-y N [--plot] [--save out.npy]

    # Example: a 100 x 100 single-cylinder weight matrix, saved to disk
    python -m ants.utils.mesh2d cylinder-1mat --cells-x 100 --cells-y 100 --save cyl.npy

    # Resize an existing weight matrix or medium map to a new grid
    python -m ants.utils.mesh2d --resize cyl.npy --cells-x 200 --cells-y 200 --save cyl200.npy

Layout-specific overrides (e.g. ``--radius``, ``--r-inner`` / ``--r-outer``,
``--length-x`` / ``--length-y``, ``--n-particles``) are available for the Monte
Carlo cylinder layouts.
