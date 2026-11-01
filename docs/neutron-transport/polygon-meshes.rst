.. _sec-nte-polygon:

Two-Dimensional Polygon Meshes
==============================

The two-dimensional slab in :ref:`the spatial discretization section
<sec-nte-spatial>` sweeps over rows and columns of rectangular cells. Hexagonal
lattices and triangular meshes cannot be represented this way without
staircasing the material boundaries. For these geometries, ANTS uses a single
sweep over convex polygon cells, so triangles, hexagons, and the cut cells along
the edge of a clipped lattice (quadrilaterals and pentagons) are all handled the
same way. The polygon mesh is set with ``Geometry.POLYGON2D`` and works with the
fixed source, criticality, time-dependent, hybrid, and nearby problem solvers.

As with the spatial discretization section, the steady-state, mono-energetic
transport equation in Equation :eq:`nte-spatial` is used, with the right-hand
side combined into a single source

.. math::

   Q_{m} \equiv \sig{s} \phi + \chi \nu \sig{f} \phi + q_{m}.


Cell Balance
------------

Each polygon cell :math:`c` has an area :math:`A_{c}` and faces :math:`f` with
length :math:`L_{f}` and unit normal :math:`\bn_{c,f}` pointing out of the cell.
For the discrete ordinate :math:`\bsOmega_{m} = (\mu_{m}, \eta_{m})`, the
projection of the direction onto each face

.. math::
   :label: nte-polygon-dot

   d_{c,f} = \bsOmega_{m} \cdot \bn_{c,f} = \mu_{m} \, n_{c,f}^{x} + \eta_{m} \, n_{c,f}^{y}

determines if neutrons enter (:math:`d_{c,f} < 0`) or leave
(:math:`d_{c,f} > 0`) the cell through face :math:`f`. Integrating Equation
:eq:`nte-spatial` over the cell and using the divergence theorem on the streaming
term gives the balance equation

.. math::
   :label: nte-polygon-balance

   \sum_{f \in c} d_{c,f} \, L_{f} \, \psi_{f,m} + \sig{t} A_{c} \, \psi_{c,m} = A_{c} \, Q_{c,m},

where :math:`\psi_{f,m}` is the face averaged angular flux and
:math:`\psi_{c,m}` is the cell averaged angular flux. Since the cell is closed,
:math:`\sum_{f} L_{f} \bn_{c,f} = 0`, so the projected width of the incoming
faces is equal to that of the outgoing faces

.. math::
   :label: nte-polygon-width

   P_{c,m} = \sum_{d_{c,f} < 0} |d_{c,f}| \, L_{f} = \sum_{d_{c,f} > 0} d_{c,f} \, L_{f}.

For a rectangle, this is :math:`|\mu_{m}| \Delta y_{j} + |\eta_{m}| \Delta x_{i}`.
The incoming face fluxes are known from the upwind cells or the boundary, which
leaves the cell and outgoing face fluxes as unknowns. A closure relation is
needed to connect them.


Closure Relation
----------------

The step, diamond difference, and step characteristic methods can all be
written as a weighted average of the incoming and outgoing fluxes

.. math::
   :label: nte-polygon-closure

   \psi_{c,m} = \frac{1}{2} \left[ (1 + W) \, \psi_{\mathrm{out},m} + (1 - W) \, \psi_{\mathrm{in},m} \right],

which is rearranged to update the outgoing flux as

.. math::
   :label: nte-polygon-outgoing

   \psi_{\mathrm{out},m} = \frac{2 \psi_{c,m} - (1 - W) \, \psi_{\mathrm{in},m}}{1 + W}.

The step method uses :math:`W = 1`, so the outgoing flux is the cell flux, and the
diamond difference method uses :math:`W = 0`, which is Equation
:eq:`nte-dd-1d`. The step characteristic method follows from solving
:math:`|\bsOmega| \, \partial \psi / \partial s + \sig{t} \psi = Q` exactly along a
chord of length :math:`h` with a flat source. Averaging the solution along the
chord gives Equation :eq:`nte-polygon-closure` with

.. math::
   :label: nte-polygon-sc

   W(\tau) = \coth\left(\frac{\tau}{2}\right) - \frac{2}{\tau}, \hspace{30pt} \tau = \frac{\sig{t} \, h}{|d|},

where :math:`\tau` is the optical thickness of the chord. The weight moves from
diamond difference in optically thin cells, :math:`W \approx \tau / 6` as
:math:`\tau \to 0`, to step in optically thick cells, :math:`W \to 1` as
:math:`\tau \to \infty`. The series is used for :math:`\tau < 10^{-4}`, which
also covers void cells. The same weight is used on the Cartesian sweeps with
:math:`h = \Delta x_{i}` and :math:`|d| = |\mu_{m}|`.

What is meant by the incoming flux and the chord length depends on the shape of
the cell, which is covered in the next two sections.


Centrally Symmetric Cells
-------------------------

Rectangles and whole hexagons are centrally symmetric, meaning every face
:math:`f` has an opposite face :math:`f'` of the same length with
:math:`\bn_{c,f'} = - \bn_{c,f}`. When :math:`f` is outgoing, :math:`f'` is
incoming with :math:`d_{c,f'} = -d_{c,f}`, so each outgoing face is paired with
its opposite face as the incoming flux in Equation :eq:`nte-polygon-outgoing`.
The chord length is the distance :math:`h_{f}` between the two faces, so each pair
has its own weight :math:`W_{f} = W(\sig{t} h_{f} / |d_{c,f}|)`.

The contribution of a pair to Equation :eq:`nte-polygon-balance` is

.. math::

   d_{c,f} L_{f} \, \psi_{f,m} + d_{c,f'} L_{f'} \, \psi_{f',m} = \frac{2}{1 + W_{f}} \, |d_{c,f'}| \, L_{f'} \left( \psi_{c,m} - \psi_{f',m} \right),

which only depends on the cell flux and the known incoming flux. Summing over the
incoming faces, the cell flux is

.. math::
   :label: nte-polygon-paired

   \psi_{c,m} = \frac{A_{c} \, Q_{c,m} + \displaystyle\sum_{d_{c,f} < 0} \frac{2}{1 + W_{f}} |d_{c,f}| \, L_{f} \, \psi_{f,m}}{\sig{t} A_{c} + \displaystyle\sum_{d_{c,f} < 0} \frac{2}{1 + W_{f}} |d_{c,f}| \, L_{f}},

and the outgoing faces are updated with Equation :eq:`nte-polygon-outgoing`. For
a rectangle, dividing by :math:`A_{c} = \Delta x_{i} \Delta y_{j}` gives

.. math::

   \left(\sig{t} + \frac{2}{1 + W_{x}} \frac{|\mu_{m}|}{\Delta x_{i}} + \frac{2}{1 + W_{y}} \frac{|\eta_{m}|}{\Delta y_{j}} \right) \psi_{i,j,m} = Q_{i,j,m} + \frac{2}{1 + W_{x}} \frac{|\mu_{m}|}{\Delta x_{i}} \psi_{\mathrm{in},x} + \frac{2}{1 + W_{y}} \frac{|\eta_{m}|}{\Delta y_{j}} \psi_{\mathrm{in},y},

which is the two-dimensional slab sweep, and the diamond difference equation in
the spatial discretization section when :math:`W_{x} = W_{y} = 0`. A rectangle
mesh built as a polygon mesh gives the same solution as ``SLAB2D`` to machine
precision for all three methods. The diamond difference and step characteristic
methods are second order accurate on these cells.


Triangles and Cut Cells
-----------------------

Triangles, and the cut cells left after clipping a lattice to a boundary, have
no opposite faces. For these cells, every outgoing face uses the flux weighted
mean of the incoming faces

.. math::
   :label: nte-polygon-mean

   \bar{\psi}_{\mathrm{in},m} = \frac{1}{P_{c,m}} \sum_{d_{c,f} < 0} |d_{c,f}| \, L_{f} \, \psi_{f,m}

as the incoming flux in Equation :eq:`nte-polygon-outgoing`. The chord length is
the mean chord length :math:`A_{c} / P_{c,m}` of the cell in direction
:math:`\bsOmega_{m}`, so there is a single weight
:math:`W_{c} = W(\sig{t} A_{c} / P_{c,m})` for the cell. Substituting into
Equation :eq:`nte-polygon-balance` with Equation :eq:`nte-polygon-width`, the
cell flux is

.. math::
   :label: nte-polygon-unpaired

   \psi_{c,m} = \frac{A_{c} \, Q_{c,m} + \displaystyle\frac{2}{1 + W_{c}} P_{c,m} \, \bar{\psi}_{\mathrm{in},m}}{\sig{t} A_{c} + \displaystyle\frac{2}{1 + W_{c}} P_{c,m}}.

This keeps the cell balance exact and is exact in an infinite medium, but every
outgoing face gets the same flux, which makes the diamond difference and step
characteristic methods first order accurate on these cells. They still have a
smaller error than the step method. On hexagonal lattices clipped to a straight
boundary, the error from the first order cut cells is carried downstream, so
``hexagon_mesh(..., clip=False)`` can be used to keep only whole hexagons when
second order convergence is needed. A second order closure on triangles would
need a linear (discontinuous Galerkin) representation in each cell.

In both cases, faces parallel to the direction (:math:`d_{c,f} = 0`) do not enter
Equation :eq:`nte-polygon-balance` and are set to the cell flux.


Sweep Order
-----------

The Cartesian sweep moves through the rows and columns in the direction of
:math:`\mu_{m}` and :math:`\eta_{m}`. On a polygon mesh there is no natural order,
so for each angle a directed graph is built where every interior face with
:math:`d_{c,f} \neq 0` connects the upwind cell to the downwind cell. A
topological sort (Kahn's algorithm) of this graph gives an order where every cell
comes after the cells that feed it. This is done once for each mesh and
quadrature. Convex cells on structured triangle and hexagon meshes cannot form a
cycle, and an error is raised if a cycle is found. Each angle is swept
independently, so the angles are split across OpenMP threads in the same way as
the Cartesian sweep.


Boundary Conditions
-------------------

The boundary of a polygon mesh is split into sides, with one outward normal
:math:`\bn_{s}` per side, such as four sides for a rectangle and six sides for a
hexagonal domain. Each side has its own boundary condition. Vacuum and source
boundaries set the incoming flux on each boundary face. For a reflected side, the
incoming flux in direction :math:`\bsOmega_{m}` is the outgoing flux in the
reflected direction

.. math::
   :label: nte-polygon-reflect

   \bsOmega_{m'} = \bsOmega_{m} - 2 \left( \bsOmega_{m} \cdot \bn_{s} \right) \bn_{s},

which must be in the angular quadrature. Sides aligned with the :math:`x` and
:math:`y` axes work with any product quadrature. Hexagonal sides at multiples of
60 degrees need the number of angles to be a multiple of six, and an error is
raised if a reflected direction is missing. The reflected fluxes are lagged by one
iteration in the source iteration sweep, and the known source sweeps are repeated
until the reflected fluxes converge.


Time-Dependent Problems
-----------------------

The temporal discretization schemes add a :math:`1 / (v \Delta t)` term to the
total cross section,
:math:`\sig{*} = \sig{t} + c / (v \Delta t)` where :math:`c` depends on the
scheme (:math:`c = 1` for backward Euler), and the previous time steps to the
source. The balance equation uses :math:`\sig{*}`, but the step characteristic
weight in Equation :eq:`nte-polygon-sc` is computed with :math:`\sig{t}` only.
The time source :math:`\psi^{n} / (v \Delta t)` has the same shape in the cell as
the flux it came from, so it cancels the time term along the chord instead of
acting as a flat source. With this, a time-dependent problem with constant
sources converges to the same solution as the fixed source problem. The same is
done on the Cartesian sweeps.

The Crank-Nicolson and TR-BDF2 schemes need the streaming term at the previous
time step. For a polygon mesh, the face fluxes and the cell fluxes are both kept
between time steps and the streaming term is

.. math::

   \bsOmega_{m} \cdot \nabla \psi^{n}_{m} \approx \frac{1}{A_{c}} \sum_{f \in c} d_{c,f} \, L_{f} \, \psi^{n}_{f,m}.

The Cartesian slabs only keep the edge fluxes, so the cell flux at the previous
time step is found from the edges with Equation :eq:`nte-polygon-closure`, where
the incoming edge is chosen by the sign of :math:`\mu_{m}` (and
:math:`\eta_{m}`). In two dimensions, the :math:`x` and :math:`y` estimates are
averaged. Using the same closure as the sweep keeps the step and step
characteristic methods consistent with the steady-state solution, where taking
the average of the edges would only be correct for diamond difference.


Accuracy
--------

The spatial order of accuracy on polygon meshes, measured with a manufactured
solution:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Mesh
     - Step
     - Diamond difference
     - Step characteristic
   * - Rectangles
     - 1st
     - 2nd
     - 2nd
   * - Whole hexagons
     - 1st
     - 2nd
     - 2nd
   * - Triangles
     - 1st
     - 1st
     - 1st
   * - Clipped hexagons (cut cells)
     - 1st
     - 1st
     - 1st
