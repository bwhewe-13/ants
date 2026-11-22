Two Dimensional Collision-Based Hybrid Method
==============================================

The collision-based hybrid method extends to two-dimensional Cartesian
problems :cite:`whewell2025,whewell2024`. The benefit is larger than in one
dimension, since the number of ordinates in a product quadrature grows with
the square of the angular order and each source iteration sweeps all of them.
As in one dimension, the transport equation is split into uncollided and
collided parts. The uncollided equation uses the full :math:`G` groups and
:math:`S_M` quadrature, while the collided equation is solved on a coarser
grid of :math:`\hG` groups and an :math:`S_{\hM}` quadrature. The formulation
is the same as in :doc:`methodology`.


Running the Hybrid Method in ANTS
-----------------------------------

The two-dimensional hybrid method is ``ants.hybrid2d.time_dependent``:

.. code-block:: python

   from ants.hybrid2d import time_dependent
   from ants.utils import hybrid as hytools

   flux = time_dependent(materials_u, materials_c, sources, geometry,
                         quadrature_u, quadrature_c, solver, time_data,
                         hybrid_data)

The uncollided inputs (``materials_u``, ``quadrature_u``, and the sources)
are the same as for ``ants.timed2d.time_dependent``. The collided inputs are
built from them:

* ``materials_c``: the cross sections and velocities collapsed to the coarse
  groups with ``hytools.coarsen_materials`` and ``hytools.coarsen_velocity``.

* ``quadrature_c``: a lower order quadrature, for example
  ``ants.angular_xy(angles_c, bc_x=bc_x, bc_y=bc_y)``.

* ``hybrid_data``: the ``HybridData`` mapping between the fine and coarse
  groups, from ``hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)``.

  * ``coarse_idx``: length :math:`G`. The coarse group that each fine group
    is part of.

  * ``fine_idx``: length :math:`\hG + 1`. The edges of the coarse groups as
    indices of the fine groups.

  * ``factor``: length :math:`G`. The width of each fine group divided by the
    width of its coarse group, used to refine the collided flux.

The scalar flux is returned with shape ``(steps, cells_x, cells_y, groups)``.
All four time discretizations (backward Euler, Crank-Nicolson, BDF2, and
TR-BDF2), anisotropic scattering, and artificial scattering are supported.
With reflecting boundaries, ``quadrature_u`` must be ordered for the boundary
conditions, since each uncollided ordinate is swept once, while
``quadrature_c`` only needs to include the reflection of each ordinate.

``ants.vhybrid2d.time_dependent`` takes the number of collided groups and
angles instead of ``materials_c``, ``quadrature_c``, and ``hybrid_data``, and
builds the collided grids itself. The examples ``hybrid_2d_chevron_01.py``,
``hybrid_2d_cylinder_01.py``, and ``vhybrid_2d_chevron_01.py`` in the
``examples/`` folder set up the full problems.


General Process in ANTS
--------------------------

These are the steps for one backward Euler time step in
``hybrid2d.time_dependent``. ``I`` and ``J`` are the number of spatial cells
in :math:`x` and :math:`y`, ``Nu`` and ``Nc`` are the number of uncollided
and collided ordinates, and ``Gu`` and ``Gc`` are the number of uncollided
and collided energy groups.

#. Calculate the star source :math:`Q^* \, (I \times J \times N_u \times G_u) = Q_u + \psi^{(n)} \, (v \, \Delta t)^{-1}`.

#. Solve the uncollided transport equation for :math:`\phi_u \, (I \times J \times G_u)`. The source is known, so each ordinate is swept once and no source iteration is needed.

#. Calculate the collided source :math:`Q_c = (\sigma_s + \sigma_f) \, \phi_u` and collapse it to :math:`(I \times J \times G_c)`. The collided equation has no boundary source.

#. Solve the collided transport equation for :math:`\phi_c \, (I \times J \times G_c)` with source iteration and Gauss-Seidel over the coarse groups and ``Nc`` ordinates.

#. Calculate the total source :math:`Q_t = Q^* + (\sigma_s + \sigma_f) \, (\phi_u + \phi_c)`, where :math:`\phi_c` is refined to :math:`(I \times J \times G_u)` with ``factor``.

#. Sweep each of the ``Nu`` ordinates once with :math:`Q_t` for the angular flux :math:`\psi^{(n+1)}`, which is the initial condition of the next time step.

Steps 2 and 6 cost one sweep of all :math:`G \times N_u` group and ordinate
pairs per time step, whatever the size of the collided grid. All of the
iterations are on the collided grid.


Numerical Results
-------------------

Three time-dependent problems were used to compare the hybrid method with
multigroup models coarsened in energy and angle :cite:`whewell2025,whewell2024`.
All three used diamond difference in space, TR-BDF2 in time, and the metrics
described in :doc:`methodology`.

.. list-table::
   :header-rows: 1
   :widths: 18 28 27 27
   :class: text-start

   * -
     - AmBe Lattice
     - Double Chevron
     - Cylinder Mixture
   * - Materials
     - Stainless steel and uranium hydride (100% U-235)
     - Natural uranium and high density polyethylene
     - Uranium dioxide and water, with a void outer region
   * - Source
     - AmBe source at the center
     - 14.1 MeV boundary source at :math:`y = 0` that rises, holds, and decays
     - 2.45 MeV boundary source at :math:`x = y = 80` cm that turns off at 5 µs
   * - Geometry
     - 7 × 7 cm, :math:`\Delta x = \Delta y = 0.05` cm, vacuum boundaries
     - 9 × 9 cm, 90 × 90 cells, vacuum boundaries
     - Quarter cylinder, 100 × 100 cells, reflecting at :math:`x = 0` and :math:`y = 0`
   * - Energy groups
     - :math:`G = 87`
     - :math:`G = 87`
     - :math:`G = 361`
   * - Time steps
     - 50 steps to 0.1 µs
     - 50 steps to 50 µs
     - 60 steps to 6 µs
   * - Reference
     - :math:`S_{40}`
     - :math:`S_{32}`
     - :math:`S_{32}`

The main results were:

* **AmBe lattice.** Ray effects from the central source made the angular
  resolution important. A hybrid model with :math:`G = \hG = 87`,
  :math:`M = 24`, and :math:`\hM = 4` matched the fission rate error of a
  multigroup model with :math:`G = 60` and :math:`M = 24` (relative error
  below 2%) in about an eighth of the wall clock time. At the same wall clock
  time, the :math:`M = 24` uncollided solve also reduced the ray effects of
  the multigroup model.

* **Double chevron.** A hybrid model with :math:`G = 87`, :math:`\hG = 21`,
  :math:`M = 24`, and :math:`\hM = 8` matched the fission rate accuracy of a
  multigroup model with :math:`G = 67` and :math:`M = 24` using 35 times less
  wall clock time. For the higher fidelity models, the hybrid method was about
  50% faster than the equivalent multigroup model.

* **Cylinder mixture.** A hybrid model with :math:`G = 361`, :math:`\hG = 90`,
  :math:`M = 24`, and :math:`\hM = 4` had a scattering rate error similar to
  a multigroup model with :math:`G = 240` and :math:`M = 24` and converged
  about 22 times faster. The highest fidelity hybrid model was 46% faster than
  the :math:`G = 361`, :math:`M = 24` multigroup model.

Except in a few cases, the hybrid method was more accurate than the
multigroup models coarsened to the same cost, and the splitting method (:math:`\hG = G`
and :math:`\hM = M`) was faster than the multigroup method with the same
solution. The advantage shrinks for low fidelity models, where the two sweeps
of every uncollided group and ordinate per time step are a larger part of the
cost. The hybrid method also stores both the uncollided and collided cross
sections.
