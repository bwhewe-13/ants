Time-Dependent Problems
===============================================

Time-dependent problems add temporal evolution to the neutron transport equation
and require additional parameters beyond the standard inputs.


Time Stepping Parameters
------------------------

Time stepping is configured via a ``TimeDependentData`` object:

.. code-block:: python

    from ants.datatypes import TimeDependentData, TemporalDiscretization

    time_data = TimeDependentData(
        steps=100,    # number of time steps
        dt=0.01,      # time step size (seconds)
        time_disc=TemporalDiscretization.BDF1,  # optional; BDF1 is the default
    )


Time Integrators
----------------

Four time integration schemes are available via the ``TemporalDiscretization`` enum:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Enum value
     - Order
     - Type
     - Notes
   * - ``BDF1``
     - 1st
     - Implicit
     - Backward Euler. Default. Unconditionally stable; lowest accuracy.
   * - ``BDF2``
     - 2nd
     - Implicit
     - Second-order backward difference. More accurate than BDF1 for smooth
       transients; still unconditionally stable.
   * - ``CN``
     - 2nd
     - Implicit
     - Crank-Nicolson. Second-order; can exhibit oscillations on stiff problems.
       Requires edge-based initial flux (see below).
   * - ``TR_BDF2``
     - 2nd
     - Implicit
     - Trapezoidal Rule / BDF2 composite. Combines CN accuracy with BDF2
       damping. Recommended for stiff multigroup problems. Requires edge-based
       initial flux (see below).


Initial Flux
------------

The shape of the initial flux in ``SourceData`` depends on the time integrator.

**BDF1 and BDF2** — cell-centered initial flux:

.. code-block:: python

    # 1D: shape (I, N, G)
    sources = SourceData(
        initial_flux=np.zeros((cells_x, angles, groups)),
        ...
    )

    # 2D: shape (I, J, N², G)
    sources = SourceData(
        initial_flux=np.zeros((cells_x, cells_y, angles**2, groups)),
        ...
    )

**CN and TR-BDF2** — edge-based initial angular flux (at cell faces):

.. code-block:: python

    # 1D: initial_flux_x shape (I+1, N, G)
    sources = SourceData(
        initial_flux_x=np.zeros((cells_x + 1, angles, groups)),
        ...
    )

    # 2D: initial_flux_x shape (I+1, J, N², G), initial_flux_y shape (I, J+1, N², G)
    sources = SourceData(
        initial_flux_x=np.zeros((cells_x + 1, cells_y, angles**2, groups)),
        initial_flux_y=np.zeros((cells_x, cells_y + 1, angles**2, groups)),
        ...
    )


Time-Dependent Sources
-----------------------

External sources that vary in time include a leading time dimension:

* **1D**: shape ``(T, I, N, G)``
* **2D**: shape ``(T, I, J, N², G)``

where ``T`` is the number of time steps. Time-independent sources are broadcast
automatically from shapes ``(1, I, ...)`` or ``(I, ...)``

For a steady source repeated across all steps, use::

    from ants.external1d import time_dependence_constant
    external = time_dependence_constant(external_ss)  # adds T=1 leading dimension

Time-dependent boundary sources similarly include a leading time dimension:

* **1D boundary**: shape ``(T, 2, N, G)``
* **2D boundary x**: shape ``(T, 2, J, N², G)``
* **2D boundary y**: shape ``(T, 2, I, N², G)``

Helper functions such as ``ants.boundary1d.time_dependence_decay_02()`` and
``ants.boundary2d.time_dependence_decay_03()`` generate these arrays from a
base boundary condition and a time axis.


Available Time-Dependent Solvers
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``ants.timed1d``
     - One-dimensional time-dependent solver
   * - ``ants.timed2d``
     - Two-dimensional time-dependent solver
   * - ``ants.hybrid1d``
     - 1D time-dependent with collision-based hybrid acceleration
   * - ``ants.hybrid2d``
     - 2D time-dependent with collision-based hybrid acceleration
   * - ``ants.vhybrid1d``
     - 1D time-dependent with vectorized hybrid (per-step coarse resolution)
   * - ``ants.vhybrid2d``
     - 2D time-dependent with vectorized hybrid (per-step coarse resolution)

All modules expose a single public function ``time_dependent()``.


Hybrid Methods for Acceleration
--------------------------------

The collision-based hybrid method decomposes the problem into:

* **Uncollided component**: solved on the fine angular/energy grid
* **Collided component**: solved on a coarser angular/energy grid

This reduces computational cost when the collided flux is smoother than the
uncollided flux (common in multigroup streaming-dominated problems).

Standard hybrid (``hybrid1d``, ``hybrid2d``) requires two ``MaterialData`` objects
(one for each resolution) and a ``HybridData`` object encoding the group indexing::

    from ants.datatypes import HybridData
    from ants.utils import hybrid as hytools

    fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    hybrid_data = HybridData(fine_idx=fine_idx, coarse_idx=coarse_idx, factor=factor)

The vectorized hybrid (``vhybrid1d``, ``vhybrid2d``) instead accepts per-step integer
arrays and handles coarsening internally::

    vgroups_c = np.array([groups_c] * steps, dtype=np.int32)
    vangles_c = np.array([angles_c] * steps, dtype=np.int32)
    flux = time_dependent(mat_data, vgroups_c, sources, geometry,
                          quadrature_u, vangles_c, solver, time_data, edges_g)

See :doc:`examples` for complete runnable examples of each solver.
