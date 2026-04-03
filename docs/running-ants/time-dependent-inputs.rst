Time-Dependent Problems
===============================================

Time-dependent problems add temporal evolution to the neutron transport equation
and require additional parameters beyond the standard inputs.


Time Stepping Parameters
------------------------

To solve time-dependent problems, you must specify:

:param int steps: Number of time steps to simulate. Defaults to 0.

:param float dt: Time step size (in seconds). Defaults to 1.0.

These parameters should be included in the parameters dictionary along with
the standard parameters described in :doc:`extra`.


Time-Dependent Source Requirements
-----------------------------------

For time-dependent problems, the external source may include a temporal dimension.
The source shape must match the ``qdim`` parameter as described in :doc:`standard-inputs`.

If your source has a temporal dimension, it should be the last dimension:

* **1D**: :math:`(I, N, G, T)` with qdim = 4
* **2D**: :math:`(I, J, N^2, G, T)` with qdim = 4

If your source is time-independent, omit the temporal dimension and use
qdim = 3 instead.


Backward Euler Method
----------------------

Time integration uses backward Euler (fully implicit, first-order accurate):

.. math::

    \phi^{n+1} = \phi^n + \Delta t \cdot \text{(implicit update)}

This method is unconditionally stable and suitable for problems with stiff source terms
or large time steps, but has lower accuracy than higher-order methods.


Available Time-Dependent Solvers
---------------------------------

The following modules provide time-dependent transport solvers:

* :py:mod:`timed1d` - One-dimensional time-dependent solver
* :py:mod:`timed2d` - Two-dimensional time-dependent solver
* :py:mod:`hybrid1d` - One-dimensional time-dependent with collision-based hybrid acceleration
* :py:mod:`hybrid2d` - Two-dimensional time-dependent with collision-based hybrid acceleration
* :py:mod:`vhybrid2d` - Two-dimensional time-dependent with vectorized hybrid acceleration


Hybrid Methods for Acceleration
--------------------------------

For expensive time-dependent calculations, the hybrid method decomposes the solution
into collision-dominated and collision-free components, allowing efficient treatment
of highly scattering regions. This can significantly accelerate convergence when
there are large variations in optical thickness.


Example: Time-Dependent 1D Problem
-----------------------------------

See :doc:`extra` for a complete working example of a time-dependent problem.
