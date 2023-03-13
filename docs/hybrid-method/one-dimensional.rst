One Dimensional Collision-Based Hybrid Method
==============================================

The collision-based hybrid method is used to accelerate the convergence
time to solve the time-dependent neutron transport equation. This 
method breaks the transport equation into collided and uncollided 
portions and solves each equation on different angular and energy grids. 
The collided portion is solved on a coarser grid to allow for 
convergence acceleration.


General Process in ANTS
--------------------------

This is the step by step process for how ``hybrid1d.backward_euler()`` 
is implemented in ``ANTS``. The dimensions of each step is also included
with ``I`` are the number of spatial cells, ``Nu, Nc`` are the uncollided
and collided number of angles, and ``Gu, Gc`` are the uncollided and 
collided number of energy groups.

#. Start with the uncollided source :math:`Q_u \, (I \times N \times G_u)`.

#. Calculate the star source: :math:`Q^* \, (I \times N \times G_u)\, = Q_u + \psi^{(\ell - 1)} \, (v dt)^{-1}`.

#. Solve the uncollided transport equation for :math:`\psi_u \, (I \times G_u)`.

#. Calculate the collided source :math:`Q_c \, (I \times G_u) \,  = \sigma_s \, \psi_u + \sigma_f \, \psi_u` and resize :math:`Q_c` to :math:`(I \times G_c)`.

#. Solve the collided transport equation for :math:`\psi_c \, (I \times G_c)`.

#. Solve for the total source :math:`Q_t \, (I \times G_u) \, = (\sigma_s + \sigma_f ) \, ( \psi_c + \psi_u)` where :math:`\psi_c` was resized to :math:`(I \times G_u)`.

#. Calculate the new star source :math:`Q^* \, (I \times N \times G_u)\, = Q_t + \psi^{(\ell - 1)} \, (v dt)^{-1}`.


Additional Parameters for ANTS
---------------------------------

* ``index_c``: Length ``Gu``. Which collided (coarse) group the uncollided (fine) group is part of. 

* ``index_u``: Length ``Gc + 1``. The edges of the collided (coarse) groups as they relate to their uncollided (fine) group counterpart.

* ``factor_u``: Length ``Gu``. The inverse of the number of uncollided (fine) groups for each collided (coarse) group.