Three Dimensional Collision-Based Hybrid Method
================================================

The three-dimensional hybrid method follows :doc:`two-dimensional` with a
third axis. The uncollided equation uses the full :math:`G` groups and the
:math:`S_M` quadrature over the full sphere, and the collided equation is
solved on :math:`\hG` groups and an :math:`S_{\hM}` quadrature. A 3D product
quadrature from ``ants.angular_xyz(N)`` has :math:`2 N^2` ordinates, so each
source iteration of the collided equation sweeps a much smaller set.


Running the Hybrid Method in ANTS
-----------------------------------

The three-dimensional hybrid method is ``ants.hybrid3d.time_dependent``, with
the same inputs as ``ants.hybrid2d.time_dependent``:

.. code-block:: python

   from ants.hybrid3d import time_dependent
   from ants.utils import hybrid as hytools

   flux = time_dependent(materials_u, materials_c, sources, geometry,
                         quadrature_u, quadrature_c, solver, time_data,
                         hybrid_data)

The uncollided inputs (``materials_u``, ``quadrature_u``, and the sources)
are the same as for ``ants.timed3d.time_dependent``, and ``quadrature_c`` is
a lower order 3D quadrature, for example
``ants.angular_xyz(angles_c, bc_x=bc_x, bc_y=bc_y, bc_z=bc_z)``. The
``materials_c`` and ``hybrid_data`` inputs are built as in 2D.

The scalar flux of the last step is returned with shape
``(cells_x, cells_y, cells_z, groups)``. All four time discretizations,
anisotropic scattering, artificial scattering, ``save_to_file``, and the BDF2
restart (``ants.hybrid3d.restart_bdf2``) are supported. With reflecting
boundaries, ``quadrature_u`` must be ordered for the boundary conditions,
while ``quadrature_c`` only needs to include the reflection of each ordinate.

``ants.vhybrid3d.time_dependent`` takes the number of collided groups and
angles of each time step instead of ``materials_c``, ``quadrature_c``, and
``hybrid_data``. The collided angles are the number of angles per dimension
given to ``ants.angular_xyz``, so ``angles_c = 2`` is the 8 ordinate set.

.. code-block:: python

   from ants.vhybrid3d import time_dependent

   vgroups_c = np.array([groups_c] * steps, dtype=np.int32)
   vangles_c = np.array([angles_c] * steps, dtype=np.int32)
   flux = time_dependent(materials_u, vgroups_c, sources, geometry,
                         quadrature_u, vangles_c, solver, time_data, edges_g)

With the same collided grid on every step, ``vhybrid3d`` and ``hybrid3d``
give the same solution.


General Process in ANTS
--------------------------

The steps of each time step are those of :doc:`two-dimensional`, with the
spatial cells ``I x J x K``. The uncollided solve and the final sweep each
cover all :math:`G \times N_u` group and ordinate pairs once, and all of the
iterations are on the collided grid. The uncollided and collided flux moments
are both stored for every cell, which adds to the memory of the 3D problem.
