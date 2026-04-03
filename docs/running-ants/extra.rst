One-Dimensional Parameters
===============================================

One-dimensional parameters are created in a Python dictionary and converted to
a Cython struct using ``parameters._to_params()``. These parameters are used for
running functions from the :py:mod:`timed1d`, :py:mod:`fixed1d`,
:py:mod:`critical1d`, :py:mod:`hybrid1d`, and :py:mod:`nearby1d` modules.


Core Parameters
---------------

.. py:data:: params_dict

   Python dictionary with the following keys and defaults:

   :param int cells_x: Number of spatial cells (I). Defaults to 10.

   :param int angles: Number of discrete angles. Must be even (N % 2 == 0). Defaults to 4.

   :param int groups: Number of energy groups (G). Defaults to 1.

   :param int materials: Number of material regions (M). Defaults to 1.

   :param int geometry: Type of geometry: 1 = slab, 2 = sphere. Defaults to 1.

   :param int spatial: Type of spatial discretization: 1 = step method, 2 = diamond difference. Defaults to 2.

   :param list[int] bc_x: Boundary conditions for [x = 0, x = X] with 0 = vacuum, 1 = reflective. Must have length 2. Defaults to [0, 0].

   :param int steps: Number of time steps. Defaults to 0.

   :param float dt: Size of the time steps. Defaults to 1.0.

   :param bool angular: If True, output angular flux; if False, output scalar flux. Defaults to False.


Optional Parameters
-------------------

   :param int edges: Flux location: 0 = cell centers, 1 = cell edges. Defaults to 0.

   :param int mg: Multigroup solver: 1 = source iteration, 2 = DMD. Defaults to 1.

   :param int dmd_k: DMD basis size. Defaults to 40.

   :param int dmd_r: DMD matrix rank. Defaults to 2.

   :param int count_nn: Angular convergence max iterations. Defaults to 100.

   :param int count_gg: Energy group convergence max iterations. Defaults to 100.

   :param int count_keff: Power iteration convergence max iterations. Defaults to 100.

   :param float change_nn: Angular convergence tolerance. Defaults to 1e-6.

   :param float change_gg: Energy group convergence tolerance. Defaults to 1e-6.

   :param float change_keff: Power iteration convergence tolerance. Defaults to 1e-6.


Examples
--------

**Fixed Source Problem**

.. code-block:: python

   import numpy as np
   import ants
   from ants.fixed1d import source_iteration

   # Geometry
   cells_x = 100
   length = 1.0
   delta_x = np.repeat(length / cells_x, cells_x)

   # Physics
   angles = 4
   groups = 1

   # Parameters
   info = {
       "cells_x": cells_x,
       "angles": angles,
       "groups": groups,
       "materials": 1,
       "geometry": 1,  # slab
       "spatial": 2,   # diamond difference
       "bc_x": [0, 0], # vacuum boundaries
       "angular": True
   }

   # Cross sections and source
   xs_total = np.array([[1.0]])
   xs_scatter = np.array([[[0.0]]])
   xs_fission = np.array([[[0.0]]])
   external = np.ones((cells_x, 1, 1))
   boundary_x = np.zeros((2, 1, 1))
   boundary_x[0] = 1.0  # boundary source at x=0

   # Setup angular quadrature and solve
   angle_x, angle_w = ants.angular_x(info)
   medium_map = np.zeros((cells_x), dtype=np.int32)
   flux = source_iteration(xs_total, xs_scatter, xs_fission, external,
                           boundary_x, medium_map, delta_x, angle_x, angle_w, info)


**Criticality Problem (Eigenvalue)**

.. code-block:: python

   import numpy as np
   import ants
   from ants.critical1d import power_iteration

   # Geometry
   cells_x = 1000
   length = 10.0
   delta_x = np.repeat(length / cells_x, cells_x)
   edges_x = np.linspace(0, length, cells_x+1)

   # Physics
   angles = 8
   groups = 87

   # Parameters
   info = {
       "cells_x": cells_x,
       "angles": angles,
       "groups": groups,
       "materials": 3,
       "geometry": 2,    # sphere
       "spatial": 2,     # diamond difference
       "bc_x": [1, 0]    # reflective at origin, vacuum at surface
   }

   # Spatial material layout
   layers = [[0, "uranium-%20%", "0-4"],
             [1, "uranium-%0%", "4-6"],
             [2, "stainless-steel-440", "6-10"]]
   medium_map = ants.spatial1d(layers, edges_x)

   # Setup angular quadrature and cross sections
   angle_x, angle_w = ants.angular_x(info)
   materials = np.array(layers)[:,1]
   xs_total, xs_scatter, xs_fission = ants.materials(87, materials)

   # Solve for eigenvalue and flux
   flux, keff = power_iteration(xs_total, xs_scatter, xs_fission, medium_map,
                                delta_x, angle_x, angle_w, info)


Requirements for External Source Dimensions
--------------------------------------------

There are certain requirements for the external source dimensions for
running different problems based on the functions being used. If a function is
not listed, there are no restrictions.

* ``timed1d.backward_euler()``: The external source must be of size (I x N x G) with qdim = 3.

* ``hybrid1d.backward_euler()``: The uncollided external source must be of size (I x Nu x Gu) with qdim = 3. The collided external source must be of size (I x Gc) with qdim = 2.

* ``critical1d.power_iteration()``: The external source must be of size (I x G) with qdim = 2.



Requirements for Boundary Condition Dimensions
-----------------------------------------------

There are certain requirements for the boundary condition dimensions:

* ``hybrid1d.backward_euler()``: The uncollided boundary condition can be of any size. The collided boundary condition must be of size (2) with bcdim = 0.

* ``critical1d.power_iteration()``: The boundary condition must be of size (2) with bcdim = 0.


Two-Dimensional Parameters
===============================================

Two-dimensional parameters are created in a Python dictionary and converted to
a Cython struct using ``parameters._to_params()``. These parameters are used for
running functions from the :py:mod:`timed2d`, :py:mod:`fixed2d`,
:py:mod:`critical2d`, :py:mod:`hybrid2d`, and :py:mod:`vhybrid2d` modules.


Core Parameters
---------------

.. py:data:: params_dict

   Python dictionary with the following keys and defaults:

   :param int cells_x: Number of spatial cells in the x direction (I). Defaults to 10.

   :param int cells_y: Number of spatial cells in the y direction (J). Defaults to 1.

   :param int angles: Number of discrete angles. Must be even (N % 2 == 0). Defaults to 4.

   :param int groups: Number of energy groups (G). Defaults to 1.

   :param int materials: Number of material regions (M). Defaults to 1.

   :param int geometry: Type of geometry: 1 = square, 2 = triangle. Defaults to 1.

   :param int spatial: Type of spatial discretization: 1 = step method, 2 = diamond difference. Defaults to 2.

   :param list[int] bc_x: Boundary conditions for [x = 0, x = X] with 0 = vacuum, 1 = reflective. Must have length 2. Defaults to [0, 0].

   :param int bcdim_x: Size of boundary conditions dimensions for the x direction: 0 = [2], 1 = [2 x J], 2 = [2 x J x G], 3 = [2 x J x N x G]. Defaults to 0.

   :param list[int] bc_y: Boundary conditions for [y = 0, y = Y] with 0 = vacuum, 1 = reflective. Must have length 2. Defaults to [0, 0].

   :param int bcdim_y: Size of boundary conditions dimensions for the y direction: 0 = [2], 1 = [2 x I], 2 = [2 x I x G], 3 = [2 x I x N x G]. Defaults to 0.

   :param int steps: Number of time steps. Defaults to 0.

   :param float dt: Size of the time steps. Defaults to 1.0.

   :param bool angular: If True, output angular flux; if False, output scalar flux. Defaults to False.


Optional Parameters
-------------------

   :param int edges: Flux location: 0 = cell centers, 1 = cell edges. Defaults to 0.

   :param int mg: Multigroup solver: 1 = source iteration, 2 = DMD. Defaults to 1.

   :param int dmd_k: DMD basis size. Defaults to 40.

   :param int dmd_r: DMD matrix rank. Defaults to 2.

   :param int count_nn: Angular convergence max iterations. Defaults to 100.

   :param int count_gg: Energy group convergence max iterations. Defaults to 100.

   :param int count_keff: Power iteration convergence max iterations. Defaults to 100.

   :param float change_nn: Angular convergence tolerance. Defaults to 1e-6.

   :param float change_gg: Energy group convergence tolerance. Defaults to 1e-6.

   :param float change_keff: Power iteration convergence tolerance. Defaults to 1e-6.


Examples
--------

**Time-Dependent Problem (2D)**

.. code-block:: python

   import numpy as np
   import ants
   from ants.timed2d import backward_euler

   # Geometry
   cells_x, cells_y = 50, 50
   length_x, length_y = 1.0, 1.0
   delta_x = np.repeat(length_x / cells_x, cells_x)
   delta_y = np.repeat(length_y / cells_y, cells_y)

   # Time stepping
   steps = 10
   dt = 0.1

   # Physics
   angles = 8
   groups = 7

   # Parameters
   info = {
       "cells_x": cells_x,
       "cells_y": cells_y,
       "angles": angles,
       "groups": groups,
       "materials": 1,
       "geometry": 1,    # square
       "spatial": 2,     # diamond difference
       "bc_x": [0, 0],
       "bcdim_x": 0,
       "bc_y": [0, 0],
       "bcdim_y": 0,
       "steps": steps,
       "dt": dt,
       "angular": True
   }

   # Cross sections
   xs_total = np.array([[1.0] * groups])
   xs_scatter = np.zeros((1, groups, groups))
   xs_fission = np.zeros((1, groups, groups))

   # External source
   external = np.ones((cells_x, cells_y, angles, groups))
   boundary_x = np.zeros((2, cells_y, groups))
   boundary_y = np.zeros((2, cells_x, groups))

   # Setup and solve
   angle_x, angle_y, angle_w = ants.angular_xy(info)
   medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
   flux = backward_euler(xs_total, xs_scatter, xs_fission, external,
                         boundary_x, boundary_y, medium_map,
                         delta_x, delta_y, angle_x, angle_y, angle_w, info)


Requirements for External Source Dimensions
--------------------------------------------

There are certain requirements for the external source dimensions for
running different problems. If a function is not listed, there are no restrictions.

* ``timed2d.backward_euler()``: The external source must be of size ((I x J) x N x G).

* ``hybrid2d.backward_euler()``: The uncollided external source must be of size ((I x J) x Nu x Gu). The collided external source must be of size ((I x J) x Gc).

* ``critical2d.power_iteration()``: The external source must be of size ((I x J) x G).



Requirements for Boundary Condition Dimensions
-----------------------------------------------

There are certain requirements for the boundary condition dimensions:

* ``hybrid2d.backward_euler()``: The uncollided boundary conditions (bc_x and bc_y) can be of any size. The collided boundary conditions (bc_x and bc_y) must be of size (2) with bcdim_x = bcdim_y = 0.

* ``critical2d.power_iteration()``: The boundary conditions (bc_x and bc_y) must be of size (2) with bcdim_x = bcdim_y = 0.
