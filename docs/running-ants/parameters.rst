One-Dimensional Parameters
===============================================

One-dimensional parameters are created in the Cython struct *params1d* 
and is built using a Python dictionary. These parameters are used for 
running functions from the :py:class:`timed1d`, :py:class:`fixed1d`, 
:py:class:`critical1d`, and :py:class:`hybrid1d` modules.


.. py:data:: params_dict = {"cells", "angles", "groups", "materials", \
   "geometry", "spatial", "qdim", "bc", "bcdim", "steps", "dt", \
   "angular", "adjoint"}

   Python dictionary converted to Cython struct using ``cytools_1d._to_params1d()``.

   :type: dict

   :param int cells: Number of spatial cells (I).
   
   :param int angles: Number of discrete angles (N % 2 == 0). 
   
   :param int groups: Number of energy groups (G). 
   
   :param int materials: Number of material regions (M).
   
   :param int geometry: Type of geometry (1: slab, 2: sphere). Defaults to {1: slab}.

   :param int spatial: Type of spatial discretization (1: step method, 2: diamond difference). Defaults to {2: diamond difference}.

   :param int qdim: Size of external dimension (0: [0], 1: [I], 2: [I x G], 3: [I x N x G]). 

   :param list[int] bc: Boundary conditions for [x = 0, x = X] with (0: vacuum, 1: reflective) of length 2. Defaults to [0, 0].

   :param int bcdim: Size of boundary conditions dimensions (0: [2], 1: [2 x G], 2: [2 x N x G]).

   :param int steps: Number of time steps. Defaults to 0.

   :param double dt: Size of the time steps. Defaults to 0.

   :param bool angular: If the flux output should be angular or scalar. Defaults to True.

   :param bool adjoint: If the adjoint or forward transport equation be solved. Defaults to False.


Requirements for External Source Dimensions
--------------------------------------------

There are certain requirements for the external source dimensions for 
running different problems based on implementing the 
``source_iteration_1d.multigroup_scalar()`` and the
``source_iteration_1d.multigroup_angular()`` functions. If a function is 
not listed, there are no restrictions. In the future, functions will be
implemented to verify these parameters.

* ``timed1d.backward_euler()``: The external source must be of size (I x N x G) with qdim = 3.

* ``hybrid1d.backward_euler()``: The uncollided external source must be of size (I x Nu x Gu) with qdim = 3. The collided external source must be of size (I x Gc) with qdim = 2.

* ``critical1d.power_iteration()``: The external source must be of size (I x G) with qdim = 2.



Requirements for Boundary Condition Dimensions
-----------------------------------------------

As with the external source dimensions, there are restrictions on the 
boundary condition dimensions. In the future, functions will be 
implemented to verify these parameters.

* ``hybrid1d.backward_euler()``: The uncollided boundary condition can be of any size. The collided boundary condition must be of size (2) with bcdim = 0.

* ``critical1d.power_iteration()``: The boundary condition must be of size (2) with bcdim = 0.



Two-Dimensional Parameters
===============================================

Two-dimensional parameters are created in the Cython struct *params2d* 
and is built using a Python dictionary. These parameters are used for 
running functions from the :py:class:`timed2d`, :py:class:`fixed2d`, 
:py:class:`critical2d`, and :py:class:`hybrid2d` modules.


.. py:data:: params_dict = {"cells_x", "cells_y", angles", "groups", \
   "materials", "geometry", "spatial", "qdim", "bc_x", "bcdim_x", "bc_y", \
   "bcdim_y", steps", "dt", "angular", "adjoint"}

   Python dictionary converted to Cython struct using ``cytools_2d._to_params2d()``.

   :type: dict

   :param int cells_x: Number of spatial cells in the x direction (I).

   :param int cells_y: Number of spatial cells in the y direction (J).

   :param int angles: Number of discrete angles (N % 2 == 0). 
   
   :param int groups: Number of energy groups (G). 
   
   :param int materials: Number of material regions (M).
   
   :param int geometry: Type of geometry (1: square, 2: triangle). Defaults to {1: slab}.

   :param int spatial: Type of spatial discretization (1: step method, 2: diamond difference). Defaults to {2: diamond difference}.

   :param int qdim: Size of external dimension (0: [0], 1: [(I x J)], 2: [(I x J) x G], 3: [(I x J) x N x G]). 

   :param list[int] bc_x: Boundary conditions for [x = 0, x = X] with (0: vacuum, 1: reflective) of length 2. Defaults to [0, 0].

   :param int bcdim_x: Size of boundary conditions dimensions for the x direction with (0: [2], 1: [2 x J], 2: [2 x J x G], 3: [2 x J x N x G]).

   :param list[int] bc_y: Boundary conditions for [y = 0, y = Y] with (0: vacuum, 1: reflective) of length 2. Defaults to [0, 0].

   :param int bcdim_y: Size of boundary conditions dimensions for the y direction with (0: [2], 1: [2 x I], 2: [2 x I x G], 3: [2 x I x N x G]).

   :param int steps: Number of time steps. Defaults to 0.

   :param double dt: Size of the time steps. Defaults to 0.

   :param bool angular: If the flux output should be angular or scalar. Defaults to True.

   :param bool adjoint: If the adjoint or forward transport equation be solved. Defaults to False.


Requirements for External Source Dimensions
--------------------------------------------

There are certain requirements for the external source dimensions for 
running different problems based on implementing the 
``source_iteration_2d.multigroup_scalar()`` and the
``source_iteration_2d.multigroup_angular()`` functions. If a function is 
not listed, there are no restrictions. In the future, functions will be
implemented to verify these parameters.

* ``timed2d.backward_euler()``: The external source must be of size ((I X J) x N x G) with qdim = 3.

* ``hybrid2d.backward_euler()``: The uncollided external source must be of size ((I x J) x Nu x Gu) with qdim = 3. The collided external source must be of size ((I x J) x Gc) with qdim = 2.

* ``critical2d.power_iteration()``: The external source must be of size ((I x J) x G) with qdim = 2.



Requirements for Boundary Condition Dimensions
-----------------------------------------------

As with the external source dimensions, there are restrictions on the 
boundary condition dimensions. In the future, functions will be 
implemented to verify these parameters.

* ``hybrid2d.backward_euler()``: The uncollided boundary conditions (bc_x and bc_y) can be of any size. The collided boundary conditions (bc_x and bc_y) must be of size (2) with bcdim_x = bcdim_y = 0.

* ``critical2d.power_iteration()``: The boundary conditions (bc_x and bc_y) must be of size (2) with bcdim_x = bcdim_y = 0.


.. external_source[group + angle * Groups :: Angles * Groups][cell]
.. point_source[group::Groups][angle]
