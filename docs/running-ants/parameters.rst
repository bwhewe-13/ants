Parameters
===============================================

Parameters are a list of integers that have a specific order to run ANTS
using Cython. This list is automatically generated when using the .inp 
files, however to get a better understanding, the order is below. 

.. table:: Variable: *params*
   :widths: grid
   :align: center

   +-------+---------------------------+-----------------+
   | Index | Argument Name             | Argument Values |
   +=======+===========================+=================+
   | 0     | One Dimensional Geometry  | 1 or 2          |
   +-------+---------------------------+-----------------+
   | 1     | Spatial Discretization    | 1 or 2          |
   +-------+---------------------------+-----------------+
   | 2     | Boundary Condition        | 0 or 1          |
   +-------+---------------------------+-----------------+
   | 3     | External Groups Index     | 1 or G          |
   +-------+---------------------------+-----------------+
   | 4     | External Angles Index     | 1 or N          |
   +-------+---------------------------+-----------------+
   | 5     | Point Source Location     | :math:`[0, I]`  |
   +-------+---------------------------+-----------------+
   | 6     | Point Source Groups Index | 1 or G          |
   +-------+---------------------------+-----------------+
   | 7     | Temporal Discretization   | 1 or 2          |
   +-------+---------------------------+-----------------+
   | 8     | Time Steps                | :math:`> 0`     |
   +-------+---------------------------+-----------------+   

| **One Dimensional Geometry**: Equals 1 when slab geometry and equals 
   2 when a spherical geometry.

| **Spatial Discretization**: Equals 1 when using the step method and 
   equals 2 when using diamond difference.

| **Boundary Condition**: Equals 0 when using a vacuum boundary condition 
   and equals 1 when using a reflected boundary condition.

| **External Groups Index**: Equals 1 when external source is of size 
   :math:`(I)`, while equals G when external source is of size 
   :math:`(I \times G)` or size :math:`(I \times N \times G)`.

| **External Angles Index**: Equals 1 when external source is of size 
   :math:`(I)` or size :math:`(I \times G)`, while equals N when 
   external source is of size :math:`(I \times N \times G)`.

| **Point Source Location**: Is any location between (and including) the
   left hand boundary (x = 0) and the right hand boundary (x = I).

| **Point Source Groups Index**: Equals 1 when point source is of size
   :math:`(N)`, while equals G when point source is of size 
   :math:`(N \times G)`.

| **Temporal Discretization**: *Optional for time-independent problems*.
   Equals 1 when using Backward Euler's and equals 2 when using Backward 
   Differentiation Formula 2 (BDF2).

| **Time Steps**: *Optional for time-independent problems*. The number of 
   time steps, must be greater than 0.


Indexing Different Source Dimensions
------------------------------------

In order to save data, both the external source and the point source are 
flattened using the ``numpy.flatten()`` function call. The issue is that 
the external source can be spatial, spatial and energy, or spatial, 
energy, and angular dependent, while the point source can be angular or 
angular and energy dependent. To account for these differences, an 
indexing scheme is used for the different cases below. For the external
source, *Groups* and *Angles* represent the External Groups Index and 
the External Angles Index, respectively. The external source is then 
indexed as ``external_source[group + angle * Groups :: Angles * Groups][cell]``
Likewise, the point source uses *Groups* as the Point Source Groups Index
and is indexed as ``point_source[group::Groups][angle]``.

.. table:: External Source Indexing
   :widths: grid
   :align: center
   
   +-------------+-------+--------+-------+--------+
   | Dimension   | group | Groups | angle | Angles |
   +=============+=======+========+=======+========+
   | \(I\)       | 0     | 1      | 0     | 1      |
   +-------------+-------+--------+-------+--------+
   | (I x G)     | g     | G      | 0     | 1      |
   +-------------+-------+--------+-------+--------+
   | (I x N x G) | g     | G      | n     | N      |
   +-------------+-------+--------+-------+--------+


.. table:: Point Source Indexing
   :widths: grid
   :align: center
   
   +-----------+-------+--------+
   | Dimension | group | Groups |
   +===========+=======+========+
   | \(N\)     | 0     | 1      |
   +-----------+-------+--------+
   | (N x G)   | g     | G      |
   +-----------+-------+--------+

