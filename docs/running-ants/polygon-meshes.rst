Polygon Meshes
===============================================

Two-dimensional problems can use triangles, hexagons, or any mesh of convex
polygons in place of the Cartesian ``SLAB2D`` grid. The sweep is described in
:ref:`the polygon mesh section <sec-nte-polygon>`. Polygon meshes work with the
fixed source, criticality, time-dependent, hybrid, and nearby problem solvers,
and with the step, diamond difference, and step characteristic methods.


Building a Mesh
---------------

A mesh is stored in a ``PolygonMesh`` object, which holds the connectivity
(cells, faces, normals, areas) needed by the sweep. Meshes are made with the
functions in ``ants.utils.mesh2d``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Mesh
   * - ``rectangle_mesh(edges_x, edges_y)``
     - Rectangles, the same cells as ``SLAB2D``.
   * - ``triangle_mesh(edges_x, edges_y, pattern)``
     - Rectangles split into triangles, ``"banded"``, ``"union_jack"``, or
       ``"crossed"``.
   * - ``triangle_mesh_delaunay(points)``
     - Delaunay triangulation of a set of points.
   * - ``hexagon_mesh(pitch, boundary, orientation, clip)``
     - Hexagonal lattice clipped to a convex boundary. ``clip=False`` keeps
       only whole hexagons.
   * - ``polygon_mesh(vertices, cells)``
     - Any mesh of convex polygons from vertex coordinates and the vertex
       indices of each cell.

The boundary of the mesh is split into sides, each with its own boundary
condition. Rectangular domains have four sides ordered as
:math:`[x = 0, x = X, y = 0, y = Y]`, and hexagonal domains from
``hexagon_boundary`` have six sides in counter-clockwise order. The number of
sides is ``mesh.sides`` and the side of each boundary face is
``mesh.face_side[mesh.boundary_faces]``.

.. autofunction:: ants.utils.mesh2d.polygon_mesh

.. autofunction:: ants.utils.mesh2d.triangle_mesh

.. autofunction:: ants.utils.mesh2d.hexagon_mesh

.. autofunction:: ants.utils.mesh2d.hexagon_boundary


Material Locations
------------------

Materials are assigned to cells at the cell centroids with
``polygon_medium_map``, which uses the same shapes as :doc:`medium-map` with the
addition of hexagons. For material fractions in each cell, use
``polygon_weight_matrix``.

.. autofunction:: ants.utils.mesh2d.polygon_medium_map


Input Shapes
------------

A polygon mesh with :math:`C` cells, :math:`F` faces, and :math:`F_b` boundary
faces is passed to the 2D solvers as a :math:`(C \times 1)` grid, so the
``medium_map`` and cell arrays have a dummy second axis.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Input
     - Shape
     - Notes
   * - ``medium_map``
     - :math:`(C, 1)`
     - From ``polygon_medium_map``.
   * - ``external``
     - :math:`(C, 1, N^2, G)`
     - Broadcast as in :doc:`standard-inputs`.
   * - ``boundary_x``
     - :math:`(1, F_b, N^2, G)`
     - One value per boundary face, ordered as ``mesh.boundary_faces``.
   * - ``boundary_y``
     - :math:`(1, 1, 1, 1)`
     - Placeholder.
   * - ``bc_sides``
     - length :math:`S`
     - 0 for vacuum, 1 for reflected, one per side.

The ``delta_x`` and ``delta_y`` inputs are not needed. Time-dependent sources
have a leading time axis, and the boundary sources only use the incoming angles
(:math:`\bsOmega \cdot \mathbf{n} < 0`) of each face. For Crank-Nicolson and
TR-BDF2, ``initial_flux_x`` holds the face fluxes :math:`(F, 1, N^2, G)` and
``initial_flux_y`` holds the cell fluxes :math:`(C, 1, N^2, G)`.

The solvers return the cell flux with shape :math:`(C, 1, G)`, or
:math:`(C, 1, N^2, G)` with ``SolverData(angular=True)``. With
``SolverData(flux_at_edges=1)``, the fixed source solver returns the face fluxes
:math:`(F, G)`.


Reflected Sides
---------------

A reflected side needs the reflected direction of every outgoing angle to be in
the quadrature. Sides along the :math:`x` and :math:`y` axes work with any
``angular_xy`` quadrature. The sides of a hexagonal domain are at multiples of
60 degrees, which need the number of angles to be a multiple of six, such as
``angular_xy(6)`` or ``angular_xy(12)``. An error is raised if a reflected
direction is missing.


Examples
--------

**Hexagonal assembly criticality.** A hexagonal domain with reflected sides and
seven fuel pins.

.. code-block:: python

    import numpy as np

    import ants
    from ants import critical2d
    from ants.datatypes import (
        Geometry,
        GeometryData,
        MaterialData,
        SolverData,
        SpatialDiscretization,
    )
    from ants.utils import mesh2d

    # Hexagonal domain (6 cm flat-to-flat) of 0.25 cm hexagons
    boundary = mesh2d.hexagon_boundary((0.0, 0.0), 6.0, "pointy")
    mesh = mesh2d.hexagon_mesh(0.25, boundary)

    # Fuel pins (material 1) in moderator (material 0)
    centers = [(0.0, 0.0), (1.5, 0.0), (-1.5, 0.0), (0.75, 1.3), \
               (-0.75, 1.3), (0.75, -1.3), (-0.75, -1.3)]
    medium_map = mesh2d.polygon_medium_map(
        mesh,
        default=0,
        circles=[[center, (0.0, 0.5)] for center in centers],
        circle_index=[1] * len(centers),
    )

    mat_data = MaterialData(
        total=np.array([[1.0], [1.5]]),
        scatter=np.array([[[0.9]], [[0.5]]]),
        fission=np.array([[[0.0]], [[0.9]]]),
    )
    geometry = GeometryData(
        medium_map=medium_map,
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=[1] * mesh.sides,
    )

    # Reflected 60 degree sides need a multiple of six angles
    quadrature = ants.angular_xy(6)
    flux, keff = critical2d.k_criticality(mat_data, geometry, quadrature, SolverData())

**Triangles with a boundary source.** A source on the :math:`x = 0` side with
reflected :math:`y` sides.

.. code-block:: python

    from ants import fixed2d
    from ants.datatypes import SourceData

    edges_x = np.linspace(0, 2.0, 21)
    edges_y = np.linspace(0, 1.0, 11)
    mesh = mesh2d.triangle_mesh(edges_x, edges_y, "union_jack")

    # Boundary source of 1 on the faces of side 0 (x = 0)
    boundary_x = np.zeros((1, mesh.boundary_faces.size, 1, 1))
    boundary_x[0, mesh.face_side[mesh.boundary_faces] == 0] = 1.0
    sources = SourceData(
        external=np.zeros((mesh.cells, 1, 1, 1)),
        boundary_x=boundary_x,
        boundary_y=np.zeros((1, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.DIAMOND,
        mesh=mesh,
        bc_sides=[0, 0, 1, 1],
    )
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.zeros((1, 1, 1)),
    )
    quadrature = ants.angular_xy(4)
    flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, SolverData())
