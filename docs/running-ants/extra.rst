DataTypes Reference
===============================================

All ANTS solvers accept typed dataclass objects rather than raw arrays and
dictionaries. These objects group related inputs, provide defaults, and make
the solver signatures self-documenting.

The main dataclasses are defined in ``ants.datatypes``:

.. code-block:: python

    from ants.datatypes import (
        GeometryData,
        HybridData,
        MaterialData,
        QuadratureData,
        SolverData,
        SourceData,
        TemporalDiscretization,
        TimeDependentData,
    )


MaterialData
------------

Holds cross sections and neutron velocity for a set of materials.

.. code-block:: python

    mat_data = MaterialData(
        total=xs_total,       # shape (M, G)
        scatter=xs_scatter,   # shape (M, G, G)
        fission=xs_fission,   # shape (M, G, G)
        velocity=velocity,    # shape (G,) — required for time-dependent problems
    )

For fixed-source and criticality problems ``velocity`` can be omitted.
The cross sections can be loaded from ANTS's built-in library via
``ants.materials()``:

.. code-block:: python

    mat_data = ants.materials(87, ["uranium-%0.7%", "high-density-polyethyene-087"],
                              datatype=True)
    mat_data.velocity = ants.energy_velocity(groups, edges_g)


SourceData
-----------

Holds initial conditions, external source, and boundary conditions. The
required fields depend on the problem type and time integrator.

**Fixed source / criticality (1D):**

.. code-block:: python

    sources = SourceData(
        external=np.ones((cells_x, 1, 1)),  # shape (I, N, G) or broadcast
        boundary_x=np.zeros((2, 1, 1)),     # shape (2, N, G) or broadcast
    )

**Time-dependent 1D (BDF1):**

.. code-block:: python

    sources = SourceData(
        initial_flux=np.zeros((cells_x, angles, groups)),  # shape (I, N, G)
        external=external,                                  # shape (T, I, N, G)
        boundary_x=boundary_x,                             # shape (T, 2, N, G)
    )

**Time-dependent 2D with TR-BDF2 (edge-based flux):**

.. code-block:: python

    sources = SourceData(
        initial_flux_x=np.zeros((cells_x + 1, cells_y, angles**2, groups)),
        initial_flux_y=np.zeros((cells_x, cells_y + 1, angles**2, groups)),
        external=np.zeros((1, cells_x, cells_y, 1, 1)),
        boundary_x=boundary_x,  # shape (T, 2, J, N², G)
        boundary_y=boundary_y,  # shape (T, 2, I, N², G)
    )

See :doc:`time-dependent-inputs` for the complete initial flux shape rules for
each time integrator.


GeometryData
-------------

Describes the spatial grid, material layout, and geometry type.

.. code-block:: python

    # 1D
    geometry = GeometryData(
        medium_map=medium_map,  # shape (I,), integer material indices
        delta_x=delta_x,        # shape (I,), cell widths
        bc_x=bc_x,              # [left_bc, right_bc]; 0=vacuum, 1=reflective
        geometry=1,             # 1 = 1D slab, 2 = 1D sphere
    )

    # 2D
    geometry = GeometryData(
        medium_map=medium_map,  # shape (I*J,), integer material indices
        delta_x=delta_x,        # shape (I,)
        delta_y=delta_y,        # shape (J,)
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,             # 3 = 2D slab
    )


SolverData
-----------

Controls solver behavior and output format. All fields have defaults.

.. code-block:: python

    solver = SolverData(
        angular=False,          # True → return angular flux; False → scalar flux
        spatial=2,              # 1 = step method, 2 = diamond difference
        max_iter_angular=100,
        tol_angular=1e-6,
        max_iter_energy=100,
        tol_energy=1e-6,
    )


TimeDependentData
------------------

Configures the time integration.

.. code-block:: python

    from ants.datatypes import TimeDependentData, TemporalDiscretization

    time_data = TimeDependentData(
        steps=100,
        dt=0.01,
        time_disc=TemporalDiscretization.TR_BDF2,  # default: BDF1
    )

Available ``TemporalDiscretization`` values: ``BDF1``, ``BDF2``, ``CN``, ``TR_BDF2``.


HybridData
-----------

Required for the standard collision-based hybrid solvers (``hybrid1d``,
``hybrid2d``). Generated from the group indexing arrays:

.. code-block:: python

    from ants.utils import hybrid as hytools

    fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
    hybrid_data = HybridData(fine_idx=fine_idx, coarse_idx=coarse_idx, factor=factor)

Not needed for the vectorized hybrid (``vhybrid1d``, ``vhybrid2d``).


Complete Example: Fixed Source 1D
-----------------------------------

.. code-block:: python

    import numpy as np
    import ants
    from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
    from ants.fixed1d import fixed_source

    cells_x = 100
    angles = 4
    groups = 1
    bc_x = [0, 0]

    length = 1.0
    delta_x = np.repeat(length / cells_x, cells_x)
    edges_x = np.linspace(0, length, cells_x + 1)

    edges_g, _ = ants.energy_grid(None, groups)

    medium_map = np.zeros(cells_x, dtype=np.int32)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    quadrature = ants.angular_x(angles, bc_x=bc_x)

    mat_data = MaterialData(total=xs_total, scatter=xs_scatter, fission=xs_fission)
    sources = SourceData(
        external=np.ones((cells_x, 1, 1)),
        boundary_x=np.zeros((2, 1, 1)),
    )
    geometry = GeometryData(medium_map=medium_map, delta_x=delta_x, bc_x=bc_x, geometry=1)
    solver = SolverData(angular=True)

    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
    # flux shape: (I, N, G) if angular=True, else (I, G)


Complete Example: Criticality 1D
----------------------------------

.. code-block:: python

    import numpy as np
    import ants
    from ants.datatypes import GeometryData, SolverData
    from ants.critical1d import k_criticality

    cells_x = 200
    angles = 8
    groups = 87
    bc_x = [1, 0]  # reflective at origin, vacuum at surface

    length = 10.0
    delta_x = np.repeat(length / cells_x, cells_x)
    edges_x = np.linspace(0, length, cells_x + 1)
    edges_g, _ = ants.energy_grid(87, groups)

    layers = [[0, "uranium-%20%", "0-4"],
              [1, "uranium-%0%",  "4-6"],
              [2, "stainless-steel-440", "6-10"]]
    medium_map = ants.spatial1d(layers, edges_x)

    quadrature = ants.angular_x(angles, bc_x=bc_x)
    mat_data = ants.materials(87, np.array(layers)[:, 1], datatype=True)

    geometry = GeometryData(medium_map=medium_map, delta_x=delta_x, bc_x=bc_x, geometry=2)
    solver = SolverData()

    flux, keff = k_criticality(mat_data, geometry, quadrature, solver)


Complete Example: Time-Dependent 2D
--------------------------------------

.. code-block:: python

    import numpy as np
    import ants
    from ants.datatypes import (
        GeometryData, MaterialData, SolverData, SourceData,
        TemporalDiscretization, TimeDependentData,
    )
    from ants.timed2d import time_dependent

    cells_x = cells_y = 50
    angles = 4
    groups = 1
    steps = 10
    dt = 0.1
    bc_x = bc_y = [0, 0]

    delta_x = np.repeat(1.0 / cells_x, cells_x)
    delta_y = np.repeat(1.0 / cells_y, cells_y)
    edges_g, _ = ants.energy_grid(None, groups)
    velocity = ants.energy_velocity(groups, edges_g)

    medium_map = np.zeros(cells_x * cells_y, dtype=np.int32)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

    mat_data = MaterialData(total=xs_total, scatter=xs_scatter,
                            fission=xs_fission, velocity=velocity)
    sources = SourceData(
        initial_flux=np.zeros((cells_x, cells_y, angles**2, groups)),
        external=np.zeros((1, cells_x, cells_y, 1, 1)),
        boundary_x=np.zeros((1, 2, 1, 1, 1)),
        boundary_y=np.zeros((1, 2, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=medium_map, delta_x=delta_x, delta_y=delta_y,
        bc_x=bc_x, bc_y=bc_y, geometry=3,
    )
    solver = SolverData()
    time_data = TimeDependentData(steps=steps, dt=dt)

    flux = time_dependent(mat_data, sources, geometry, quadrature, solver, time_data)
    # flux shape: (T, I, J, G)
