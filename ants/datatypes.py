########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Custom datatypes to reduce argument count in solver functions
#
########################################################################

"""Typed data containers and enums used by ANTS solver interfaces.

This module groups commonly paired arrays and scalar controls into dataclasses
so solver call signatures remain compact and easier to reason about.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np


class SpatialDiscretization(IntEnum):
    """Supported spatial discretizations.

    Attributes
    ----------
    STEP : int
        Step differencing.
    DIAMOND : int
        Diamond differencing.
    STEP_CHARACTERISTIC : int
        Step-characteristic differencing.
    """

    STEP = 1
    DIAMOND = 2
    STEP_CHARACTERISTIC = 3


class TemporalDiscretization(IntEnum):
    """Supported temporal discretizations.

    Attributes
    ----------
    BDF1 : int
        Backward Euler (BDF1).
    CN : int
        Crank-Nicolson.
    BDF2 : int
        Second-order backward differentiation (BDF2).
    TR_BDF2 : int
        Trapezoidal-BDF2 (TR-BDF2).
    """

    BDF1 = 1
    CN = 2
    BDF2 = 3
    TR_BDF2 = 4


class Geometry(IntEnum):
    """Supported transport geometries.

    Attributes
    ----------
    SLAB1D : int
        One-dimensional slab geometry.
    SPHERE1D : int
        One-dimensional spherical geometry.
    SLAB2D : int
        Two-dimensional slab (Cartesian) geometry.
    """

    SLAB1D = 1
    SPHERE1D = 2
    SLAB2D = 3


class BoundaryCondition(IntEnum):
    """Supported boundary conditions.

    Attributes
    ----------
    VACUUM : int
        Vacuum (zero incoming flux).
    REFLECTED : int
        Reflected (mirror) boundary.
    """

    VACUUM = 0
    REFLECTED = 1


class MultigroupSolver(IntEnum):
    """Supported multigroup solver types.

    Attributes
    ----------
    SOURCE_ITERATION : int
        Standard source-iteration solver.
    DMD : int
        Dynamic mode decomposition (DMD)-accelerated solver.
    """

    SOURCE_ITERATION = 1
    DMD = 2


class ParallelType(IntEnum):
    """Parallelism strategy for OpenMP sweeps.

    Attributes
    ----------
    ANGLE : int
        Parallelize over angles (default).  The inner angular sweep uses
        ``num_threads`` threads. Energy groups are swept sequentially with
        Gauss-Seidel ordering.
    GROUP : int
        Parallelize over energy groups using a Jacobi outer iteration.
        Each group's angular sweep runs on a single thread.  Best when
        ``groups`` >> ``angles``.
    BOTH : int
        Parallelize over both energy groups (Jacobi) and angles
        simultaneously.  Both the outer group prange and the inner angle
        prange use ``num_threads`` threads.  Requires
        ``OMP_MAX_ACTIVE_LEVELS=2`` (or ``OMP_NESTED=TRUE``) in the
        environment for true nested parallelism; otherwise the inner
        prange is serialized by the OpenMP runtime.
    """

    ANGLE = 1
    GROUP = 2
    BOTH = 3


def _default_vacuum_bc():
    """Return default two-sided vacuum boundary conditions.

    Returns
    -------
    list[int]
        Boundary conditions ``[VACUUM, VACUUM]`` for left/right (or
        bottom/top) sides.
    """

    return [BoundaryCondition.VACUUM, BoundaryCondition.VACUUM]


@dataclass
class MaterialData:
    """Bundle of material cross-section arrays.

    Attributes
    ----------
    total : numpy.ndarray
        Total macroscopic cross section, shape ``(materials, groups)``.
    scatter : numpy.ndarray
        Scattering cross section, shape ``(materials, groups, groups)``.
    fission : numpy.ndarray
        Fission cross section, shape ``(materials, groups)`` if ``chi`` is
        ``None``, else
        ``(materials, groups, groups)``.
    chi : numpy.ndarray, optional
        Fission spectrum, shape ``(materials, groups)``.
    velocity : numpy.ndarray, optional
        Neutron velocity, shape ``(groups,)``.
    """

    total: np.ndarray
    scatter: np.ndarray
    fission: np.ndarray
    chi: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None


@dataclass
class SourceData:
    """Bundle of fixed source arrays.

    Attributes
    ----------
    external : numpy.ndarray
        External source, shape ``(materials, groups)``.
    boundary_x : numpy.ndarray
        Boundary source in x direction, shape ``(materials, groups)``.
    boundary_y : numpy.ndarray, optional
        Boundary source in y direction for 2D problems, shape
        ``(materials, groups)``.
    initial_flux : numpy.ndarray, optional
        Initial flux for time-dependent problems, shape ``(cells_x, cells_y, groups)``.
    initial_flux_x : numpy.ndarray, optional
        Initial flux at x vertical faces for time-dependent problems, shape
        ``(cells_x + 1, cells_y, groups)``.
    initial_flux_y : numpy.ndarray, optional
        Initial flux at y horizontal faces for time-dependent problems, shape
        ``(cells_x, cells_y + 1, groups)``.
    """

    external: np.ndarray
    boundary_x: np.ndarray
    boundary_y: Optional[np.ndarray] = None
    initial_flux: Optional[np.ndarray] = None
    initial_flux_x: Optional[np.ndarray] = None
    initial_flux_y: Optional[np.ndarray] = None


@dataclass
class QuadratureData:
    """Angular quadrature angles and weights.

    Attributes
    ----------
    angle_x : numpy.ndarray
        Direction cosines in x, shape ``(angles,)``.
    angle_w : numpy.ndarray
        Angular weights, shape ``(angles,)``.
    angle_y : numpy.ndarray, optional
        Direction cosines in y for 2D problems, shape ``(angles,)``.
    P : numpy.ndarray, optional
        Precomputed Legendre polynomials, shape ``(L + 1, angles)``.
    P_weights : numpy.ndarray, optional
        Preweighted Legendre polynomials, shape ``(L + 1, angles)``.
    """

    angle_x: np.ndarray
    angle_w: np.ndarray
    angle_y: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    P_weights: Optional[np.ndarray] = None


@dataclass
class GeometryData:
    """Spatial cell widths.

    Attributes
    ----------
    medium_map : numpy.ndarray
        Maps each spatial cell to a material index.
    delta_x : numpy.ndarray
        Cell widths in x, shape ``(cells_x,)``.
    delta_y : numpy.ndarray, optional
        Cell widths in y for 2D problems, shape ``(cells_y,)``.
    bc_x : list[int], optional
        Left/right boundary conditions as ``BoundaryCondition`` values.
    bc_y : list[int], optional
        Bottom/top boundary conditions (2D only) as
        ``BoundaryCondition`` values.
    geometry : Geometry
        Geometry type.
    space_disc : SpatialDiscretization
        Spatial discretization type.
    """

    medium_map: np.ndarray
    delta_x: np.ndarray
    delta_y: Optional[np.ndarray] = None
    bc_x: Optional[List[int]] = field(default_factory=_default_vacuum_bc)
    bc_y: Optional[List[int]] = field(default_factory=_default_vacuum_bc)
    geometry: Geometry = Geometry.SLAB1D
    space_disc: SpatialDiscretization = SpatialDiscretization.DIAMOND


@dataclass
class HybridData:
    """Coarse-to-fine group mapping for hybrid solvers.

    Attributes
    ----------
    fine_idx : numpy.ndarray
        Fine-group indices, shape ``(groups_fine,)``.
    coarse_idx : numpy.ndarray
        Coarse-group index for each fine group, shape ``(groups_fine,)``.
    factor : numpy.ndarray
        Coarse-to-fine scaling factor, shape ``(groups_fine,)``.
    """

    fine_idx: np.ndarray
    coarse_idx: np.ndarray
    factor: np.ndarray


@dataclass
class TimeDependentData:
    """Time-dependent problem data.

    Attributes
    ----------
    steps : int
        Number of time steps.
    dt : float
        Time step width.
    time_disc : TemporalDiscretization
        Temporal discretization type.
    """

    steps: int = 0
    dt: float = 1.0
    time_disc: TemporalDiscretization = TemporalDiscretization.BDF1


@dataclass
class SolverData:
    """Bundle of solver parameters and data arrays.

    Attributes
    ----------
    angular : bool
        If True, return angular flux instead of scalar flux.
    flux_at_edges : int
        Flux location: 0 = cell centers, 1 = cell edges.
    num_threads : int
        Number of OpenMP threads for angular sweeps.  Default is 1 (no
        parallelism).  Set to 0 to use all logical CPUs, or any positive
        integer to use that many threads.
    parallel : ParallelType
        Parallelism strategy.  ``ANGLE`` (default) parallelizes over
        angles with Gauss-Seidel energy iteration.  ``GROUP`` parallelizes
        over energy groups with Jacobi iteration (single-threaded angle
        sweep per group).  ``BOTH`` runs Jacobi group prange and angle
        prange simultaneously. Requires ``OMP_MAX_ACTIVE_LEVELS=2`` for
        true nested parallelism.
    mg_solver : MultigroupSolver
        Multigroup solver type.
    dmd_snapshots : int
        Number of DMD snapshots.
    dmd_rank : int
        Number of DMD rank-1 updates before extrapolation.
    sigma_as : float
        Artificial scattering strength (0 disables).
    beta_as : float
        Artificial scattering kernel width parameter.
    max_iter_angular : int
        Maximum angular (within-group) iterations.
    max_iter_energy : int
        Maximum energy-group iterations.
    max_iter_keff : int
        Maximum power iterations for k-eigenvalue problems.
    tol_angular : float
        Angular convergence tolerance.
    tol_energy : float
        Energy-group convergence tolerance.
    tol_keff : float
        k-eigenvalue convergence tolerance.
    """

    angular: bool = False
    flux_at_edges: int = 0
    num_threads: int = 1
    parallel: ParallelType = ParallelType.ANGLE
    mg_solver: MultigroupSolver = MultigroupSolver.SOURCE_ITERATION
    dmd_snapshots: int = 20
    dmd_rank: int = 2
    sigma_as: float = 0.0
    beta_as: float = 4.5
    max_iter_angular: int = 100
    max_iter_energy: int = 100
    max_iter_keff: int = 100
    tol_angular: float = 1e-12
    tol_energy: float = 1e-08
    tol_keff: float = 1e-06


@dataclass
class ProblemParameters:
    """Scalar parameters for a neutron transport problem.

    Attributes
    ----------
    cells_x : int
        Number of spatial cells in the x direction.
    cells_y : int
        Number of spatial cells in the y direction (2D only).
    angles : int
        Number of discrete ordinates (must be even).
    groups : int
        Number of energy groups.
    materials : int
        Number of distinct materials.
    geometry : Geometry
        Geometry type.
    space_disc : SpatialDiscretization
        Spatial discretization type.
    bc_x : list[int]
        Left/right boundary conditions as ``BoundaryCondition`` values.
    bc_y : list[int]
        Bottom/top boundary conditions (2D only) as
        ``BoundaryCondition`` values.
    steps : int
        Number of time steps (time-dependent problems).
    dt : float
        Time step width (time-dependent problems).
    time_disc : TemporalDiscretization
        Temporal discretization type (time-dependent problems).
    angular : bool
        If True, return angular flux instead of scalar flux.
    flux_at_edges : int
        Flux location: 0 = cell centers, 1 = cell edges.
    num_threads : int
        Number of OpenMP threads for angular sweeps.
    parallel_type : ParallelType
        Parallelism strategy (ANGLE, GROUP, or BOTH).
    mg_solver : MultigroupSolver
        Multigroup solver type.
    dmd_snapshots : int
        Number of DMD snapshots.
    dmd_rank : int
        Number of DMD rank-1 updates before extrapolation.
    sigma_as : float
        Artificial scattering strength (0 disables).
    beta_as : float
        Artificial scattering kernel width parameter.
    max_iter_angular : int
        Maximum angular (within-group) iterations.
    max_iter_energy : int
        Maximum energy-group iterations.
    max_iter_keff : int
        Maximum power iterations for k-eigenvalue problems.
    tol_angular : float
        Angular convergence tolerance.
    tol_energy : float
        Energy-group convergence tolerance.
    tol_keff : float
        k-eigenvalue convergence tolerance.
    """

    cells_x: int
    cells_y: int
    angles: int
    groups: int
    materials: int
    geometry: Geometry
    space_disc: SpatialDiscretization
    bc_x: List[int]
    bc_y: List[int]
    steps: int
    dt: float
    time_disc: TemporalDiscretization
    angular: bool
    flux_at_edges: int
    num_threads: int
    parallel_type: ParallelType
    mg_solver: MultigroupSolver
    dmd_snapshots: int
    dmd_rank: int
    sigma_as: float
    beta_as: float
    max_iter_angular: int
    max_iter_energy: int
    max_iter_keff: int
    tol_angular: float
    tol_energy: float
    tol_keff: float


def create_params(materials, quadrature, geometry, solver, time=None):
    """Create a :class:`ProblemParameters` instance from data bundles.

    Parameters
    ----------
    materials : MaterialData
        Material cross sections and related arrays.
    quadrature : QuadratureData
        Angular quadrature arrays.
    geometry : GeometryData
        Spatial discretization data and boundary conditions.
    solver : SolverData
        Solver control options and convergence tolerances.
    time : TimeDependentData, optional
        Time-stepping data. If ``None``, defaults to ``TimeDependentData()``.

    Returns
    -------
    ProblemParameters
        Flattened scalar and enum parameters consumed by low-level solvers.

    """

    time_data = time if time is not None else TimeDependentData()
    angles = (
        quadrature.angle_x.size
        if quadrature.angle_y is None
        else int(np.sqrt(quadrature.angle_x.size))
    )

    return ProblemParameters(
        cells_x=geometry.delta_x.size,
        cells_y=geometry.delta_y.size if geometry.delta_y is not None else 1,
        angles=angles,
        groups=materials.total.shape[1],
        materials=materials.total.shape[0],
        geometry=geometry.geometry,
        space_disc=geometry.space_disc,
        bc_x=geometry.bc_x,
        bc_y=geometry.bc_y,
        steps=time_data.steps,
        dt=time_data.dt,
        time_disc=time_data.time_disc,
        angular=solver.angular,
        flux_at_edges=solver.flux_at_edges,
        num_threads=solver.num_threads,
        parallel_type=solver.parallel,
        mg_solver=solver.mg_solver,
        dmd_snapshots=solver.dmd_snapshots,
        dmd_rank=solver.dmd_rank,
        sigma_as=solver.sigma_as,
        beta_as=solver.beta_as,
        max_iter_angular=solver.max_iter_angular,
        max_iter_energy=solver.max_iter_energy,
        max_iter_keff=solver.max_iter_keff,
        tol_angular=solver.tol_angular,
        tol_energy=solver.tol_energy,
        tol_keff=solver.tol_keff,
    )
