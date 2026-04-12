########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for OpenMP parallel angular sweeps.
#
# Correctness tests: verify that num_threads > 1 gives the same flux
# as num_threads = 1 for all four sweep types (slab fixed-source,
# sphere fixed-source, 2D fixed-source, time-dependent 1D).
#
# Speedup tests: verify that a multi-threaded run is meaningfully faster
# than a single-threaded run on a problem large enough to show parallelism
# (skipped if only one CPU is available).
#
########################################################################

import os
import time

import numpy as np
import pytest

import ants
from ants.datatypes import (
    GeometryData,
    MaterialData,
    ParallelType,
    SolverData,
    SourceData,
    TimeDependentData,
)
from ants.fixed1d import fixed_source as fixed_source_1d
from ants.fixed2d import fixed_source as fixed_source_2d
from ants.timed1d import time_dependent as timed_1d
from tests import problems1d, problems2d

N_CPUS = os.cpu_count()

# Under pytest-xdist (-n auto), multiple worker processes run simultaneously
# and compete for the same CPU cores.  Timing-based speedup tests are
# unreliable in that environment and are skipped automatically.
_UNDER_XDIST = os.environ.get("PYTEST_XDIST_WORKER") is not None


########################################################################
# Helpers
########################################################################


def _solver_1t():
    """SolverData pinned to 1 thread."""
    return SolverData(num_threads=1)


def _solver_nt(n=N_CPUS, parallel=ParallelType.ANGLE):
    """SolverData using n threads (default = all CPUs, angle parallelism)."""
    return SolverData(num_threads=n, parallel=parallel)


def _atol(spatial):
    return 1e-10 if spatial == 2 else 1e-10


########################################################################
# Correctness – 1D slab fixed source
########################################################################

SPATIAL = [1, 2, 3]


@pytest.mark.smoke
@pytest.mark.parametrize("spatial", SPATIAL)
def test_slab_correctness(spatial):
    """Parallel slab sweep matches single-threaded result."""
    mat_data, sources, geo, quadrature, _ = problems1d.manufactured_ss_03(200, 8)
    geo.space_disc = spatial

    solver_1 = _solver_1t()
    solver_n = _solver_nt()

    flux_1 = fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
    flux_n = fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    assert np.allclose(
        flux_1, flux_n, atol=1e-10
    ), f"Slab parallel flux differs from serial (spatial={spatial})"


########################################################################
# Correctness – 1D sphere fixed source
########################################################################


@pytest.mark.smoke
@pytest.mark.parametrize("spatial", [1, 2])
def test_sphere_correctness(spatial):
    """Sphere sweep with num_threads > 1 matches serial (sphere is sequential)."""
    mat_data, sources, geo, quadrature, solver, _ = problems1d.sphere_01("fixed")

    solver_1 = SolverData(num_threads=1, tol_angular=1e-12, max_iter_angular=500)
    solver_n = SolverData(num_threads=N_CPUS, tol_angular=1e-12, max_iter_angular=500)
    geo.space_disc = spatial

    flux_1 = fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
    flux_n = fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    err = "Sphere flux differs between num_threads=1 and"
    err += f" num_threads={N_CPUS} (spatial={spatial})"
    assert np.allclose(flux_1, flux_n, atol=1e-10), err


########################################################################
# Correctness – 2D slab fixed source
########################################################################


@pytest.mark.smoke
def test_2d_correctness():
    """Parallel 2D sweep matches single-threaded result."""
    mat_data, sources, geo, quadrature, _, _, _ = problems2d.manufactured_ss_01(40, 4)

    solver_1 = _solver_1t()
    solver_n = _solver_nt()

    flux_1 = fixed_source_2d(mat_data, sources, geo, quadrature, solver_1)
    flux_n = fixed_source_2d(mat_data, sources, geo, quadrature, solver_n)

    assert np.allclose(
        flux_1, flux_n, atol=1e-10
    ), "2D parallel flux differs from serial"


########################################################################
# Correctness – 1D time-dependent slab
########################################################################


@pytest.mark.smoke
def test_timed_1d_correctness():
    """Parallel time-dependent slab sweep matches single-threaded result."""
    mat_data, sources, geo, quadrature, _, time_data = problems1d.sphere_01("timed")

    # Use slab geometry for this timing test (sphere is separate above)
    mat_data, sources, geo, quadrature, _ = problems1d.manufactured_ss_03(100, 8)
    time_data = TimeDependentData(steps=5, dt=0.1)
    mat_data.velocity = np.ones(1)
    sources = SourceData(
        initial_flux=np.zeros((100, 8, 1)),
        external=ants.external1d.time_dependence_constant(sources.external),
        boundary_x=sources.boundary_x[None, ...],
    )

    solver_1 = _solver_1t()
    solver_n = _solver_nt()

    flux_1 = timed_1d(mat_data, sources, geo, quadrature, solver_1, time_data)
    flux_n = timed_1d(mat_data, sources, geo, quadrature, solver_n, time_data)

    assert np.allclose(
        flux_1, flux_n, atol=1e-10
    ), "Time-dependent parallel flux differs from serial"


########################################################################
# Correctness – num_threads=0 resolves to all CPUs without error
########################################################################


@pytest.mark.smoke
def test_zero_threads_default():
    """num_threads=0 (default) runs successfully and matches serial."""
    mat_data, sources, geo, quadrature, _ = problems1d.manufactured_ss_01(100, 4)

    solver_default = SolverData(num_threads=0)  # should resolve to cpu_count
    solver_1 = _solver_1t()

    flux_default = fixed_source_1d(mat_data, sources, geo, quadrature, solver_default)
    flux_1 = fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)

    assert np.allclose(
        flux_default, flux_1, atol=1e-10
    ), "Default (num_threads=0) flux differs from serial"


########################################################################
# Speedup tests – skipped when only 1 CPU is available
########################################################################


@pytest.mark.skipif(N_CPUS < 2, reason="Speedup test requires at least 2 CPUs")
@pytest.mark.skipif(
    _UNDER_XDIST, reason="Speedup tests unreliable under pytest-xdist (-n auto)"
)
def test_slab_speedup():
    """Multi-threaded slab sweep is faster than single-threaded on a large problem."""
    N_ANGLES = 16
    mat_data, sources, geo, quadrature, _ = problems1d.manufactured_ss_03(
        10000, N_ANGLES
    )

    # Use 4 threads: beyond N_ANGLES/4 the OpenMP barrier cost dominates the
    # gain from parallelising 16 angles, so a modest thread count gives the
    # best and most reproducible speedup.
    n_threads = min(N_CPUS, 4)
    solver_1 = _solver_1t()
    solver_n = SolverData(num_threads=n_threads)

    # Warm up, then take the minimum of several timed runs to reduce noise.
    for _ in range(2):
        fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
        fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    def _tmin(solver, reps=5):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fixed_source_1d(mat_data, sources, geo, quadrature, solver)
            best = min(best, time.perf_counter() - t0)
        return best

    t_serial = _tmin(solver_1)
    t_parallel = _tmin(solver_n)

    speedup = t_serial / t_parallel
    # Require at least 1.5× speedup when multiple CPUs are available.
    # This is intentionally modest to stay robust across CI environments.
    assert speedup >= 1.5, (
        f"Slab parallel speedup {speedup:.2f}× is below 1.5× threshold "
        f"(serial={t_serial:.3f}s, parallel={t_parallel:.3f}s, threads={n_threads})"
    )


@pytest.mark.skipif(N_CPUS < 2, reason="Speedup test requires at least 2 CPUs")
@pytest.mark.skipif(
    _UNDER_XDIST, reason="Speedup tests unreliable under pytest-xdist (-n auto)"
)
def test_2d_speedup():
    """Multi-threaded 2D sweep is faster than single-threaded on a large problem."""
    N_ANGLES = 8  # N² = 64 discrete ordinates
    mat_data, sources, geo, quadrature, _, _, _ = problems2d.manufactured_ss_01(
        120, N_ANGLES
    )

    # 8 threads gives near-linear scaling for 64 angles without excessive
    # barrier overhead.
    n_threads = min(N_CPUS, 8)
    solver_1 = _solver_1t()
    solver_n = SolverData(num_threads=n_threads)

    # Warm up, then take the minimum of several timed runs.
    for _ in range(2):
        fixed_source_2d(mat_data, sources, geo, quadrature, solver_1)
        fixed_source_2d(mat_data, sources, geo, quadrature, solver_n)

    def _tmin(solver, reps=5):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fixed_source_2d(mat_data, sources, geo, quadrature, solver)
            best = min(best, time.perf_counter() - t0)
        return best

    t_serial = _tmin(solver_1)
    t_parallel = _tmin(solver_n)

    speedup = t_serial / t_parallel
    assert speedup >= 1.5, (
        f"2D parallel speedup {speedup:.2f}× is below 1.5× threshold "
        f"(serial={t_serial:.3f}s, parallel={t_parallel:.3f}s, threads={n_threads})"
    )


########################################################################
# Group-level parallelism – correctness and speedup
########################################################################


def _multigroup_problem_1d(n_cells, n_angles, n_groups):
    """Build a simple 1D fixed-source multigroup problem.

    Single material with purely within-group scatter (diagonal scatter matrix).
    No cross-group coupling means Jacobi and Gauss-Seidel are mathematically
    identical (off-scatter = 0 for all groups), so both converge in exactly
    one energy iteration.  This isolates the pure parallelism speedup from
    any iteration-count difference between the two methods.
    No fission, vacuum boundary conditions.
    """
    scatter = np.zeros((1, n_groups, n_groups))
    for g in range(n_groups):
        scatter[0, g, g] = 0.5  # within-group scatter only

    mat_data = MaterialData(
        total=np.ones((1, n_groups)),
        scatter=scatter,
        fission=np.zeros((1, n_groups, n_groups)),
    )
    sources = SourceData(
        external=np.ones((n_cells, 1, n_groups)),
        boundary_x=np.zeros((2, 1, n_groups)),
    )
    geo = GeometryData(
        medium_map=np.zeros(n_cells, dtype=np.int32),
        delta_x=np.repeat(1.0 / n_cells, n_cells),
    )
    quadrature = ants.angular_x(n_angles)
    return mat_data, sources, geo, quadrature


@pytest.mark.smoke
def test_multigroup_1d_correctness():
    """Group-parallel (Jacobi) multigroup result matches serial Gauss-Seidel."""
    mat_data, sources, geo, quadrature = _multigroup_problem_1d(
        n_cells=200, n_angles=4, n_groups=16
    )

    solver_1 = SolverData(num_threads=1, tol_energy=1e-10, max_iter_energy=500)
    solver_n = SolverData(
        num_threads=N_CPUS,
        parallel=ParallelType.GROUP,
        tol_energy=1e-10,
        max_iter_energy=500,
    )

    flux_1 = fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
    flux_n = fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    assert np.allclose(flux_1, flux_n, atol=1e-6), (
        "Multigroup group-parallel flux differs from serial Gauss-Seidel "
        f"(max_diff={np.abs(flux_1 - flux_n).max():.2e})"
    )


@pytest.mark.smoke
def test_multigroup_1d_both_correctness():
    """BOTH mode (group+angle prange) matches serial Gauss-Seidel."""
    mat_data, sources, geo, quadrature = _multigroup_problem_1d(
        n_cells=200, n_angles=4, n_groups=16
    )

    solver_1 = SolverData(num_threads=1, tol_energy=1e-10, max_iter_energy=500)
    solver_n = SolverData(
        num_threads=N_CPUS,
        parallel=ParallelType.BOTH,
        tol_energy=1e-10,
        max_iter_energy=500,
    )

    flux_1 = fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
    flux_n = fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    assert np.allclose(flux_1, flux_n, atol=1e-6), (
        "Multigroup BOTH-parallel flux differs from serial Gauss-Seidel "
        f"(max_diff={np.abs(flux_1 - flux_n).max():.2e})"
    )


@pytest.mark.skipif(N_CPUS < 2, reason="Speedup test requires at least 2 CPUs")
@pytest.mark.skipif(
    _UNDER_XDIST, reason="Speedup tests unreliable under pytest-xdist (-n auto)"
)
def test_multigroup_1d_speedup():
    """Group-parallel sweep is faster than serial for a large multigroup problem.

    Group parallelism (Jacobi over groups) shines when the group count exceeds
    the angle count.  With 32 groups and 4 angles the group prange dominates
    and gives near-linear speedup up to min(N_CPUS, 32) threads.
    """
    N_GROUPS = 32
    mat_data, sources, geo, quadrature = _multigroup_problem_1d(
        n_cells=1000, n_angles=4, n_groups=N_GROUPS
    )

    # Cap at N_GROUPS threads; more threads won't parallelize additional groups.
    n_threads = min(N_CPUS, N_GROUPS, 16)
    solver_1 = SolverData(num_threads=1, tol_energy=1e-8)
    solver_n = SolverData(
        num_threads=n_threads, parallel=ParallelType.GROUP, tol_energy=1e-8
    )

    # Warm up
    for _ in range(2):
        fixed_source_1d(mat_data, sources, geo, quadrature, solver_1)
        fixed_source_1d(mat_data, sources, geo, quadrature, solver_n)

    def _tmin(solver, reps=5):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fixed_source_1d(mat_data, sources, geo, quadrature, solver)
            best = min(best, time.perf_counter() - t0)
        return best

    t_serial = _tmin(solver_1)
    t_parallel = _tmin(solver_n)

    speedup = t_serial / t_parallel
    assert speedup >= 1.5, (
        f"Multigroup group-parallel speedup {speedup:.2f}× is below 1.5× threshold "
        f"(serial={t_serial:.3f}s, parallel={t_parallel:.3f}s, threads={n_threads})"
    )
    t_parallel = _tmin(solver_n)

    speedup = t_serial / t_parallel
    assert speedup >= 1.5, (
        f"Multigroup group-parallel speedup {speedup:.2f}× is below 1.5× threshold "
        f"(serial={t_serial:.3f}s, parallel={t_parallel:.3f}s, threads={n_threads})"
    )
