########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Solution for three-dimensional multigroup neutron transport problems.
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.math cimport isinf, isnan

import numpy as np
from cython.parallel import prange, threadid

from ants cimport cytools_3d as tools
from ants.parameters cimport params
from ants.spatial_sweep_3d cimport (
    _known_center_sweep,
    _known_interface_sweep,
    discrete_ordinates,
    discrete_ordinates_aniso,
    discrete_ordinates_aniso_nogil,
    discrete_ordinates_nogil,
)

from ants.utils.pytools import dmd_2d, warn_not_converged


cdef double[:,:,:,:] source_iteration(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        int[:,:] q_index=None):
    # q_index (N x G) is the slab of each angle and group for a compact
    # external source (I x J x K x S x 1), and None for a dense source

    # Jacobi iteration over groups when parallel_type == GROUP (2) or BOTH (3)
    if info.parallel_type >= 2 and info.groups > 1:
        return jacobi_iteration(flux_guess, xs_total, xs_scatter, external, \
                    boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                    delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, \
                    info, q_index)

    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)

    # Initialize flux
    flux = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, info.groups)
    flux_old = flux_guess.copy()

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Gauss-Seidel iteration over groups
    flux_1g = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)
    off_scatter = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)

    while not converged:

        flux[:,:,:,:] = 0.0

        for gg in range(info.groups):

            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg

            flux_1g[:,:,:] = flux_old[:,:,:,gg]

            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)

            discrete_ordinates(flux[:,:,:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, angle_w, info)

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:]


cdef double[:,:,:,:] jacobi_iteration(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        int[:,:] q_index=None):
    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz, tid
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)
    cdef params info_1t
    cdef int NN = info.angles

    # Initialize flux
    cdef double[:,:,:,:] flux = tools.array_4d(info.cells_x, info.cells_y, \
                                               info.cells_z, info.groups)
    cdef double[:,:,:,:] flux_old = flux_guess.copy()

    # Off-scattering term and snapshot of flux_old for each group, see the
    # 2D jacobi_iteration
    cdef double[:,:,:,:] off_scatter_all = tools.array_4d(info.groups, \
                                info.cells_x, info.cells_y, info.cells_z)
    cdef double[:,:,:,:] flux_old_snap = tools.array_4d(info.groups, \
                                info.cells_x, info.cells_y, info.cells_z)

    # GROUP mode sweeps each group on one thread, BOTH mode keeps the
    # inner angle prange (requires OMP_MAX_ACTIVE_LEVELS=2)
    info_1t = info
    if info.parallel_type == 2:
        info_1t.num_threads = 1

    # Sweep work arrays for each thread of the group prange
    cdef int threads = info.num_threads
    cdef double[:,:,:,:,:] thread_flux = tools.array_5d(threads, \
            info_1t.num_threads, info.cells_x, info.cells_y, info.cells_z)
    cdef double[:,:,:,:,:] reflected_x = tools.array_5d(threads, 2, \
                                            info.cells_y, info.cells_z, NN)
    cdef double[:,:,:,:,:] reflected_y = tools.array_5d(threads, 2, \
                                            info.cells_x, info.cells_z, NN)
    cdef double[:,:,:,:,:] reflected_z = tools.array_5d(threads, 2, \
                                            info.cells_x, info.cells_y, NN)
    cdef double[:,:,:,:] known_x_work = tools.array_4d(threads, NN, \
                                            info.cells_y, info.cells_z)
    cdef double[:,:,:,:] known_y_work = tools.array_4d(threads, NN, \
                                            info.cells_x, info.cells_z)
    cdef double[:,:,:,:] known_z_work = tools.array_4d(threads, NN, \
                                            info.cells_x, info.cells_y)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:
        flux[:,:,:,:] = 0.0

        # Refresh per-group snapshot from the current flux_old
        for gg in range(info.groups):
            flux_old_snap[gg,:,:,:] = flux_old[:,:,:,gg]

        # Compute Jacobi off-scatter for every group in parallel (nogil)
        for gg in prange(info.groups, nogil=True, num_threads=threads):
            tools._off_scatter_jacobi(flux_old, medium_map, xs_scatter, \
                                      off_scatter_all, info, gg)

        # Sweep all groups in parallel
        for gg in prange(info.groups, nogil=True, num_threads=threads):
            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg
            tid = threadid()
            discrete_ordinates_nogil(flux[:,:,:,gg], flux_old_snap[gg], \
                    xs_total[:,gg], xs_scatter[:,gg,gg], off_scatter_all[gg], \
                    external[:,:,:,:,qq], q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, angle_w, info_1t, thread_flux[tid], \
                    reflected_x[tid], reflected_y[tid], reflected_z[tid], \
                    known_x_work[tid], known_y_work[tid], known_z_work[tid])

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1
        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:]


cdef double[:,:,:,:] dynamic_mode_decomp(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        int[:,:] q_index=None):

    # Initialize components
    cdef int gg, rk, kk, qq, bcx, bcy, bcz
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)

    # Initialize flux
    flux = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)

    # Initialize Y_plus and Y_minus
    y_plus = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            info.groups, info.dmd_snapshots - 1)
    y_minus = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                             info.groups, info.dmd_snapshots - 1)

    # Create off-scattering term
    off_scatter = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)

    # Set convergence limits
    cdef bint converged = False
    cdef double change = 0.0

    # Iterate over removed source iterations
    for rk in range(info.dmd_rank + info.dmd_snapshots):

        # Return flux if there is convergence
        if converged:
            return flux[:,:,:,:]

        # Zero out flux
        flux[:,:,:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg

            # Select the specific group from last iteration
            flux_1g[:,:,:] = flux_old[:,:,:,gg]

            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,:,:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy)

        # Collect difference for DMD on K iterations
        if rk >= info.dmd_rank:
            kk = rk - info.dmd_rank
            tools._dmd_subtraction(y_minus, y_plus, flux, flux_old, kk, info)

        # Update old flux
        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Perform DMD on (cells_x, cells_y * cells_z, groups), which has the
    # same flattened order as the snapshots
    flux_dmd = dmd_2d(np.asarray(flux).reshape(info.cells_x, \
                      info.cells_y * info.cells_z, info.groups), y_minus, \
                      y_plus, info.dmd_snapshots)
    flux_dmd = np.ascontiguousarray(flux_dmd).reshape(info.cells_x, \
                            info.cells_y, info.cells_z, info.groups)

    return flux_dmd[:,:,:,:]


cdef void _known_source_angular(double[:,:,:,:,:]& angular_flux, \
        double[:,:]& xs_total, double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        double[:,:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        params info):
    # source = flux * xs_scatter + external source
    # angular_flux (cells_x, cells_y, cells_z, angles, groups) is
    # overwritten and must not share memory with source

    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz

    # The sweeps add to the angular flux
    angular_flux[:,:,:,:,:] = 0.0
    # Set zero matrix placeholder for scattering
    zero_3d = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[4] == 1 else gg
        bcx = 0 if boundary_x.shape[4] == 1 else gg
        bcy = 0 if boundary_y.shape[4] == 1 else gg
        bcz = 0 if boundary_z.shape[4] == 1 else gg

        # Perform angular sweep
        _known_center_sweep(angular_flux[:,:,:,:,gg], xs_total[:,gg], zero_3d, \
                source[:,:,:,:,qq], boundary_x[:,:,:,:,bcx], \
                boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], medium_map, \
                delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, info)



cdef void _interface_flux(double[:,:,:,:,:]& flux_edge_x, \
        double[:,:,:,:,:]& flux_edge_y, double[:,:,:,:,:]& flux_edge_z, \
        double[:,:]& xs_total, double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        double[:,:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source
    # The face fluxes are (cells_x + 1, cells_y, cells_z, xdim, groups),
    # (cells_x, cells_y + 1, cells_z, xdim, groups), and (cells_x, cells_y,
    # cells_z + 1, xdim, groups), where xdim is the number of angles for
    # the angular flux and 1 for the scalar flux

    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz

    # The sweeps add to the face fluxes
    flux_edge_x[:,:,:,:,:] = 0.0
    flux_edge_y[:,:,:,:,:] = 0.0
    flux_edge_z[:,:,:,:,:] = 0.0

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[4] == 1 else gg
        bcx = 0 if boundary_x.shape[4] == 1 else gg
        bcy = 0 if boundary_y.shape[4] == 1 else gg
        bcz = 0 if boundary_z.shape[4] == 1 else gg

        # Perform angular sweep
        _known_interface_sweep(flux_edge_x[:,:,:,:,gg], flux_edge_y[:,:,:,:,gg], \
                flux_edge_z[:,:,:,:,gg], xs_total[:,gg], source[:,:,:,:,qq], \
                boundary_x[:,:,:,:,bcx], boundary_y[:,:,:,:,bcy], \
                boundary_z[:,:,:,:,bcz], medium_map, delta_x, delta_y, \
                delta_z, angle_x, angle_y, angle_z, angle_w, info)


########################################################################
# Anisotropic Scattering (spherical harmonic moments)
########################################################################
# Flux moments are (cells_x, cells_y, cells_z, groups, moments) and
# xs_scatter is (materials, groups, groups, moments). A single moment runs
# the scalar flux solvers above, so isotropic problems are unchanged.

cdef double[:,:,:,:,:] multi_group_moments(double[:,:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):
    # Isotropic scattering
    if info.moments == 1:
        flux = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                              info.groups, 1)
        if info.mg_solver == 2:
            flux[:,:,:,:,0] = dynamic_mode_decomp(flux_guess[:,:,:,:,0], \
                    xs_total, xs_scatter[:,:,:,0], external, boundary_x, \
                    boundary_y, boundary_z, medium_map, delta_x, delta_y, \
                    delta_z, angle_x, angle_y, angle_z, angle_w, info, q_index)
        else:
            flux[:,:,:,:,0] = source_iteration(flux_guess[:,:,:,:,0], \
                    xs_total, xs_scatter[:,:,:,0], external, boundary_x, \
                    boundary_y, boundary_z, medium_map, delta_x, delta_y, \
                    delta_z, angle_x, angle_y, angle_z, angle_w, info, q_index)
        return flux[:,:,:,:,:]
    # Source Iteration
    if info.mg_solver == 1:
        # Activated when parallel_type == GROUP (2) or BOTH (3)
        if info.parallel_type >= 2 and info.groups > 1:
            return jacobi_iteration_aniso(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, boundary_z, medium_map, \
                    delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, \
                    angle_w, P_source, P_weights, info, q_index)
        return source_iteration_aniso(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, boundary_z, medium_map, \
                    delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, \
                    angle_w, P_source, P_weights, info, q_index)
    # Dynamic Mode Decomposition
    elif info.mg_solver == 2:
        return dynamic_mode_decomp_aniso(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, boundary_z, medium_map, \
                    delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, \
                    angle_w, P_source, P_weights, info, q_index)


cdef double[:,:,:,:,:] source_iteration_aniso(double[:,:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)

    # Initialize flux moments
    flux = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                          info.groups, info.moments)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, \
                             info.moments)

    # Create off-scattering moments
    off_scatter = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, \
                                 info.moments)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:

        flux[:,:,:,:,:] = 0.0

        for gg in range(info.groups):

            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg

            flux_1g[:,:,:,:] = flux_old[:,:,:,gg,:]

            tools._off_scatter_moments(flux, flux_old, medium_map, xs_scatter, \
                                       off_scatter, info, gg)

            discrete_ordinates_aniso(flux[:,:,:,gg,:], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg,:], off_scatter, external[:,:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, P_source, P_weights, info)

        change = tools.group_convergence(flux[:,:,:,:,0], flux_old[:,:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        flux_old[:,:,:,:,:] = flux[:,:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:,:]


cdef double[:,:,:,:,:] jacobi_iteration_aniso(double[:,:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy, bcz, tid
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)
    cdef params info_1t
    cdef int NN = info.angles

    # Initialize flux moments
    cdef double[:,:,:,:,:] flux = tools.array_5d(info.cells_x, info.cells_y, \
                                    info.cells_z, info.groups, info.moments)
    cdef double[:,:,:,:,:] flux_old = flux_guess.copy()

    # Off-scattering moments and snapshot of flux_old for each group
    cdef double[:,:,:,:,:] off_scatter_all = tools.array_5d(info.groups, \
                    info.cells_x, info.cells_y, info.cells_z, info.moments)
    cdef double[:,:,:,:,:] flux_old_snap = tools.array_5d(info.groups, \
                    info.cells_x, info.cells_y, info.cells_z, info.moments)

    # Thread settings, see jacobi_iteration
    info_1t = info
    if info.parallel_type == 2:
        info_1t.num_threads = 1

    # Sweep work arrays for each thread of the group prange, see
    # jacobi_iteration. The zeros are read-only and shared.
    cdef int threads = info.num_threads
    cdef double[:,:,:,:,:] thread_psi = tools.array_5d(threads, \
            info_1t.num_threads, info.cells_x, info.cells_y, info.cells_z)
    cdef double[:,:,:,:,:] thread_source = tools.array_5d(threads, \
            info_1t.num_threads, info.cells_x, info.cells_y, info.cells_z)
    cdef double[:,:,:,:,:,:] thread_flux = tools.array_6d(threads, \
            info_1t.num_threads, info.cells_x, info.cells_y, info.cells_z, \
            info.moments)
    cdef double[:,:,:] zero_3d = tools.array_3d(info.cells_x, info.cells_y, \
                                                info.cells_z)
    cdef double[:] zero_xs = tools.array_1d(info.materials)
    cdef double[:,:,:,:,:] reflected_x = tools.array_5d(threads, 2, \
                                            info.cells_y, info.cells_z, NN)
    cdef double[:,:,:,:,:] reflected_y = tools.array_5d(threads, 2, \
                                            info.cells_x, info.cells_z, NN)
    cdef double[:,:,:,:,:] reflected_z = tools.array_5d(threads, 2, \
                                            info.cells_x, info.cells_y, NN)
    cdef double[:,:,:,:] known_x_work = tools.array_4d(threads, NN, \
                                            info.cells_y, info.cells_z)
    cdef double[:,:,:,:] known_y_work = tools.array_4d(threads, NN, \
                                            info.cells_x, info.cells_z)
    cdef double[:,:,:,:] known_z_work = tools.array_4d(threads, NN, \
                                            info.cells_x, info.cells_y)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:
        flux[:,:,:,:,:] = 0.0

        # Refresh per-group snapshot from the current flux_old
        for gg in range(info.groups):
            flux_old_snap[gg,:,:,:,:] = flux_old[:,:,:,gg,:]

        # Compute Jacobi off-scatter for every group in parallel (nogil)
        for gg in prange(info.groups, nogil=True, num_threads=threads):
            tools._off_scatter_jacobi_moments(flux_old, medium_map, xs_scatter, \
                                              off_scatter_all, info, gg)

        # Sweep all groups in parallel
        for gg in prange(info.groups, nogil=True, num_threads=threads):
            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg
            tid = threadid()
            discrete_ordinates_aniso_nogil(flux[:,:,:,gg,:], flux_old_snap[gg], \
                    xs_total[:,gg], xs_scatter[:,gg,gg,:], off_scatter_all[gg], \
                    external[:,:,:,:,qq], q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, P_source, P_weights, info_1t, thread_psi[tid], \
                    thread_source[tid], thread_flux[tid], zero_3d, zero_xs, \
                    reflected_x[tid], reflected_y[tid], reflected_z[tid], \
                    known_x_work[tid], known_y_work[tid], known_z_work[tid])

        change = tools.group_convergence(flux[:,:,:,:,0], flux_old[:,:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1
        flux_old[:,:,:,:,:] = flux[:,:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:,:]


cdef double[:,:,:,:,:] dynamic_mode_decomp_aniso(double[:,:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, rk, kk, qq, bcx, bcy, bcz
    cdef int[:,:] q_cols = tools._source_columns(external.shape[3], \
                                                 q_index, info)
    cdef int unknowns = info.groups * info.moments

    # Initialize flux moments
    flux = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                          info.groups, info.moments)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, \
                             info.moments)

    # Initialize Y_plus and Y_minus, with the groups and moments flattened
    y_plus = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            unknowns, info.dmd_snapshots - 1)
    y_minus = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                             unknowns, info.dmd_snapshots - 1)

    # Create off-scattering moments
    off_scatter = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, \
                                 info.moments)

    # Set convergence limits
    cdef bint converged = False
    cdef double change = 0.0

    # Iterate over removed source iterations
    for rk in range(info.dmd_rank + info.dmd_snapshots):

        # Return flux if there is convergence
        if converged:
            return flux[:,:,:,:,:]

        # Zero out flux
        flux[:,:,:,:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq  = 0 if external.shape[4]   == 1 else gg
            bcx = 0 if boundary_x.shape[4] == 1 else gg
            bcy = 0 if boundary_y.shape[4] == 1 else gg
            bcz = 0 if boundary_z.shape[4] == 1 else gg

            # Select the specific group from last iteration
            flux_1g[:,:,:,:] = flux_old[:,:,:,gg,:]

            # Calculate up and down scattering moments using Gauss-Seidel
            tools._off_scatter_moments(flux, flux_old, medium_map, xs_scatter, \
                                       off_scatter, info, gg)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates_aniso(flux[:,:,:,gg,:], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg,:], off_scatter, external[:,:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,:,bcx], \
                    boundary_y[:,:,:,:,bcy], boundary_z[:,:,:,:,bcz], \
                    medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                    angle_z, P_source, P_weights, info)

        # Check for convergence
        change = tools.group_convergence(flux[:,:,:,:,0], flux_old[:,:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy)

        # Collect difference for DMD on K iterations
        if rk >= info.dmd_rank:
            kk = rk - info.dmd_rank
            tools._dmd_subtraction_moments(y_minus, y_plus, flux, flux_old, kk, info)

        # Update old flux
        flux_old[:,:,:,:,:] = flux[:,:,:,:,:]

    # Perform DMD on (cells_x, cells_y * cells_z, groups * moments)
    flux_dmd = dmd_2d(np.asarray(flux).reshape(info.cells_x, \
                      info.cells_y * info.cells_z, unknowns), y_minus, y_plus, \
                      info.dmd_snapshots)
    flux_dmd = np.ascontiguousarray(flux_dmd).reshape(info.cells_x, \
                    info.cells_y, info.cells_z, info.groups, info.moments)

    return flux_dmd[:,:,:,:,:]
