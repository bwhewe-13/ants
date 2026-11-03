########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Solution for two-dimensional multigroup neutron transport problems.
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
from cython.parallel import prange

from ants cimport cytools_2d as tools
from ants.cytools_1d cimport _variable_cross_sections, _variable_cross_sections_aniso
from ants.parameters cimport params
from ants.spatial_sweep_2d cimport (
    _known_center_sweep,
    _known_interface_sweep,
    discrete_ordinates,
    discrete_ordinates_aniso,
)

from ants.utils.pytools import dmd_2d, warn_not_converged


cdef double[:,:,:] multi_group(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, \
        int[:,:] q_index=None):
    # q_index (N^2 x G) is the slab of each angle and group for a compact
    # external source (I x J x S x 1), and None for a dense external source
    # Source Iteration
    if info.mg_solver == 1:
        # Activated when parallel_type == GROUP (2) or BOTH (3).
        if info.parallel_type >= 2 and info.groups > 1:
            return jacobi_iteration(flux_guess, xs_total, xs_scatter, external, \
                            boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info, q_index)
        return source_iteration(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info, q_index)
    # Dynamic Mode Decomposition
    elif info.mg_solver == 2:
        return dynamic_mode_decomp(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info, q_index)


cdef double[:,:,:] source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)
    cdef params info_1t

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0


    if info.parallel_type >= 2 and info.groups > 1:

        # Per-group off-scatter buffer: off_scatter_all[gg, ii, jj]
        off_scatter_all = tools.array_3d(info.groups, info.cells_x, info.cells_y)

        # Row-major snapshot: flux_old_snap[gg, ii, jj].
        # square_ordinates writes flux_old[:,:] = flux[:,:] at the end of
        # each angular inner iteration.  Passing flux_old[:,:,gg] directly
        # would corrupt the outer flux_old and make group_convergence() report
        # zero change after the first sweep, causing premature exit.
        flux_old_snap = tools.array_3d(info.groups, info.cells_x, info.cells_y)

        info_1t = info
        if info.parallel_type == 2:   # GROUP only
            info_1t.num_threads = 1

        while not converged:

            flux[:,:,:] = 0.0

            # Refresh per-group snapshot from the current flux_old
            for gg in range(info.groups):
                flux_old_snap[gg, :, :] = flux_old[:, :, gg]

            # Compute Jacobi off-scatter for every group in parallel (nogil)
            for gg in prange(info.groups, nogil=True, \
                             num_threads=info.num_threads):
                tools._off_scatter_jacobi(flux_old, medium_map, xs_scatter, \
                                          off_scatter_all, info, gg)

            # Sweep all groups in parallel
            for gg in prange(info.groups, nogil=True, \
                             num_threads=info.num_threads):
                qq  = 0 if external.shape[3]   == 1 else gg
                bcx = 0 if boundary_x.shape[3]  == 1 else gg
                bcy = 0 if boundary_y.shape[3]  == 1 else gg
                with gil:
                    discrete_ordinates(flux[:,:,gg], flux_old_snap[gg], \
                            xs_total[:,gg], xs_scatter[:,gg,gg], \
                            off_scatter_all[gg], external[:,:,:,qq], q_cols[:,gg], \
                            boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
                            medium_map, delta_x, delta_y, angle_x, angle_y, \
                            angle_w, info_1t)

            change = tools.group_convergence(flux, flux_old, info)
            if isnan(change) or isinf(change):
                change = 0.5
            converged = (change < info.tol_energy) \
                        or (count >= info.max_iter_energy)
            count += 1
            flux_old[:,:,:] = flux[:,:,:]

        # Warn if the energy iterations did not converge
        if change >= info.tol_energy:
            warn_not_converged("Multigroup", "max_iter_energy")

        return flux[:,:,:]

    # -----------------------------------------------------------------------
    # Sequential path: Gauss-Seidel iteration
    # -----------------------------------------------------------------------
    flux_1g = tools.array_2d(info.cells_x, info.cells_y)
    off_scatter = tools.array_2d(info.cells_x, info.cells_y)

    while not converged:

        flux[:,:,:] = 0.0

        for gg in range(info.groups):

            qq  = 0 if external.shape[3]  == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            flux_1g[:,:] = flux_old[:,:,gg]

            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)

            discrete_ordinates(flux[:,:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        flux_old[:,:,:] = flux[:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:]


cdef double[:,:,:] jacobi_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, int[:,:] q_index=None):
    # Initialize components
    cdef int gg, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)
    cdef params info_1t

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()

    # Create off-scattering term
    off_scatter_all = tools.array_3d(info.groups, info.cells_x, info.cells_y)
    # Passing flux_old[:,:,gg] directly to flux would corrupt the outer flux_old
    flux_old_snap = tools.array_3d(info.groups, info.cells_x, info.cells_y)

    # Copy params; for GROUP mode disable inner angle prange to avoid
    # oversubscription; for BOTH mode keep the full thread count so the
    # inner angle prange also runs in parallel (requires OMP_MAX_ACTIVE_LEVELS=2).
    info_1t = info
    if info.parallel_type == 2:
        info_1t.num_threads = 1

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:
        flux[:,:,:] = 0.0

        # Refresh per-group snapshot from the current flux_old
        for gg in range(info.groups):
            flux_old_snap[gg, :, :] = flux_old[:, :, gg]

        # Compute Jacobi off-scatter for every group in parallel (nogil)
        for gg in prange(info.groups, nogil=True, num_threads=info.num_threads):
            tools._off_scatter_jacobi(flux_old, medium_map, xs_scatter, \
                                    off_scatter_all, info, gg)

        # Sweep all groups in parallel
        for gg in prange(info.groups, nogil=True, num_threads=info.num_threads):
            qq  = 0 if external.shape[3]   == 1 else gg
            bcx = 0 if boundary_x.shape[3]  == 1 else gg
            bcy = 0 if boundary_y.shape[3]  == 1 else gg
            with gil:
                discrete_ordinates(flux[:,:,gg], flux_old_snap[gg], xs_total[:,gg], \
                        xs_scatter[:,gg,gg], off_scatter_all[gg], \
                        external[:,:,:,qq], q_cols[:,gg], boundary_x[:,:,:,bcx], \
                        boundary_y[:,:,:,bcy], medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, info_1t)

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:]


cdef double[:,:,:] variable_source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_u, double[:]& xs_total_c, double[:]& star_coef_c, \
        double[:,:,:]& xs_scatter_u, double[:]& xs_scatter_c, double[:,:]& off_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        double[:]& edges_g, int[:]& edges_gidx_c, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy, idx1, idx2
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until energy group convergence
    while not (converged):

        # Zero out flux
        flux[:,:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[3] == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg+1]

            _variable_cross_sections(xs_total_c, xs_total_u, star_coef_c[gg], \
                        xs_scatter_c, xs_scatter_u, edges_g, idx1, idx2, info)

            # Calculate up and down scattering term using Gauss-Seidel
            tools._variable_off_scatter(flux, flux_old, medium_map, xs_scatter_u, \
                                        off_scatter, gg, edges_g, edges_gidx_c, \
                                        idx1, idx2, info)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,:,gg], flux_old[:,:,gg].copy(), xs_total_c, \
                    xs_scatter_c, off_scatter, external[:,:,:,qq], q_cols[:,gg], \
                    boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        # Update old flux
        flux_old[:,:,:] = flux[:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:]


cdef double[:,:,:] dynamic_mode_decomp(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, rk, kk, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_2d(info.cells_x, info.cells_y)

    # Initialize Y_plus and Y_minus
    y_plus = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.dmd_snapshots - 1)
    y_minus = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.dmd_snapshots - 1)

    # Create off-scattering term
    off_scatter = tools.array_2d(info.cells_x, info.cells_y)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1

    # Iterate over removed source iterations
    for rk in range(info.dmd_rank + info.dmd_snapshots):

        # Return flux if there is convergence
        if converged:
            return flux[:,:,:]

        # Zero out flux
        flux[:,:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[3] == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            # Select the specific group from last iteration
            flux_1g[:,:] = flux_old[:,:,gg]

            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy)

        # Collect difference for DMD on K iterations
        if rk >= info.dmd_rank:
            # Get indexing
            kk = rk - info.dmd_rank
            tools._dmd_subtraction(y_minus, y_plus, flux, flux_old, kk, info)

        # Update old flux
        flux_old[:,:,:] = flux[:,:,:]

    # Perform DMD (new name keeps flux a typed memoryview in the loop)
    flux_dmd = dmd_2d(flux, y_minus, y_plus, info.dmd_snapshots)

    return flux_dmd[:,:,:]


cdef void _known_source_angular(double[:,:,:,:]& angular_flux, \
        double[:,:]& xs_total, double[:,:,:,:]& source, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    # source = flux * xs_scatter + external source
    # angular_flux (cells_x, cells_y, angles**2, groups) is overwritten
    # and must not share memory with source

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # The sweeps add to the angular flux
    angular_flux[:,:,:,:] = 0.0
    # Set zero matrix placeholder for scattering
    zero_2d = tools.array_2d(info.cells_x, info.cells_y)

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else gg
        bcx = 0 if boundary_x.shape[3] == 1 else gg
        bcy = 0 if boundary_y.shape[3] == 1 else gg

        # Perform angular sweep
        _known_center_sweep(angular_flux[:,:,:,gg], xs_total[:,gg], zero_2d, \
            source[:,:,:,qq], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)


cdef double[:,:,:] _known_source_scalar(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # Initialize scalar flux
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, 1)

    # Set zero matrix placeholder for scattering
    zero_2d = tools.array_2d(info.cells_x, info.cells_y)

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else gg
        bcx = 0 if boundary_x.shape[3] == 1 else gg
        bcy = 0 if boundary_y.shape[3] == 1 else gg

        # Perform angular sweep
        _known_center_sweep(scalar_flux[:,:,gg], xs_total[:,gg], zero_2d, \
            source[:,:,:,qq], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)

    return scalar_flux[:,:,:,0]


cdef double[:,:,:,:] _known_source_single(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, int group, params info):
    # source = flux * xs_scatter + external source

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # Initialize angular flux
    angular_flux = tools.array_4d(info.cells_x, info.cells_y, \
                                  info.angles * info.angles, 1)
    # Set zero matrix placeholder for scattering
    zero_2d = tools.array_2d(info.cells_x, info.cells_y)

    # Iterate over groups
    for gg in range(group, group+1):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else gg
        bcx = 0 if boundary_x.shape[3] == 1 else gg
        bcy = 0 if boundary_y.shape[3] == 1 else gg

        # Perform angular sweep
        _known_center_sweep(angular_flux[:,:,:,0], xs_total[:,gg], zero_2d, \
            source[:,:,:,qq], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
            medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)

    return angular_flux[:,:,:,:]


cdef void _interface_angular(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source

    # flux_edge_x = [(I+1) x J], flux_edge_y = [I x (J+1)]
    flux_edge_x[:,:,:,:] = 0.0
    flux_edge_y[:,:,:,:] = 0.0

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else gg
        bcx = 0 if boundary_x.shape[3] == 1 else gg
        bcy = 0 if boundary_y.shape[3] == 1 else gg

        # Perform angular sweep
        _known_interface_sweep(flux_edge_x[:,:,:,gg], flux_edge_y[:,:,:,gg], \
                xs_total[:,gg], source[:,:,:,qq], boundary_x[:,:,:,bcx], \
                boundary_y[:,:,:,bcy], medium_map, delta_x, delta_y, \
                angle_x, angle_y, angle_w, info)


cdef void _interface_scalar(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source
    # flux_edge_x = [(I+1) x J], flux_edge_y = [I x (J+1)]

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else gg
        bcx = 0 if boundary_x.shape[3] == 1 else gg
        bcy = 0 if boundary_y.shape[3] == 1 else gg

        # Perform angular sweep
        _known_interface_sweep(flux_edge_x[:,:,gg], flux_edge_y[:,:,gg], \
                xs_total[:,gg], source[:,:,:,qq], boundary_x[:,:,:,bcx], \
                boundary_y[:,:,:,bcy], medium_map, delta_x, delta_y, \
                angle_x, angle_y, angle_w, info)


########################################################################
# Anisotropic Scattering (spherical harmonic moments)
########################################################################
# Flux moments are (cells_x, cells_y, groups, moments) and xs_scatter is
# (materials, groups, groups, moments). A single moment runs the scalar
# flux solvers above, so isotropic problems are unchanged.

cdef double[:,:,:,:] multi_group_moments(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):
    # Isotropic scattering
    if info.moments == 1:
        flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, 1)
        flux[:,:,:,0] = multi_group(flux_guess[:,:,:,0], xs_total, \
                        xs_scatter[:,:,:,0], external, boundary_x, boundary_y, \
                        medium_map, delta_x, delta_y, angle_x, angle_y, \
                        angle_w, info, q_index)
        return flux[:,:,:,:]
    # Source Iteration
    if info.mg_solver == 1:
        # Activated when parallel_type == GROUP (2) or BOTH (3).
        if info.parallel_type >= 2 and info.groups > 1:
            return jacobi_iteration_aniso(flux_guess, xs_total, xs_scatter, \
                        external, boundary_x, boundary_y, medium_map, delta_x, \
                        delta_y, angle_x, angle_y, angle_w, P_source, \
                        P_weights, info, q_index)
        return source_iteration_aniso(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, delta_x, \
                    delta_y, angle_x, angle_y, angle_w, P_source, P_weights, \
                    info, q_index)
    # Dynamic Mode Decomposition
    elif info.mg_solver == 2:
        return dynamic_mode_decomp_aniso(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, delta_x, \
                    delta_y, angle_x, angle_y, angle_w, P_source, P_weights, \
                    info, q_index)


cdef double[:,:,:,:] source_iteration_aniso(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)

    # Initialize flux moments
    flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_3d(info.cells_x, info.cells_y, info.moments)

    # Create off-scattering moments
    off_scatter = tools.array_3d(info.cells_x, info.cells_y, info.moments)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:

        flux[:,:,:,:] = 0.0

        for gg in range(info.groups):

            qq  = 0 if external.shape[3]  == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            flux_1g[:,:,:] = flux_old[:,:,gg,:]

            tools._off_scatter_moments(flux, flux_old, medium_map, xs_scatter, \
                                       off_scatter, info, gg)

            discrete_ordinates_aniso(flux[:,:,gg,:], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg,:], off_scatter, external[:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, \
                    P_source, P_weights, info)

        change = tools.group_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:]


cdef double[:,:,:,:] jacobi_iteration_aniso(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)
    cdef params info_1t

    # Initialize flux moments
    flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)
    flux_old = flux_guess.copy()

    # Create off-scattering moments
    off_scatter_all = tools.array_4d(info.groups, info.cells_x, info.cells_y, \
                                     info.moments)
    flux_old_snap = tools.array_4d(info.groups, info.cells_x, info.cells_y, \
                                   info.moments)

    # Thread settings, see jacobi_iteration
    info_1t = info
    if info.parallel_type == 2:
        info_1t.num_threads = 1

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:
        flux[:,:,:,:] = 0.0

        # Refresh per-group snapshot from the current flux_old
        for gg in range(info.groups):
            flux_old_snap[gg,:,:,:] = flux_old[:,:,gg,:]

        # Compute Jacobi off-scatter for every group in parallel (nogil)
        for gg in prange(info.groups, nogil=True, num_threads=info.num_threads):
            tools._off_scatter_jacobi_moments(flux_old, medium_map, xs_scatter, \
                                              off_scatter_all, info, gg)

        # Sweep all groups in parallel
        for gg in prange(info.groups, nogil=True, num_threads=info.num_threads):
            qq  = 0 if external.shape[3]   == 1 else gg
            bcx = 0 if boundary_x.shape[3]  == 1 else gg
            bcy = 0 if boundary_y.shape[3]  == 1 else gg
            with gil:
                discrete_ordinates_aniso(flux[:,:,gg,:], flux_old_snap[gg], \
                        xs_total[:,gg], xs_scatter[:,gg,gg,:], off_scatter_all[gg], \
                        external[:,:,:,qq], q_cols[:,gg], boundary_x[:,:,:,bcx], \
                        boundary_y[:,:,:,bcy], medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, P_source, P_weights, info_1t)

        change = tools.group_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1
        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:]


cdef double[:,:,:,:] dynamic_mode_decomp_aniso(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, rk, kk, qq, bcx, bcy
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)
    cdef int unknowns = info.groups * info.moments

    # DMD acts on all moments, flattened to (cells_x, cells_y, groups * moments)
    cdef params info_d = info
    info_d.groups = unknowns

    # Initialize flux moments
    flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_3d(info.cells_x, info.cells_y, info.moments)

    # Initialize Y_plus and Y_minus
    y_plus = tools.array_4d(info.cells_x, info.cells_y, unknowns, \
                            info.dmd_snapshots - 1)
    y_minus = tools.array_4d(info.cells_x, info.cells_y, unknowns, \
                             info.dmd_snapshots - 1)

    # Create off-scattering moments
    off_scatter = tools.array_3d(info.cells_x, info.cells_y, info.moments)

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
            qq = 0 if external.shape[3] == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            # Select the specific group from last iteration
            flux_1g[:,:,:] = flux_old[:,:,gg,:]

            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter_moments(flux, flux_old, medium_map, xs_scatter, \
                                       off_scatter, info, gg)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates_aniso(flux[:,:,gg,:], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg,:], off_scatter, external[:,:,:,qq], \
                    q_cols[:,gg], boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, \
                    P_source, P_weights, info)

        # Check for convergence
        change = tools.group_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy)

        # Collect difference for DMD on K iterations
        if rk >= info.dmd_rank:
            # Get indexing
            kk = rk - info.dmd_rank
            tools._dmd_subtraction(y_minus, y_plus, \
                    np.asarray(flux).reshape(info.cells_x, info.cells_y, unknowns), \
                    np.asarray(flux_old).reshape(info.cells_x, info.cells_y, unknowns), \
                    kk, info_d)

        # Update old flux
        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Perform DMD (new name keeps flux a typed memoryview in the loop)
    flux_dmd = dmd_2d(np.asarray(flux).reshape(info.cells_x, info.cells_y, \
                      unknowns), y_minus, y_plus, info.dmd_snapshots)
    flux_dmd = np.ascontiguousarray(flux_dmd).reshape(info.cells_x, \
                            info.cells_y, info.groups, info.moments)

    return flux_dmd[:,:,:,:]


cdef double[:,:,:,:] variable_source_iteration_moments( \
        double[:,:,:,:]& flux_guess, double[:,:]& xs_total_u, \
        double[:]& star_coef_c, double[:,:,:,:]& xs_scatter_u, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, double[:]& edges_g, int[:]& edges_gidx_c, \
        double[:,:]& P_source, double[:,:]& P_weights, params info, int[:,:] q_index=None):

    # Initialize components
    cdef int gg, qq, bcx, bcy, idx1, idx2
    cdef int[:,:] q_cols = tools._source_columns(external.shape[2], \
                                                 q_index, info)

    # Initialize flux moments
    flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_3d(info.cells_x, info.cells_y, info.moments)

    # Create collided and off-scattering terms
    xs_total_c = tools.array_1d(info.materials)
    xs_scatter_c = tools.array_2d(info.materials, info.moments)
    off_scatter = tools.array_3d(info.cells_x, info.cells_y, info.moments)

    # Isotropic scattering
    if info.moments == 1:
        flux[:,:,:,0] = variable_source_iteration(flux_guess[:,:,:,0], \
                    xs_total_u, xs_total_c, star_coef_c, xs_scatter_u[:,:,:,0], \
                    xs_scatter_c[:,0], off_scatter[:,:,0], external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, edges_g, edges_gidx_c, info, \
                    q_index)
        return flux[:,:,:,:]

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until energy group convergence
    while not (converged):

        # Zero out flux
        flux[:,:,:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[3] == 1 else gg
            bcx = 0 if boundary_x.shape[3] == 1 else gg
            bcy = 0 if boundary_y.shape[3] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg+1]

            _variable_cross_sections_aniso(xs_total_c, xs_total_u, \
                        star_coef_c[gg], xs_scatter_c, xs_scatter_u, edges_g, \
                        idx1, idx2, info)

            # Select the specific group from last iteration
            flux_1g[:,:,:] = flux_old[:,:,gg,:]

            # Calculate up and down scattering moments using Gauss-Seidel
            tools._variable_off_scatter_aniso(flux, flux_old, medium_map, \
                        xs_scatter_u, off_scatter, gg, edges_g, edges_gidx_c, \
                        idx1, idx2, info)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates_aniso(flux[:,:,gg,:], flux_1g, xs_total_c, \
                    xs_scatter_c, off_scatter, external[:,:,:,qq], q_cols[:,gg], \
                    boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, P_source, \
                    P_weights, info)

        # Check for convergence
        change = tools.group_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        # Update old flux
        flux_old[:,:,:,:] = flux[:,:,:,:]

    # Warn if the energy iterations did not converge
    if change >= info.tol_energy:
        warn_not_converged("Multigroup", "max_iter_energy")

    return flux[:,:,:,:]


cdef double[:,:,:,:] _known_source_moments(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_weights, \
        params info):
    # Flux moments from a known source (I x J x N^2 x G)
    flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)
    if info.moments == 1:
        flux[:,:,:,0] = _known_source_scalar(xs_total, source, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, angle_x, \
                        angle_y, angle_w, info)
        return flux[:,:,:,:]
    angular_flux = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)
    _known_source_angular(angular_flux, xs_total, source, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, angle_x, \
                        angle_y, angle_w, info)
    tools._angular_to_moments(angular_flux, flux, P_weights, info)
    return flux[:,:,:,:]
