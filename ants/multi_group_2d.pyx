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

from cython.parallel import prange

from ants cimport cytools_2d as tools
from ants.cytools_1d cimport _variable_cross_sections
from ants.parameters cimport params
from ants.spatial_sweep_2d cimport (
    _known_center_sweep,
    _known_interface_sweep,
    discrete_ordinates,
)

from ants.utils.pytools import dmd_2d


cdef double[:,:,:] multi_group(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # Source Iteration
    if info.mg_solver == 1:
        # Activated when parallel_type == GROUP (2) or BOTH (3).
        if info.parallel_type >= 2 and info.groups > 1:
            return jacobi_iteration(flux_guess, xs_total, xs_scatter, external, \
                            boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info)
        return source_iteration(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)
    # Dynamic Mode Decomposition
    elif info.mg_solver == 2:
        return dynamic_mode_decomp(flux_guess, xs_total, xs_scatter, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)


cdef double[:,:,:] source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):

    # Initialize components
    cdef int gg, qq, bcx, bcy
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
                            off_scatter_all[gg], external[:,:,:,qq], \
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
                    boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        flux_old[:,:,:] = flux[:,:,:]

    return flux[:,:,:]


cdef double[:,:,:] jacobi_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # Initialize components
    cdef int gg, qq, bcx, bcy
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
                        xs_scatter[:,gg,gg], off_scatter_all[gg], external[:,:,:,qq], \
                        boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info_1t)

        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]

    return flux[:,:,:]


cdef double[:,:,:] variable_source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_u, double[:]& xs_total_c, double[:]& star_coef_c, \
        double[:,:,:]& xs_scatter_u, double[:]& xs_scatter_c, double[:,:]& off_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        double[:]& edges_g, int[:]& edges_gidx_c, params info):

    # Initialize components
    cdef int gg, qq, bcx, bcy, idx1, idx2

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
            discrete_ordinates(flux[:,:,gg], flux_old[:,:,gg].copy(), xs_total_c, xs_scatter_c, \
                    off_scatter, external[:,:,:,qq], boundary_x[:,:,bcx], \
                    boundary_y[:,:,bcy], medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.tol_energy) or (count >= info.max_iter_energy)
        count += 1

        # Update old flux
        flux_old[:,:,:] = flux[:,:,:]

    return flux[:,:,:]


cdef double[:,:,:] dynamic_mode_decomp(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):

    # Initialize components
    cdef int gg, rk, kk, qq, bcx, bcy

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
                    boundary_x[:,:,:,bcx], boundary_y[:,:,:,bcy], medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)

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

    # Perform DMD
    flux = dmd_2d(flux, y_minus, y_plus, info.dmd_snapshots)

    return flux[:,:,:]


cdef double[:,:,:,:] _known_source_angular(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source

    # Initialize components
    cdef int gg, qq, bcx, bcy

    # Initialize angular flux
    angular_flux = tools.array_4d(info.cells_x, info.cells_y, \
                                  info.angles * info.angles, info.groups)
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

    return angular_flux[:,:,:,:]


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
