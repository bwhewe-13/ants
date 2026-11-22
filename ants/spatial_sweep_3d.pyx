########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Spatial sweeps for three-dimensional neutron transport problems.
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

from libc.math cimport fabs

from cython.parallel import prange, threadid

from ants cimport cytools_3d as tools
from ants.cytools_shared cimport step_characteristic_weight
from ants.parameters cimport params

########################################################################
# Iterative Sweep - Cube Geometry
#
# Same layout as the 2D square sweep. The angle loop is parallelized
# with prange, where each thread adds to its own (I x J x K) flux array.
# The known edges of angle nn are in row nn of known_{x,y,z}_work, so no
# two threads share them. The reflected arrays are only read inside
# prange and are updated afterward from the exit edges.
#
# In 3D, info.angles is the number of ordinates.
########################################################################

cdef void discrete_ordinates(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info):
    # Allocate the work arrays for a single sweep
    cdef int NN = info.angles
    thread_flux = tools.array_4d(info.num_threads, info.cells_x, \
                                 info.cells_y, info.cells_z)
    reflected_x = tools.array_4d(2, info.cells_y, info.cells_z, NN)
    reflected_y = tools.array_4d(2, info.cells_x, info.cells_z, NN)
    reflected_z = tools.array_4d(2, info.cells_x, info.cells_y, NN)
    known_x_work = tools.array_3d(NN, info.cells_y, info.cells_z)
    known_y_work = tools.array_3d(NN, info.cells_x, info.cells_z)
    known_z_work = tools.array_3d(NN, info.cells_x, info.cells_y)
    discrete_ordinates_nogil(flux, flux_old, xs_total, xs_scatter, \
                off_scatter, external, q_col, boundary_x, boundary_y, \
                boundary_z, medium_map, delta_x, delta_y, delta_z, angle_x, \
                angle_y, angle_z, angle_w, info, thread_flux, reflected_x, \
                reflected_y, reflected_z, known_x_work, known_y_work, \
                known_z_work)


cdef void discrete_ordinates_nogil(double[:,:,:]& flux, \
        double[:,:,:]& flux_old, double[:]& xs_total, double[:]& xs_scatter, \
        double[:,:,:]& off_scatter, double[:,:,:,:]& external, int[:]& q_col, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info, double[:,:,:,:]& thread_flux, \
        double[:,:,:,:]& reflected_x, double[:,:,:,:]& reflected_y, \
        double[:,:,:,:]& reflected_z, double[:,:,:]& known_x_work, \
        double[:,:,:]& known_y_work, double[:,:,:]& known_z_work) noexcept nogil:
    # Rectangular spatial cells (SLAB3D = 4)
    if info.geometry == 4:
        cube_ordinates(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                external, q_col, boundary_x, boundary_y, boundary_z, \
                medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                angle_z, angle_w, info, thread_flux, reflected_x, \
                reflected_y, reflected_z, known_x_work, known_y_work, \
                known_z_work)


cdef void cube_ordinates(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        double[:,:,:,:]& thread_flux, double[:,:,:,:]& reflected_x, \
        double[:,:,:,:]& reflected_y, double[:,:,:,:]& reflected_z, \
        double[:,:,:]& known_x_work, double[:,:,:]& known_y_work, \
        double[:,:,:]& known_z_work) noexcept nogil:

    cdef int nn, ii, jj, kk, qq, bcx, bcy, bcz, tid
    cdef int NN = info.angles

    # Reflector arrays - read-only inside prange, updated sequentially below
    reflected_x[:,:,:,:] = 0.0
    reflected_y[:,:,:,:] = 0.0
    reflected_z[:,:,:,:] = 0.0

    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:

        flux[:,:,:] = 0.0
        thread_flux[:,:,:,:] = 0.0

        # Initialize per-angle known edges from boundary/reflector arrays
        for nn in range(NN):
            bcx = 0 if boundary_x.shape[3] == 1 else nn
            bcy = 0 if boundary_y.shape[3] == 1 else nn
            bcz = 0 if boundary_z.shape[3] == 1 else nn
            tools.initialize_known(known_x_work[nn], boundary_x[:,:,:,bcx], \
                    reflected_x, angle_x[nn], nn, info.bc_x[0], info.bc_x[1])
            tools.initialize_known(known_y_work[nn], boundary_y[:,:,:,bcy], \
                    reflected_y, angle_y[nn], nn, info.bc_y[0], info.bc_y[1])
            tools.initialize_known(known_z_work[nn], boundary_z[:,:,:,bcz], \
                    reflected_z, angle_z[nn], nn, info.bc_z[0], info.bc_z[1])

        # Parallel angular sweep, see square_ordinates in 2D
        for nn in prange(NN, schedule="static", num_threads=info.num_threads):
            qq = q_col[nn]
            tid = threadid()
            cube_sweep(thread_flux[tid], flux_old, xs_total, xs_scatter, \
                    off_scatter, external[:,:,:,qq], known_x_work[nn], \
                    known_y_work[nn], known_z_work[nn], medium_map, delta_x, \
                    delta_y, delta_z, angle_x[nn], angle_y[nn], angle_z[nn], \
                    angle_w[nn], info)

        # Sequential reduction into scalar flux
        for nn in range(info.num_threads):
            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        flux[ii,jj,kk] += thread_flux[nn,ii,jj,kk]

        # Update reflectors from exit edges left in known_{x,y,z}_work
        for nn in range(NN):
            tools.update_reflector(known_x_work[nn], reflected_x, \
                    known_y_work[nn], reflected_y, known_z_work[nn], \
                    reflected_z, angle_x, angle_y, angle_z, nn, info)

        change = tools.angle_convergence(flux, flux_old, info)
        converged = (change < info.tol_angular) or (count >= info.max_iter_angular)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]


cdef double spatial_coef(int spatial) noexcept nogil:
    if (spatial == 1):
        return 1.0
    elif (spatial == 2):
        return 0.0
    return 1.0


cdef void cube_sweep(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:,:]& off_scatter, \
        double[:,:,:]& external, double[:,:]& known_x, double[:,:]& known_y, \
        double[:,:]& known_z, int[:,:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& delta_z, double angle_x, double angle_y, \
        double angle_z, double angle_w, params info) noexcept nogil:
    # Single-angle sweep into a private flux array.  Thread-safe.
    # known_{x,y,z} start as the incoming edges on each face and end as the
    # outgoing edges.

    # Initialize iterables
    cdef int i0, j0, k0, ii, jj, kk, mat
    cdef double center, edge_x

    # Spatial discretization (W = alpha unless step characteristic)
    cdef double alpha = spatial_coef(info.spatial)
    cdef double coef_x, coef_y, coef_z, W_x, W_y, W_z

    # Sweep in the direction of the angle on each axis
    cdef double mu_x = fabs(angle_x)
    cdef double mu_y = fabs(angle_y)
    cdef double mu_z = fabs(angle_z)

    for k0 in range(info.cells_z):
        kk = k0 if angle_z > 0.0 else info.cells_z - 1 - k0

        for j0 in range(info.cells_y):
            jj = j0 if angle_y > 0.0 else info.cells_y - 1 - j0
            edge_x = known_x[jj,kk]

            for i0 in range(info.cells_x):
                ii = i0 if angle_x > 0.0 else info.cells_x - 1 - i0
                mat = medium_map[ii,jj,kk]

                # Angular coefficients
                coef_x = mu_x / delta_x[ii]
                coef_y = mu_y / delta_y[jj]
                coef_z = mu_z / delta_z[kk]

                if info.spatial == 3:
                    W_x = step_characteristic_weight(xs_total[mat] / coef_x)
                    W_y = step_characteristic_weight(xs_total[mat] / coef_y)
                    W_z = step_characteristic_weight(xs_total[mat] / coef_z)
                else:
                    W_x = alpha
                    W_y = alpha
                    W_z = alpha

                coef_x = 2.0 / (1.0 + W_x) * coef_x
                coef_y = 2.0 / (1.0 + W_y) * coef_y
                coef_z = 2.0 / (1.0 + W_z) * coef_z

                # Calculate flux center
                center = (coef_x * edge_x + coef_y * known_y[ii,kk] \
                        + coef_z * known_z[ii,jj] + xs_scatter[mat] \
                        * flux_old[ii,jj,kk] + external[ii,jj,kk] \
                        + off_scatter[ii,jj,kk]) \
                        / (xs_total[mat] + coef_x + coef_y + coef_z)

                # Update flux with cell centers
                flux[ii,jj,kk] += angle_w * center

                # Update known flux
                edge_x = (2.0 * center - (1.0 - W_x) * edge_x) / (1.0 + W_x)
                known_y[ii,kk] = (2.0 * center - (1.0 - W_y) * known_y[ii,kk]) \
                                / (1.0 + W_y)
                known_z[ii,jj] = (2.0 * center - (1.0 - W_z) * known_z[ii,jj]) \
                                / (1.0 + W_z)

            known_x[jj,kk] = edge_x


########################################################################
# Known Source Spatial Sweeps
#
# Single-pass sweeps with no convergence loop. The angles are swept in
# order so the reflected edges are used by the angles that follow.
########################################################################

cdef void _known_center_sweep(double[:,:,:,:]& flux, double[:]& xs_total, \
        double[:,:,:]& zero_3d, double[:,:,:,:]& source, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        params info):
    # Angular flux (cells_x, cells_y, cells_z, angles) for a known source

    # Initialize indices etc
    cdef int nn, qq, bcx, bcy, bcz
    cdef int NN = info.angles

    # Add known edge and reflector arrays
    known_x = tools.array_2d(info.cells_y, info.cells_z)
    known_y = tools.array_2d(info.cells_x, info.cells_z)
    known_z = tools.array_2d(info.cells_x, info.cells_y)
    reflected_x = tools.array_4d(2, info.cells_y, info.cells_z, NN)
    reflected_y = tools.array_4d(2, info.cells_x, info.cells_z, NN)
    reflected_z = tools.array_4d(2, info.cells_x, info.cells_y, NN)

    # Add zero placeholder
    zero_1d = tools.array_1d(info.materials)

    # Iterate over angles
    for nn in range(NN):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else nn
        bcx = 0 if boundary_x.shape[3] == 1 else nn
        bcy = 0 if boundary_y.shape[3] == 1 else nn
        bcz = 0 if boundary_z.shape[3] == 1 else nn

        # Initialize known x, y, and z
        tools.initialize_known(known_x, boundary_x[:,:,:,bcx], reflected_x, \
                               angle_x[nn], nn, info.bc_x[0], info.bc_x[1])
        tools.initialize_known(known_y, boundary_y[:,:,:,bcy], reflected_y, \
                               angle_y[nn], nn, info.bc_y[0], info.bc_y[1])
        tools.initialize_known(known_z, boundary_z[:,:,:,bcz], reflected_z, \
                               angle_z[nn], nn, info.bc_z[0], info.bc_z[1])

        # Perform spatial sweep - angular flux
        cube_sweep(flux[:,:,:,nn], zero_3d, xs_total, zero_1d, zero_3d, \
                source[:,:,:,qq], known_x, known_y, known_z, medium_map, \
                delta_x, delta_y, delta_z, angle_x[nn], angle_y[nn], \
                angle_z[nn], 1.0, info)

        # Save known_x, known_y, known_z into reflected
        tools.update_reflector(known_x, reflected_x, known_y, reflected_y, \
                               known_z, reflected_z, angle_x, angle_y, \
                               angle_z, nn, info)



cdef void _known_interface_sweep(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:,:,:]& flux_edge_z, \
        double[:]& xs_total, double[:,:,:,:]& source, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info):
    # Flux at the x, y, and z faces, (cells_x + 1, cells_y, cells_z, xdim),
    # (cells_x, cells_y + 1, cells_z, xdim), and (cells_x, cells_y,
    # cells_z + 1, xdim), where xdim is the number of angles for the angular
    # flux and 1 for the scalar flux

    # Initialize indices etc
    cdef int nn, qq, bcx, bcy, bcz
    cdef int NN = info.angles

    # Add dummy dimension to run both angular and scalar fluxes
    cdef int xdim = flux_edge_x.shape[3]

    # Add known edge and reflector arrays
    known_x = tools.array_2d(info.cells_y, info.cells_z)
    known_y = tools.array_2d(info.cells_x, info.cells_z)
    known_z = tools.array_2d(info.cells_x, info.cells_y)
    reflected_x = tools.array_4d(2, info.cells_y, info.cells_z, NN)
    reflected_y = tools.array_4d(2, info.cells_x, info.cells_z, NN)
    reflected_z = tools.array_4d(2, info.cells_x, info.cells_y, NN)

    # Iterate over angles
    for nn in range(NN):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[3] == 1 else nn
        bcx = 0 if boundary_x.shape[3] == 1 else nn
        bcy = 0 if boundary_y.shape[3] == 1 else nn
        bcz = 0 if boundary_z.shape[3] == 1 else nn

        # Initialize known x, y, and z
        tools.initialize_known(known_x, boundary_x[:,:,:,bcx], reflected_x, \
                               angle_x[nn], nn, info.bc_x[0], info.bc_x[1])
        tools.initialize_known(known_y, boundary_y[:,:,:,bcy], reflected_y, \
                               angle_y[nn], nn, info.bc_y[0], info.bc_y[1])
        tools.initialize_known(known_z, boundary_z[:,:,:,bcz], reflected_z, \
                               angle_z[nn], nn, info.bc_z[0], info.bc_z[1])

        if (xdim == 1):
            # Perform spatial sweep - scalar flux
            interface_sweep(flux_edge_x[:,:,:,0], flux_edge_y[:,:,:,0], \
                    flux_edge_z[:,:,:,0], xs_total, source[:,:,:,qq], known_x, \
                    known_y, known_z, medium_map, delta_x, delta_y, delta_z, \
                    angle_x[nn], angle_y[nn], angle_z[nn], angle_w[nn], info)
        else:
            # Perform spatial sweep - angular flux
            interface_sweep(flux_edge_x[:,:,:,nn], flux_edge_y[:,:,:,nn], \
                    flux_edge_z[:,:,:,nn], xs_total, source[:,:,:,qq], known_x, \
                    known_y, known_z, medium_map, delta_x, delta_y, delta_z, \
                    angle_x[nn], angle_y[nn], angle_z[nn], 1.0, info)

        # Save known_x, known_y, known_z into reflected
        tools.update_reflector(known_x, reflected_x, known_y, reflected_y, \
                               known_z, reflected_z, angle_x, angle_y, \
                               angle_z, nn, info)


cdef void interface_sweep(double[:,:,:]& flux_edge_x, \
        double[:,:,:]& flux_edge_y, double[:,:,:]& flux_edge_z, \
        double[:]& xs_total, double[:,:,:]& external, double[:,:]& known_x, \
        double[:,:]& known_y, double[:,:]& known_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double angle_x, double angle_y, double angle_z, double angle_w, \
        params info):
    # Same as cube_sweep without scattering, adding angle_w times the edges
    # to the face fluxes. The incoming edges are added at the first cell
    # along each axis and the outgoing edges at every cell.

    # Initialize iterables
    cdef int i0, j0, k0, ii, jj, kk, mat
    cdef int in_x, in_y, in_z, out_x, out_y, out_z
    cdef double center, edge_x

    # Spatial discretization (W = alpha unless step characteristic)
    cdef double alpha = spatial_coef(info.spatial)
    cdef double coef_x, coef_y, coef_z, W_x, W_y, W_z

    # Sweep in the direction of the angle on each axis
    cdef double mu_x = fabs(angle_x)
    cdef double mu_y = fabs(angle_y)
    cdef double mu_z = fabs(angle_z)

    for k0 in range(info.cells_z):
        kk = k0 if angle_z > 0.0 else info.cells_z - 1 - k0
        in_z = kk if angle_z > 0.0 else kk + 1
        out_z = kk + 1 if angle_z > 0.0 else kk

        for j0 in range(info.cells_y):
            jj = j0 if angle_y > 0.0 else info.cells_y - 1 - j0
            in_y = jj if angle_y > 0.0 else jj + 1
            out_y = jj + 1 if angle_y > 0.0 else jj
            edge_x = known_x[jj,kk]

            for i0 in range(info.cells_x):
                ii = i0 if angle_x > 0.0 else info.cells_x - 1 - i0
                in_x = ii if angle_x > 0.0 else ii + 1
                out_x = ii + 1 if angle_x > 0.0 else ii
                mat = medium_map[ii,jj,kk]

                # Incoming boundary edges
                if i0 == 0:
                    flux_edge_x[in_x,jj,kk] += angle_w * edge_x
                if j0 == 0:
                    flux_edge_y[ii,in_y,kk] += angle_w * known_y[ii,kk]
                if k0 == 0:
                    flux_edge_z[ii,jj,in_z] += angle_w * known_z[ii,jj]

                # Angular coefficients
                coef_x = mu_x / delta_x[ii]
                coef_y = mu_y / delta_y[jj]
                coef_z = mu_z / delta_z[kk]

                if info.spatial == 3:
                    W_x = step_characteristic_weight(xs_total[mat] / coef_x)
                    W_y = step_characteristic_weight(xs_total[mat] / coef_y)
                    W_z = step_characteristic_weight(xs_total[mat] / coef_z)
                else:
                    W_x = alpha
                    W_y = alpha
                    W_z = alpha

                coef_x = 2.0 / (1.0 + W_x) * coef_x
                coef_y = 2.0 / (1.0 + W_y) * coef_y
                coef_z = 2.0 / (1.0 + W_z) * coef_z

                # Calculate flux center
                center = (coef_x * edge_x + coef_y * known_y[ii,kk] \
                        + coef_z * known_z[ii,jj] + external[ii,jj,kk]) \
                        / (xs_total[mat] + coef_x + coef_y + coef_z)

                # Update known flux
                edge_x = (2.0 * center - (1.0 - W_x) * edge_x) / (1.0 + W_x)
                known_y[ii,kk] = (2.0 * center - (1.0 - W_y) * known_y[ii,kk]) \
                                / (1.0 + W_y)
                known_z[ii,jj] = (2.0 * center - (1.0 - W_z) * known_z[ii,jj]) \
                                / (1.0 + W_z)

                # Outgoing edges
                flux_edge_x[out_x,jj,kk] += angle_w * edge_x
                flux_edge_y[ii,out_y,kk] += angle_w * known_y[ii,kk]
                flux_edge_z[ii,jj,out_z] += angle_w * known_z[ii,jj]

            known_x[jj,kk] = edge_x


########################################################################
# Anisotropic Scattering (spherical harmonic moments)
########################################################################
# flux and flux_old are (cells_x, cells_y, cells_z, moments), xs_scatter is
# the within-group (materials, moments) and off_scatter is the off-group
# (cells_x, cells_y, cells_z, moments). Each ordinate builds its scattering
# source and reuses the isotropic sweep with zero scattering, then adds
# w_n Y_k(Omega_n) psi_n to the moments.
#
# Work arrays, see discrete_ordinates.
#   thread_psi    (num_threads, cells_x, cells_y, cells_z)          angular flux
#   thread_source (num_threads, cells_x, cells_y, cells_z)          angular source
#   thread_flux   (num_threads, cells_x, cells_y, cells_z, moments) per-thread moments
#   zero_3d (cells_x, cells_y, cells_z), zero_xs (materials)        read-only zeros
#   reflected_{x,y,z}, known_{x,y,z}_work                           see above

cdef void discrete_ordinates_aniso(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, double[:]& xs_total, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:,:]& P_source, double[:,:]& P_weights, \
        params info):
    # Allocate the work arrays for a single sweep
    cdef int NN = info.angles
    thread_psi = tools.array_4d(info.num_threads, info.cells_x, \
                                info.cells_y, info.cells_z)
    thread_source = tools.array_4d(info.num_threads, info.cells_x, \
                                   info.cells_y, info.cells_z)
    thread_flux = tools.array_5d(info.num_threads, info.cells_x, \
                                 info.cells_y, info.cells_z, info.moments)
    zero_3d = tools.array_3d(info.cells_x, info.cells_y, info.cells_z)
    zero_xs = tools.array_1d(info.materials)
    reflected_x = tools.array_4d(2, info.cells_y, info.cells_z, NN)
    reflected_y = tools.array_4d(2, info.cells_x, info.cells_z, NN)
    reflected_z = tools.array_4d(2, info.cells_x, info.cells_y, NN)
    known_x_work = tools.array_3d(NN, info.cells_y, info.cells_z)
    known_y_work = tools.array_3d(NN, info.cells_x, info.cells_z)
    known_z_work = tools.array_3d(NN, info.cells_x, info.cells_y)
    discrete_ordinates_aniso_nogil(flux, flux_old, xs_total, xs_scatter, \
                off_scatter, external, q_col, boundary_x, boundary_y, \
                boundary_z, medium_map, delta_x, delta_y, delta_z, angle_x, \
                angle_y, angle_z, P_source, P_weights, info, thread_psi, \
                thread_source, thread_flux, zero_3d, zero_xs, reflected_x, \
                reflected_y, reflected_z, known_x_work, known_y_work, \
                known_z_work)


cdef void discrete_ordinates_aniso_nogil(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, double[:]& xs_total, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:,:]& P_source, double[:,:]& P_weights, \
        params info, double[:,:,:,:]& thread_psi, \
        double[:,:,:,:]& thread_source, double[:,:,:,:,:]& thread_flux, \
        double[:,:,:]& zero_3d, double[:]& zero_xs, \
        double[:,:,:,:]& reflected_x, double[:,:,:,:]& reflected_y, \
        double[:,:,:,:]& reflected_z, double[:,:,:]& known_x_work, \
        double[:,:,:]& known_y_work, double[:,:,:]& known_z_work) noexcept nogil:
    # Rectangular spatial cells (SLAB3D = 4)
    if info.geometry == 4:
        cube_ordinates_aniso(flux, flux_old, xs_total, xs_scatter, \
                off_scatter, external, q_col, boundary_x, boundary_y, \
                boundary_z, medium_map, delta_x, delta_y, delta_z, angle_x, \
                angle_y, angle_z, P_source, P_weights, info, thread_psi, \
                thread_source, thread_flux, zero_3d, zero_xs, reflected_x, \
                reflected_y, reflected_z, known_x_work, known_y_work, \
                known_z_work)


cdef void _angle_source(double[:,:,:]& source, double[:,:,:,:]& flux_old, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:]& external, int[:,:,:]& medium_map, double[:]& P_source, \
        params info) noexcept nogil:
    # sum_k (2l + 1) Y_k(Omega) (sigma_l * phi_k + off_k) + external
    cdef int ii, jj, kk, mm, mat
    cdef double one_angle
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                one_angle = external[ii,jj,kk]
                for mm in range(info.moments):
                    one_angle += P_source[mm] * (xs_scatter[mat,mm] \
                            * flux_old[ii,jj,kk,mm] + off_scatter[ii,jj,kk,mm])
                source[ii,jj,kk] = one_angle


cdef void _project_moments(double[:,:,:,:]& flux, double[:,:,:]& psi, \
        double[:]& P_weights, params info) noexcept nogil:
    # Add w_n Y_k(Omega_n) psi_n to the moments and reset psi for the next angle
    cdef int ii, jj, kk, mm
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for mm in range(info.moments):
                    flux[ii,jj,kk,mm] += P_weights[mm] * psi[ii,jj,kk]
                psi[ii,jj,kk] = 0.0


cdef void cube_ordinates_aniso(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, double[:]& xs_total, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:,:]& P_source, double[:,:]& P_weights, \
        params info, double[:,:,:,:]& thread_psi, \
        double[:,:,:,:]& thread_source, double[:,:,:,:,:]& thread_flux, \
        double[:,:,:]& zero_3d, double[:]& zero_xs, \
        double[:,:,:,:]& reflected_x, double[:,:,:,:]& reflected_y, \
        double[:,:,:,:]& reflected_z, double[:,:,:]& known_x_work, \
        double[:,:,:]& known_y_work, double[:,:,:]& known_z_work) noexcept nogil:

    cdef int nn, ii, jj, kk, mm, qq, bcx, bcy, bcz, tid
    cdef int NN = info.angles

    # Angular flux starts at zero, _project_moments resets it per angle
    thread_psi[:,:,:,:] = 0.0

    # Reflector and per-angle known-edge arrays, see cube_ordinates
    reflected_x[:,:,:,:] = 0.0
    reflected_y[:,:,:,:] = 0.0
    reflected_z[:,:,:,:] = 0.0

    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:

        flux[:,:,:,:] = 0.0
        thread_flux[:,:,:,:,:] = 0.0

        # Initialize per-angle known edges from boundary/reflector arrays
        for nn in range(NN):
            bcx = 0 if boundary_x.shape[3] == 1 else nn
            bcy = 0 if boundary_y.shape[3] == 1 else nn
            bcz = 0 if boundary_z.shape[3] == 1 else nn
            tools.initialize_known(known_x_work[nn], boundary_x[:,:,:,bcx], \
                    reflected_x, angle_x[nn], nn, info.bc_x[0], info.bc_x[1])
            tools.initialize_known(known_y_work[nn], boundary_y[:,:,:,bcy], \
                    reflected_y, angle_y[nn], nn, info.bc_y[0], info.bc_y[1])
            tools.initialize_known(known_z_work[nn], boundary_z[:,:,:,bcz], \
                    reflected_z, angle_z[nn], nn, info.bc_z[0], info.bc_z[1])

        # Parallel angular sweep, see cube_ordinates
        for nn in prange(NN, schedule="static", num_threads=info.num_threads):
            qq = q_col[nn]
            tid = threadid()
            _angle_source(thread_source[tid], flux_old, xs_scatter, \
                    off_scatter, external[:,:,:,qq], medium_map, P_source[:,nn], \
                    info)
            cube_sweep(thread_psi[tid], zero_3d, xs_total, zero_xs, zero_3d, \
                    thread_source[tid], known_x_work[nn], known_y_work[nn], \
                    known_z_work[nn], medium_map, delta_x, delta_y, delta_z, \
                    angle_x[nn], angle_y[nn], angle_z[nn], 1.0, info)
            _project_moments(thread_flux[tid], thread_psi[tid], \
                             P_weights[:,nn], info)

        # Sequential reduction into flux moments
        for nn in range(info.num_threads):
            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        for mm in range(info.moments):
                            flux[ii,jj,kk,mm] += thread_flux[nn,ii,jj,kk,mm]

        # Update reflectors from exit edges left in known_{x,y,z}_work
        for nn in range(NN):
            tools.update_reflector(known_x_work[nn], reflected_x, \
                    known_y_work[nn], reflected_y, known_z_work[nn], \
                    reflected_z, angle_x, angle_y, angle_z, nn, info)

        change = tools.angle_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        converged = (change < info.tol_angular) or (count >= info.max_iter_angular)
        count += 1
        flux_old[:,:,:,:] = flux[:,:,:,:]
