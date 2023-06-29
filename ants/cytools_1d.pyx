########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Functions needed for both fixed source, criticality, and 
# time-dependent problems in one-dimensional neutron transport 
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language = c++
# cython: profile=True

from libc.math cimport sqrt, pow, erfc, ceil
from cython.view cimport array as cvarray

from ants.parameters cimport params

########################################################################
# Memoryview functions
########################################################################
cdef double[:] array_1d(int dim1):
    dd1 = cvarray((dim1,), itemsize=sizeof(double), format="d")
    cdef double[:] arr = dd1
    arr[:] = 0.0
    return arr


cdef double[:,:] array_2d(int dim1, int dim2):
    dd2 = cvarray((dim1, dim2), itemsize=sizeof(double), format="d")
    cdef double[:,:] arr = dd2
    arr[:,:] = 0.0
    return arr


cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3):
    dd3 = cvarray((dim1, dim2, dim3), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] arr = dd3
    arr[:,:,:] = 0.0
    return arr


cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4):
    dd4 = cvarray((dim1, dim2, dim3, dim4), itemsize=sizeof(double), format="d")
    cdef double[:,:,:,:] arr = dd4
    arr[:,:,:,:] = 0.0
    return arr

########################################################################
# Convergence functions
########################################################################
cdef double group_convergence(double[:,:]& arr1, double[:,:]& arr2, params info):
    # Calculate the L2 convergence of the scalar flux in the energy loop
    cdef int ii, gg
    cdef double change = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            if arr1[ii,gg] == 0.0:
                continue
            change += pow((arr1[ii,gg] - arr2[ii,gg]) / arr1[ii,gg] \
                          / info.cells_x, 2)
    change = sqrt(change)
    return change


cdef double angle_convergence(double[:]& arr1, double[:]& arr2, params info):
    # Calculate the L2 convergence of the scalar flux in the ordinates loop
    cdef int ii
    cdef double change = 0.0
    for ii in range(info.cells_x):
        if arr1[ii] == 0.0:
            continue
        change += pow((arr1[ii] - arr2[ii]) / arr1[ii] / info.cells_x, 2)
    change = sqrt(change)
    return change

########################################################################
# Multigroup functions
########################################################################

cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, params info):
    # Initialize iterables
    cdef int ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig] += mat2[mat,og,ig]


cdef void _off_scatter(double[:,:]& flux, double[:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:]& off_scatter, params info, int group):
    # Initialize iterables
    cdef int ii, mat, og
    # Zero out previous values
    off_scatter[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(0, group):
            off_scatter[ii] += xs_matrix[mat,group,og] * flux[ii,og]
        for og in range(group + 1, info.groups):
            off_scatter[ii] += xs_matrix[mat,group,og] * flux_old[ii,og]


cdef void _source_total(double[:]& source, double[:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:]& medium_map, \
        double[:]& external, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, nn, ig, og, mat, loc
    # Zero out previous values
    source[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for nn in range(info.angles):
            for og in range(info.groups):
                loc = og + info.groups * (nn + ii * info.angles)
                for ig in range(info.groups):
                    source[loc] += flux[ii,ig] * xs_matrix[mat,og,ig]
                source[loc] += external[loc]


cdef double[:,:] _angular_to_scalar(double[:,:,:]& angular_flux,
        double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, nn, gg
    # Initialize scalar flux term
    scalar_flux = array_2d(info.cells_x + info.edges, info.groups)
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x + info.edges):
        for nn in range(info.angles):
            for gg in range(info.groups):
                scalar_flux[ii,gg] += angular_flux[ii,nn,gg] * angle_w[nn]
    return scalar_flux


########################################################################
# Time Dependent functions
########################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, params info):
    # Create sigma_t + 1 / (v * dt)
    cdef int mm, gg
    for gg in range(info.groups):
        for mm in range(info.materials):
            xs_total[mm,gg] += 1 / (velocity[gg] * info.dt)


cdef void _time_source_total(double[:]& source, double[:,:]& scalar_flux, \
        double[:,:,:]& angular_flux, double[:,:,:]& xs_matrix, \
        double[:]& velocity, int[:]& medium_map, double[:]& external, \
        params info):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Initialize iterables
    cdef int ii, nn, ig, og, mat, loc
    # Zero out previous values
    source[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for nn in range(info.angles):
            for og in range(info.groups):
                loc = og + info.groups * (nn + ii * info.angles)
                for ig in range(info.groups):
                    source[loc] += scalar_flux[ii,ig] * xs_matrix[mat,og,ig]
                source[loc] += external[loc] + angular_flux[ii,nn,og] \
                                * 1 / (velocity[og] * info.dt)


cdef void _time_source_star(double[:,:,:]& angular_flux, double[:]& q_star, \
        double[:]& external, double[:]& velocity, params info):
    # Combining the source (I x N x G) with the angular flux (I x N x G)
    # Initialize iterables
    cdef int ii, nn, gg, loc
    # Zero out previous values
    q_star[:] = 0.0
    for gg in range(info.groups):
        for nn in range(info.angles):
            for ii in range(info.cells_x):
                loc = gg + info.groups * (nn + ii * info.angles)
                q_star[loc] = external[loc] + angular_flux[ii,nn,gg] \
                                * 1 / (velocity[gg] * info.dt)


cdef void boundary_decay(double[:]& boundary_x, int step, params info):
    # Calculate elapsed time
    cdef double t = info.dt * step
    # Cycle through different decay processes
    if info.bcdecay_x == 0: # Do nothing
        pass
    elif info.bcdecay_x == 1: # Turn off after one step
        _decay_01(boundary_x, step, info)
    elif info.bcdecay_x == 2: # Step decay
        _decay_02(boundary_x, t, info)


cdef int _boundary_length(params info):
    # Initialize boundary length
    cdef int bc_length
    if info.bcdim_x == 1:
        bc_length = 2
    elif info.bcdim_x == 2:
        bc_length = 2 * info.groups
    elif info.bcdim_x == 3:
        bc_length = 2 * info.groups * info.angles
    return bc_length


cdef void _decay_01(double[:]& boundary_x, int step, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info)
    cdef double magnitude = 0.0 if step > 0 else 1.0
    for cell in range(bc_length):
        boundary_x[cell] = magnitude


cdef void _decay_02(double[:]& boundary_x, double t, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info)
    cdef double k, err_arg
    t *= 1e6 # Convert elapsed time
    for cell in range(bc_length):
        if boundary_x[cell] == 0.0:
            continue
        if t < 0.2:
            boundary_x[cell] = 1.
        else:
            k = ceil((t - 0.2) / 0.1)
            err_arg = (t - 0.1 * (1 + k)) / (0.01)
            boundary_x[cell] = pow(0.5, k) * (1 + 2 * erfc(err_arg))


########################################################################
# Criticality functions
########################################################################

cdef void _normalize_flux(double[:,:]& flux, params info):
    cdef int ii, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            keff += pow(flux[ii,gg], 2)
    keff = sqrt(keff)
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            flux[ii,gg] /= keff


cdef void _fission_source(double[:,:] flux, double[:,:,:] xs_fission, \
        double[:] source, int[:] medium_map, params info, double keff):
    # Calculate the fission source (I x G) for the power iteration
    # (keff^{-1} * sigma_f * phi)
    # Initialize iterables
    cdef int ii, mat, ig, og, loc
    # Zero out previous power source
    source[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            loc = og + ii * info.groups
            for ig in range(info.groups):
                source[loc] += flux[ii,ig] * xs_fission[mat,og,ig]
            source[loc] /= keff


cdef double _update_keffective(double[:,:] flux_new, double[:,:] flux_old, \
        double[:,:,:] xs_fission, int[:] medium_map, params info, double keff):
    # Initialize iterables
    cdef int ii, mat, ig, og
    # Initialize fission rates for 2 fluxes
    cdef double rate_new = 0.0
    cdef double rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            for ig in range(info.groups):
                rate_new += flux_new[ii,ig] * xs_fission[mat,og,ig]
                rate_old += flux_old[ii,ig] * xs_fission[mat,og,ig]
    return (rate_new * keff) / rate_old

########################################################################
# Nearby Problems Criticality functions
########################################################################

cdef void _nearby_fission_source(double[:,:]& flux, double[:,:,:]& xs_fission, \
        double[:]& source, double[:]& n_source, int[:]& medium_map, \
        params info, double keff):
    # Initialize iterables
    cdef int ii, mat, nn, ig, og, loc
    # Zero out previous power iteration
    source[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for nn in range(info.angles):
            for og in range(info.groups):
                for ig in range(info.groups):
                    loc = og + info.groups * (nn + ii * info.angles)
                    source[loc] += flux[ii,ig] * xs_fission[mat,og,ig] / keff
                # Add nearby residual
                loc = ig + info.groups * (nn + ii * info.angles)
                source[loc] += n_source[loc]


cdef double _nearby_keffective(double[:,:]& flux, double rate, params info):
    cdef int ii, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            keff += rate * flux[ii, gg]
    return keff

########################################################################
# Hybrid Method Time Dependent Problems
########################################################################

cdef void _hybrid_source_collided(double[:,:]& flux, double[:,:,:]& xs_scatter, \
        double[:]& source_c, int[:]& medium_map, int[:]& index_c, \
        params info_u, params info_c):
    # Initialize iterables
    cdef int ii, mat, og, ig
    # Create scalar flux scattering rate density
    scatter_rate = array_2d(info_u.cells_x + info_u.edges, info_u.groups)
    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_u.groups):
            for ig in range(info_u.groups):
                scatter_rate[ii,og] += flux[ii,ig] * xs_scatter[mat,og,ig]
    # Shrink to size G hat
    _reduce_hybrid_source(scatter_rate, source_c, index_c, info_u, info_c)


# Big to small
cdef void _reduce_hybrid_source(double[:,:]& scatter_rate, double[:]& source_c, \
        int[:]& index_c, params info_u, params info_c):
    # Initialize iterables
    cdef int ii, gg, loc
    # Zero out previous source
    source_c[:] = 0.0
    for ii in range(info_u.cells_x):
        for gg in range(info_u.groups):
            loc = index_c[gg] + ii * info_c.groups
            source_c[loc] += scatter_rate[ii,gg]


cdef void _hybrid_source_total(double[:,:]& flux_t, double[:,:]& flux_u, \
        double[:,:,:]& xs_matrix, double[:]& source, int[:]& medium_map, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
    # Initialize iterables
    cdef int ii, mat, nn, ig, og, loc
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    # source[:] = 0.0
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for nn in range(info_u.angles):
            for og in range(info_u.groups):
                loc = og + info_u.groups * (nn + ii * info_u.angles)
                for ig in range(info_u.groups):
                    source[loc] += (flux_t[ii,ig] + flux_u[ii,ig]) \
                                    * xs_matrix[mat,og,ig]


cdef void _expand_hybrid_source(double[:,:]& flux_t, double[:,:]& flux_c, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
    # Initialize iterables
    cdef int cell, gu, gc
    flux_t[:,:] = 0.0
    # Create uncollided flux size
    for cell in range(info_c.cells_x):
        for gc in range(info_c.groups):
            for gu in range(index_u[gc], index_u[gc+1]):
                flux_t[cell,gu] = flux_c[cell,gc] * factor_u[gu]