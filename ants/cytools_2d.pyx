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
# cython: profile=True
# distutils: language = c++

from libc.math cimport sqrt, pow, erfc, ceil
from cython.view cimport array as cvarray
from cython.parallel import prange

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

cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5):
    dd5 = cvarray((dim1, dim2, dim3, dim4, dim5), itemsize=sizeof(double), \
                    format="d")
    cdef double[:,:,:,:,:] arr = dd5
    arr[:,:,:,:,:] = 0.0
    return arr

########################################################################
# Convergence functions
########################################################################
cdef double group_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info):
    # Calculate the L2 convergence of the scalar flux in the energy loop
    cdef int ii, jj, gg
    cdef int cells = info.cells_x * info.cells_y
    cdef double change = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for gg in range(info.groups):
                if arr1[ii,jj,gg] == 0.0:
                    continue
                change += pow((arr1[ii,jj,gg] - arr2[ii,jj,gg]) \
                              / arr1[ii,jj,gg] / cells, 2)
    change = sqrt(change)
    return change


cdef double angle_convergence(double[:,:]& arr1, double[:,:]& arr2, params info):
    # Calculate the L2 convergence of the scalar flux in the ordinates loop
    cdef int ii, jj
    cdef int cells = info.cells_x * info.cells_y
    cdef double change = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            if arr1[ii,jj] == 0.0:
                continue
            change += pow((arr1[ii,jj] - arr2[ii,jj]) / arr1[ii,jj] / cells, 2)
    change = sqrt(change)
    return change

########################################################################
# Multigroup functions
########################################################################

cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
    double[:,:,:]& mat3, params info):
    # Initialize iterables
    cdef int ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig] = (mat2[mat,og,ig] + mat3[mat,og,ig])


cdef void _dmd_subtraction(double[:,:,:,:]& y_minus, double[:,:,:,:]& y_plus, \
        double[:,:,:]& flux, double[:,:,:]& flux_old, int kk, params info):
    # Initialize iterables
    cdef int ii, jj, gg
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):            
            for gg in range(info.groups):
                if (kk < info.dmd_k - 1):
                    y_minus[ii,jj,gg,kk] = (flux[ii,jj,gg] - flux_old[ii,jj,gg])
                
                if (kk > 0):
                    y_plus[ii,jj,gg,kk-1] = (flux[ii,jj,gg] - flux_old[ii,jj,gg])


cdef void _off_scatter(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:]& off_scatter, params info, int group):
    # Initialize iterables
    cdef int ii, jj, mat, og
    # Zero out previous values
    off_scatter[:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(0, group):
                off_scatter[ii,jj] += xs_matrix[mat,group,og] * flux[ii,jj,og]
            for og in range(group + 1, info.groups):
                off_scatter[ii,jj] += xs_matrix[mat,group,og] * flux_old[ii,jj,og]


cdef void _source_total(double[:,:,:,:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, nn_q, og_q
    cdef double one_group

    # Zero out previous values
    source[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] * xs_matrix[mat,og,ig]

                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    source[ii,jj,nn,og] += one_group
                    source[ii,jj,nn,og] += external[ii,jj,nn_q,og_q]


cdef void _angular_to_scalar(double[:,:,:,:]& angular_flux, \
        double[:,:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    scalar_flux[ii,jj,gg] += angular_flux[ii,jj,nn,gg] * angle_w[nn]


cdef void _angular_edge_to_scalar(double[:,:,:,:]& psi_x, double[:,:,:,:]& psi_y, \
        double[:,:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    scalar_flux[ii,jj,gg] += 0.25 * angle_w[nn] \
                            * (psi_x[ii,jj,nn,gg] + psi_x[ii+1,jj,nn,gg] \
                            + psi_y[ii,jj,nn,gg] + psi_y[ii,jj+1,nn,gg])


cdef void initialize_known_y(double[:]& known_y, double[:,:]& boundary_y, \
        double[:,:,:]& reflected_y, double[:]& angle_y, int angle, params info):
    # Initialize location
    cdef int loc
    # Zero out flux
    known_y[:] = 0.0
    # Update with reflected array
    if (info.bc_y[0] == 1) and (angle_y[angle] > 0.0):
        known_y[...] = reflected_y[0,:,angle]
    elif (info.bc_y[1] == 1) and (angle_y[angle] < 0.0):
        known_y[...] = reflected_y[1,:,angle]
    else:
        # Pick left / right location
        loc = 0 if angle_y[angle] > 0.0 else 1
        known_y[...] = boundary_y[loc,:]


cdef void initialize_known_x(double[:]& known_x, double[:,:]& boundary_x, \
        double[:,:,:]& reflected_x, double[:]& angle_x, int angle, params info):
    # Initialize location
    cdef int loc
    # Zero out flux
    known_x[:] = 0.0
    # Update with reflected array
    if (info.bc_x[0] == 1) and (angle_x[angle] > 0.0):
        known_x[...] = reflected_x[0,:,angle]
    elif (info.bc_x[1] == 1) and (angle_x[angle] < 0.0):
        known_x[...] = reflected_x[1,:,angle]
    else:
        # Pick left / right location
        loc = 0 if angle_x[angle] > 0.0 else 1
        known_x[...] = boundary_x[loc,:]


cdef void update_reflector(double[:]& known_x, double[:,:,:]& reflected_x, \
        double[:]& angle_x, double[:]& known_y, double[:,:,:]& reflected_y, \
        double[:]& angle_y, int angle, params info):
    # Initialize iterables
    cdef int opp_idx
    # Return nothing for 4 vacuum boundaries
    if (info.bc_x == [0, 0]) and (info.bc_y == [0, 0]):
        return
    # Update reflected_x
    if (angle_x[angle] > 0.0) and (info.bc_x[1] == 1):
        opp_idx = _reflected_index(angle_x, angle_y, angle, info)
        reflected_x[1,:,opp_idx] = known_x[:]
    elif (angle_x[angle] < 0.0) and (info.bc_x[0] == 1):
        opp_idx = _reflected_index(angle_x, angle_y, angle, info)
        reflected_x[0,:,opp_idx] = known_x[:]
    # Update reflected_y
    if (angle_y[angle] > 0.0) and (info.bc_y[1] == 1):
        opp_idx = _reflected_index(angle_y, angle_x, angle, info)
        reflected_y[1,:,opp_idx] = known_y[:]
    elif (angle_y[angle] < 0.0) and (info.bc_y[0] == 1):
        opp_idx = _reflected_index(angle_y, angle_x, angle, info)
        reflected_y[0,:,opp_idx] = known_y[:]


cdef int _reflected_index(double[:]& angle_opp, double[:]& angle_sim, \
        int angle, params info):
    # Initialize iterable
    cdef int nn
    for nn in range(info.angles * info.angles):
        if (angle_opp[nn] == -angle_opp[angle]) \
                and (angle_sim[nn] == angle_sim[angle]):
            return nn
    # Returns error
    return -1


########################################################################
# Time Dependent functions
########################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info):
    # Create sigma_t + 1 / (v * dt)
    cdef int mm, gg
    for gg in range(info.groups):
        for mm in range(info.materials):
            xs_total[mm,gg] += constant / (velocity[gg] * info.dt)


cdef void _time_source_star_bdf1(double[:,:,:,:]& flux, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& external, \
        double[:]& velocity, params info):
    # Combining the source (I x J x N^2 x G) with the angular 
    #     flux (I x J x N^2 x G)
    
    # Initialize iterables
    cdef int ii, jj, nn, gg, nn_q, gg_q
    cdef int directions = info.angles * info.angles

    # Zero out previous values
    q_star[:,:,:,:] = 0.0
    # Iterate over cells, angles, groups
    for nn in prange(directions, nogil=True):
        nn_q = 0 if external.shape[2] == 1 else nn
    
        for gg in range(info.groups):
            gg_q = 0 if external.shape[3] == 1 else gg
        
            for jj in range(info.cells_y):
                for ii in range(info.cells_x):
                    # loc = gg + info.groups * (nn + info.angles * info.angles \
                    #                         * (jj + ii * info.cells_y))
                    q_star[ii,jj,nn,gg] = external[ii,jj,nn_q,gg_q] \
                                        + flux[ii,jj,nn,gg] \
                                        * 1 / (velocity[gg] * info.dt)


cdef void _time_source_total_bdf1(double[:,:,:]& scalar, double[:,:,:,:]& angular, \
        double[:,:,:]& xs_matrix, double[:]& velocity, double[:,:,:,:]& qstar, \
        double[:,:,:,:]& external, int[:,:]& medium_map, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, nn_q, og_q
    cdef double one_group

    # Zero out previous values
    qstar[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += scalar[ii,jj,ig] * xs_matrix[mat,og,ig]

                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    qstar[ii,jj,nn,og] = one_group + external[ii,jj,nn_q,og_q] \
                            + angular[ii,jj,nn,og] * 1 / (velocity[og] * info.dt)


cdef void _time_source_star_cn(double[:,:,:,:]& psi_x, double[:,:,:,:]& psi_y, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external_prev, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double constant, params info):

    # Combining the source (I x N x G) with the angular flux (I x N x G)
    # external_prev is time step \ell, source is time step \ell + 1

    # Initialize iterables
    cdef int ii, jj, mat, nn, og, ig, nn_q, og_q
    cdef double one_group

    # Initialize angular flux center estimates
    cdef double psi, dpsi_x, dpsi_y

    # Zero out previous values
    q_star[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                one_group = 0.0
                # Add scalar term
                for ig in range(info.groups):
                    one_group += phi[ii,jj,ig] * xs_scatter[mat,og,ig]
                
                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    # Calculate angular flux center
                    psi = 0.25 * (psi_x[ii,jj,nn,og] + psi_x[ii+1,jj,nn,og] \
                               + psi_y[ii,jj,nn,og] + psi_y[ii,jj+1,nn,og])
                    # Calculate cell flux derivative
                    dpsi_x = (psi_x[ii+1,jj,nn,og] - psi_x[ii,jj,nn,og]) / delta_x[ii]
                    dpsi_y = (psi_y[ii,jj+1,nn,og] - psi_y[ii,jj,nn,og]) / delta_y[jj]
                    # Add angular terms
                    q_star[ii,jj,nn,og] += one_group + external[ii,jj,nn_q,og_q]
                    q_star[ii,jj,nn,og] += external_prev[ii,jj,nn_q,og_q] \
                            - angle_x[nn] * dpsi_x - angle_y[nn] * dpsi_y + psi \
                            * (constant / (velocity[og] * info.dt) - xs_total[mat,og]) 


cdef void _time_source_star_bdf2(double[:,:,:,:]& flux_1, \
        double[:,:,:,:]& flux_2, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external, double[:]& velocity, params info):
    # Combining the source (I x N x G) with the angular flux (I x N x G)
    # flux_1 is time step \ell - 1, flux_2 is time step \ell - 2
    # Initialize iterables
    cdef int ii, jj, nn, gg, nn_q, gg_q
    cdef int directions = info.angles * info.angles

    # Zero out previous values
    q_star[:,:,:,:] = 0.0

    # Iterate over all cells, angles, and groups
    for nn in prange(directions, nogil=True):
        nn_q = 0 if external.shape[2] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[3] == 1 else gg

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    # loc = gg + info.groups * (nn + info.angles * info.angles \
                    #     * (jj + ii * info.cells_y))
                    q_star[ii,jj,nn,gg] = external[ii,jj,nn_q,gg_q] \
                            + flux_1[ii,jj,nn,gg] * 2 / (velocity[gg] * info.dt) \
                            - flux_2[ii,jj,nn,gg] * 1 / (2 * velocity[gg] * info.dt)


cdef void _time_source_total_bdf2(double[:,:,:]& scalar, \
        double[:,:,:,:]& angular_1, double[:,:,:,:]& angular_2, \
        double[:,:,:]& xs_matrix, double[:]& velocity, double[:,:,:,:]& qstar, \
        double[:,:,:,:]& external, int[:,:]& medium_map, params info):

    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, nn_q, og_q
    cdef double one_group

    # Zero out previous values
    qstar[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += scalar[ii,jj,ig] * xs_matrix[mat,og,ig]

                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    qstar[ii,jj,nn,og] = one_group + external[ii,jj,nn_q,og_q] \
                            + angular_1[ii,jj,nn,og] * 2 / (velocity[og] * info.dt) \
                            - angular_2[ii,jj,nn,og] * 1 / (2 * velocity[og] * info.dt)


cdef void _time_source_star_tr_bdf2(double[:,:,:,:]& psi_x, \
        double[:,:,:,:]& psi_y, double[:,:,:,:]& flux_2, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& external, \
        double[:]& velocity, double gamma, params info):
    # Combining the source (I x J x N^2 x G) with the angular flux (I x J x N^2 x G)
    # psi_x is time step \ell (edges), flux_2 is time step \ell + gamma (centers)
    
    # Initialize iterables
    cdef int ii, jj, nn, gg, nn_q, gg_q
    cdef int directions = info.angles * info.angles

    # Initialize angular flux center
    cdef double psi

    # Zero out previous values
    q_star[:,:,:,:] = 0.0

    # Iterate over all cells, angles, and groups
    for nn in prange(directions, nogil=True):
        nn_q = 0 if external.shape[2] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[3] == 1 else gg

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    # loc = gg + info.groups * (nn + info.angles * info.angles \
                    #         * (jj + ii * info.cells_y))

                    psi = 0.25 * (psi_x[ii,jj,nn,gg] + psi_x[ii+1,jj,nn,gg] \
                               + psi_y[ii,jj,nn,gg] + psi_y[ii,jj+1,nn,gg])

                    q_star[ii,jj,nn,gg] = external[ii,jj,nn_q,gg_q] \
                            + flux_2[ii,jj,nn,gg] * 1 / (gamma * (1 - gamma) \
                                * velocity[gg] * info.dt) \
                            - psi * (1 - gamma) / (gamma * velocity[gg] * info.dt)


cdef void _time_right_side(double[:,:,:,:]& q_star, double[:,:,:]& flux, \
        double[:,:,:]& xs_scatter, int[:,:]& medium_map, params info):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat
    cdef double one_group
    # Iterate over dimensions
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]            
            for og in range(info.groups):
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] * xs_scatter[mat,og,ig]
                for nn in range(info.angles * info.angles):
                    q_star[ii,jj,nn,og] += one_group


########################################################################
# Criticality functions
########################################################################

cdef void _normalize_flux(double[:,:,:]& flux, params info):
    cdef int ii, jj, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for jj in range(info.cells_y):
            for ii in range(info.cells_x):
                keff += (flux[ii,jj,gg] * flux[ii,jj,gg])
    keff = sqrt(keff)
    for gg in range(info.groups):
        for jj in range(info.cells_y):
            for ii in range(info.cells_x):
                flux[ii,jj,gg] /= keff


cdef void _fission_source(double[:,:,:]& flux, double[:,:,:]& xs_fission, \
        double[:,:,:,:]& source, int[:,:]& medium_map, params info, \
        double keff):
    # Calculate the fission source (I x G) for the power iteration
    # (keff^{-1} * sigma_f * phi)
    # Initialize iterables
    cdef int ii, jj, mat, ig, og
    # Zero out previous power source
    source[:,:,:,:] = 0.0
    # Iterate over all cells and groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                # loc = og + info.groups * (jj + ii * info.cells_y)
                for ig in range(info.groups):
                    source[ii,jj,0,og] += flux[ii,jj,ig] * xs_fission[mat,og,ig]
                source[ii,jj,0,og] /= keff


cdef double _update_keffective(double[:,:,:] flux_new, double[:,:,:] flux_old, \
        double[:,:,:] xs_fission, int[:,:] medium_map, params info, double keff):
    # Initialize iterables
    cdef int ii, jj, mat, ig, og
    # Initialize fission rates for 2 fluxes
    cdef double rate_new = 0.0
    cdef double rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                for ig in range(info.groups):
                    rate_new += flux_new[ii,jj,ig] * xs_fission[mat,og,ig]
                    rate_old += flux_old[ii,jj,ig] * xs_fission[mat,og,ig]
    return (rate_new * keff) / rate_old


cdef void _source_total_critical(double[:,:,:,:]& source, \
        double[:,:,:]& flux, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:,:]& medium_map, double keff, \
        params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, ig, og, mat
    # Zero out previous values
    source[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                # loc = og + info.groups * (jj + ii * info.cells_y)
                for ig in range(info.groups):
                    source[ii,jj,0,og] += (flux[ii,jj,ig] * xs_scatter[mat,og,ig]) \
                               + (flux[ii,jj,ig] * xs_fission[mat,og,ig]) / keff


########################################################################
# Nearby Problems Criticality functions
########################################################################

cdef void _nearby_fission_source(double[:,:,:]& flux, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& source, \
        double[:,:,:,:]& residual, int[:,:]& medium_map, params info, \
        double keff):
    # Initialize iterables
    cdef int ii, jj, mat, nn, ig, og, nn_r
    cdef double one_group
    # Zero out previous power iteration
    source[:,:,:,:] = 0.0
    # Iterate over all cells, angles, and groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] / keff * xs_fission[mat,og,ig]
                for nn in range(info.angles * info.angles):
                    # Add nearby residual
                    nn_r = 0 if residual.shape[2] == 1 else nn
                    source[ii,jj,nn,og] += one_group + residual[ii,jj,nn_r,og]


cdef double _nearby_keffective(double[:,:,:]& flux, double rate, params info):
    cdef int ii, jj, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                keff += rate * flux[ii,jj,gg]
    return keff

########################################################################
# Hybrid Method Time Dependent Problems
########################################################################

cdef void _hybrid_source_collided(double[:,:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:,:]& source_c, \
        int[:,:]& medium_map, int[:]& coarse_idx, params info_u):
    
    # Initialize iterables
    cdef int ii, jj, mat, og, ig
    cdef double one_group
    
    # Zero out previous source
    source_c[:,:,:,:] = 0.0
    
    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info_u.groups):
                one_group = 0.0
                for ig in range(info_u.groups):
                    one_group += flux[ii,jj,ig] * xs_scatter[mat,og,ig]
                source_c[ii,jj,0,coarse_idx[og]] += one_group


cdef void _hybrid_source_total(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        double[:,:,:]& xs_matrix, double[:,:,:,:]& source, int[:,:]& medium_map, \
        int[:]& coarse_idx, double[:]& factor_u, params info_u, params info_c):
    # Initialize iterables
    cdef int ii, jj, mat, nn, ig, og
    cdef double one_group
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_x):
            mat = medium_map[ii,jj]
            # Combine fluxes
            for og in range(info_u.groups):
                flux_u[ii,jj,og] = flux_u[ii,jj,og] \
                            + flux_c[ii,jj,coarse_idx[og]] * factor_u[og]
            # Add flux-xs product to source
            for og in range(info_u.groups):
                one_group = 0.0
                for ig in range(info_u.groups):
                    one_group += flux_u[ii,jj,ig] * xs_matrix[mat,og,ig]
                for nn in range(info_u.angles * info_u.angles):
                    source[ii,jj,nn,og] += one_group


# cdef void _hybrid_source_total(double[:,:,:]& flux_t, double[:,:,:]& flux_u, \
#         double[:,:,:]& xs_matrix, double[:,:,:,:]& source, int[:,:]& medium_map, \
#         int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
#     # Initialize iterables
#     cdef int ii, jj, mat, nn, NN, ig, og#, loc
#     # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
#     # Get all angular directions
#     NN = info_u.angles * info_u.angles
#     # source[:] = 0.0
#     for ii in range(info_u.cells_x):
#         for jj in range(info_u.cells_x):
#             mat = medium_map[ii,jj]
#             for nn in range(NN):
#                 for og in range(info_u.groups):
#                     # loc = og + info_u.groups * (nn + NN * (jj + ii * info_u.cells_y))
#                     for ig in range(info_u.groups):
#                         source[ii,jj,nn,og] += (flux_t[ii,jj,ig] + flux_u[ii,jj,ig]) \
#                                             * xs_matrix[mat,og,ig]


# cdef void _expand_hybrid_source(double[:,:,:]& flux_t, double[:,:,:]& flux_c, \
#         int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
#     # Initialize iterables
#     cdef int ii, jj, gu, gc
#     flux_t[:,:,:] = 0.0
#     # Create uncollided flux size
#     for ii in range(info_c.cells_x):
#         for jj in range(info_c.cells_y):
#             for gc in range(info_c.groups):
#                 for gu in range(index_u[gc], index_u[gc+1]):
#                     flux_t[ii,jj,gu] = flux_c[ii,jj,gc] * factor_u[gu]
