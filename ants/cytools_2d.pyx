################################################################################
#                            ___    _   _____________
#                           /   |  / | / /_  __/ ___/
#                          / /| | /  |/ / / /  \__ \
#                         / ___ |/ /|  / / /  ___/ /
#                        /_/  |_/_/ |_/ /_/  /____/
#
# Functions needed for both fixed source, criticality, and time-dependent
# problems in one-dimensional neutron transport
#
################################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.math cimport sqrt

from cython.parallel import prange

from cython.view cimport array as cvarray

from ants.cytools_1d cimport _legendre_sum
from ants.cytools_1d cimport _xs_matrix_aniso as _shared_xs_matrix_aniso
from ants.cytools_shared cimport _fission_matrix as _shared_fission_matrix
from ants.cytools_shared cimport _normalize_flux as _shared_normalize_flux
from ants.cytools_shared cimport _source_columns as _shared_source_columns
from ants.cytools_shared cimport _total_velocity as _shared_total_velocity
from ants.cytools_shared cimport _update_keffective as _shared_update_keffective
from ants.cytools_shared cimport angle_convergence as _shared_angle_convergence
from ants.cytools_shared cimport array_1d as _shared_array_1d
from ants.cytools_shared cimport array_2d as _shared_array_2d
from ants.cytools_shared cimport array_3d as _shared_array_3d
from ants.cytools_shared cimport array_4d as _shared_array_4d
from ants.cytools_shared cimport array_5d as _shared_array_5d
from ants.cytools_shared cimport farray_4d as _shared_farray_4d
from ants.cytools_shared cimport farray_5d as _shared_farray_5d
from ants.cytools_shared cimport group_convergence as _shared_group_convergence
from ants.parameters cimport params


################################################################################
# Memoryview functions -- delegates to cytools_shared
################################################################################
cdef double[:] array_1d(int dim1):
    return _shared_array_1d(dim1)


cdef double[:,:] array_2d(int dim1, int dim2):
    return _shared_array_2d(dim1, dim2)


cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3):
    return _shared_array_3d(dim1, dim2, dim3)


cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4):
    return _shared_array_4d(dim1, dim2, dim3, dim4)


cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5):
    return _shared_array_5d(dim1, dim2, dim3, dim4, dim5)


cdef float[:,:,:,:] farray_4d(int dim1, int dim2, int dim3, int dim4):
    return _shared_farray_4d(dim1, dim2, dim3, dim4)


cdef float[:,:,:,:,:] farray_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5):
    return _shared_farray_5d(dim1, dim2, dim3, dim4, dim5)

################################################################################
# Convergence functions
################################################################################
cdef double group_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info):
    cdef double[:,:,:] _arr1 = arr1
    cdef double[:,:,:] _arr2 = arr2
    return _shared_group_convergence(_arr1, _arr2, info)


cdef double angle_convergence(double[:,:]& arr1, double[:,:]& arr2, params info):
    cdef double[:,:] _arr1 = arr1
    cdef double[:,:] _arr2 = arr2
    return _shared_angle_convergence(_arr1, _arr2, info)

################################################################################
# Multigroup functions
################################################################################

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
                if (kk < info.dmd_snapshots - 1):
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


cdef void _off_scatter_jacobi(double[:,:,:]& flux_old, int[:,:]& medium_map, \
        double[:,:,:]& xs_matrix, double[:,:,:]& off_scatter_all, \
        params info, int group) noexcept nogil:
    # Jacobi variant: uses flux_old for ALL off-diagonal groups so that each
    # group's scattering source is independent of the sweep order.
    cdef int ii, jj, mat, og
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            off_scatter_all[group, ii, jj] = 0.0
            mat = medium_map[ii, jj]
            for og in range(info.groups):
                if og != group:
                    off_scatter_all[group, ii, jj] += xs_matrix[mat, group, og] * flux_old[ii, jj, og]


cdef void _source_total(double[:,:,:,:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, params info, int[:,:] q_index=None):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, og_q
    cdef double one_group
    # External source column of each angle and group
    cdef int[:,:] q_cols = _source_columns(external.shape[2], q_index, info)

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
                    source[ii,jj,nn,og] += one_group
                    source[ii,jj,nn,og] += external[ii,jj,q_cols[nn,og],og_q]


cdef int[:,:] _source_columns(int columns, int[:,:] q_index, params info):
    # External source column for each (angle, group), see cytools_shared
    return _shared_source_columns(columns, q_index, info.angles * info.angles, \
                                  info.groups)


cdef void _source_total_single(double[:,:,:,:]& source, \
        double[:,:,:]& flux, double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, int group, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, ig, og, mat, og_q
    cdef double one_group

    # Zero out previous values
    source[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            for og in range(group, group + 1):
                og_q = 0 if external.shape[3] == 1 else og
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] * xs_matrix[mat,og,ig]

                source[ii,jj,0,0] += one_group
                source[ii,jj,0,0] += external[ii,jj,0,og_q]


cdef void _source_total_nsingle(double[:,:,:,:]& source, \
        double[:,:,:]& flux, double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, int group, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, og_q
    cdef double one_group

    # Zero out previous values
    source[:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            for og in range(group, group + 1):
                og_q = 0 if external.shape[3] == 1 else og
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] * xs_matrix[mat,og,ig]

                for nn in range(info.angles * info.angles):
                    source[ii,jj,nn,0] += one_group
                    source[ii,jj,nn,0] += external[ii,jj,nn,og_q]


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


################################################################################
# Time Dependent functions
################################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info):
    cdef double[:,:] _xs_total = xs_total
    cdef double[:] _velocity = velocity
    _shared_total_velocity(_xs_total, _velocity, constant, info)


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
    for nn in range(directions): #, nogil=True):
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
    for nn in range(directions): #, nogil=True):
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
    for nn in range(directions): #, nogil=True):
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

cdef void _time_source_star_tr_bdf2_mem(double[:,:,:,:]& psi_x, double[:,:,:,:]& psi_y, \
        double[:,:,:,:]& flux_2, double[:,:,:,:]& external, double[:]& velocity, \
        double gamma, params info):
    # Combining the source (I x J x N^2 x G) with the angular flux (I x J x N^2 x G)
    # psi_x is time step \ell (edges), flux_2 is time step \ell + gamma (centers)

    # Initialize iterables
    cdef int ii, jj, nn, gg, nn_q, gg_q
    cdef int directions = info.angles * info.angles
    cdef double gamma_vel_01, gamma_vel_02

    # Initialize angular flux center
    cdef double psi

    # Iterate over all cells, angles, and groups
    for nn in range(directions): #, nogil=True):
        nn_q = 0 if external.shape[2] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[3] == 1 else gg
            gamma_vel_01 = 1 / (gamma * (1 - gamma) * velocity[gg] * info.dt)
            gamma_vel_02 = (1 - gamma) / (gamma * velocity[gg] * info.dt)

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    psi = 0.25 * (psi_x[ii,jj,nn,gg] + psi_x[ii+1,jj,nn,gg] \
                               + psi_y[ii,jj,nn,gg] + psi_y[ii,jj+1,nn,gg])

                    flux_2[ii,jj,nn,gg] = flux_2[ii,jj,nn,gg] * gamma_vel_01 \
                                    + external[ii,jj,nn_q,gg_q] - psi * gamma_vel_02


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


################################################################################
# Criticality functions
################################################################################

cdef double[:,:,:] _fission_matrix(object fission, object chi):
    return _shared_fission_matrix(fission, chi)


cdef void _normalize_flux(double[:,:,:]& flux, params info):
    cdef double[:,:,:] _flux = flux
    _shared_normalize_flux(_flux, info)


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
    return _shared_update_keffective(flux_new, flux_old, xs_fission, medium_map, info, keff)


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


################################################################################
# Nearby Problems
################################################################################

cdef void _nearby_flux_to_scalar(double[:,:,:]& scalar_flux, \
        double[:,:]& angular_spatial, double angle_w, int gg, params info):
    # Initialize iterables
    cdef int ii, jj
    # Iterate over all spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            scalar_flux[ii,jj,gg] += angular_spatial[ii,jj] * angle_w


cdef void _nearby_boundary_to_scalar(double[:,:,:]& boundary_flux, \
        double[:,:]& angular_spatial, double angle_w, int gg, params info):
    # Initialize iterables
    cdef int ii
    cdef int cells = boundary_flux.shape[1]
    # Iterate over all spatial cells
    for ii in range(cells):
        boundary_flux[0,ii,gg] += angular_spatial[0,ii] * angle_w
        boundary_flux[1,ii,gg] += angular_spatial[1,ii] * angle_w


cdef void _nearby_off_scatter(double[:,:,:]& residual, \
    double[:,:,:]& scalar_flux, double[:,:,:]& xs_scatter, \
    double[:,:,:]& xs_fission, int[:,:]& medium_map, params info):

    # Initialize iterables
    cdef int ii, jj, mat, og, ig

    # Initialize off-scattering term
    cdef float off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over groups
            for og in range(info.groups):
                off_scatter = 0.0
                for ig in range(info.groups):
                    off_scatter += scalar_flux[ii,jj,ig] \
                                * (xs_scatter[mat,og,ig] + xs_fission[mat,og,ig])
                residual[ii,jj,og] -= (off_scatter)


cdef void _nearby_on_scatter(double[:,:,:]& residual, double[:,:]& int_angular, \
        double[:,:]& int_dx_angular, double[:,:]& int_dy_angular,
        double[:,:]& xs_total, double[:,:,:]& external, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double angle_x, double angle_y, \
        double angle_w, int gg0, int gg1, params info):

    # Initialize iterables
    cdef int ii, jj, mat

    # Allow for energy independent sources
    cdef int gg_q = 0 if external.shape[2] == 1 else gg1

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            residual[ii,jj,gg0] += angle_w * ((angle_x * int_dx_angular[ii,jj]) \
                            + (angle_y * int_dy_angular[ii,jj]) \
                            + (int_angular[ii,jj] * xs_total[mat,gg1]) \
                            - (external[ii,jj,gg_q] * delta_x[ii] * delta_y[jj]))


cdef void _nearby_populate_outer(double[:,:,:,:]& modified_flux, \
        double[:,:,:,:]& flux_center, double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg
    # Zero out modified flux
    modified_flux[:,:,:,:] = 0.0
    # Iterate over all cells, angles, and groups
    for gg in range(info.groups):
        for nn in range(info.angles * info.angles):
            for ii in range(info.cells_x):
                # Medium edges
                if (info.bc_x[0] == 1):
                    modified_flux[ii+1,0,nn,gg] = flux_edge_y[ii,0,nn,gg]
                if (info.bc_x[1] == 1):
                    modified_flux[ii+1,info.cells_y+1,nn,gg] = flux_edge_y[ii,info.cells_y,nn,gg]

                # Center fluxes
                for jj in range(info.cells_y):
                    modified_flux[ii+1,jj+1,nn,gg] = flux_center[ii,jj,nn,gg]

            # Medium edges
            for jj in range(info.cells_y):
                if (info.bc_y[0] == 1):
                    modified_flux[0,jj+1,nn,gg] = flux_edge_x[0,jj,nn,gg]
                if (info.bc_y[1] == 1):
                    modified_flux[info.cells_x+1,jj+1,nn,gg] = flux_edge_x[info.cells_x,jj,nn,gg]

            # # Corner average (0, 0)
            # modified_flux[0,0,nn,gg] = 0.5 * (modified_flux[1,0,nn,gg] \
            #                                 + modified_flux[0,1,nn,gg])
            # # Corner average (-1, -1)
            # modified_flux[info.cells_x+1,info.cells_y+1,nn,gg] = 0.5 \
            #                     * (modified_flux[info.cells_x,info.cells_y+1,nn,gg] \
            #                     + modified_flux[info.cells_x+1,info.cells_y,nn,gg])
            # # Corner average (0, -1)
            # modified_flux[0,info.cells_y+1,nn,gg] = 0.5 \
            #                     * (modified_flux[1,info.cells_y+1,nn,gg] \
            #                     + modified_flux[0,info.cells_y,nn,gg])
            # # Corner average (-1, 0)
            # modified_flux[info.cells_x+1,0,nn,gg] = 0.5 \
            #                     * (modified_flux[info.cells_x,0,nn,gg] \
            #                     + modified_flux[info.cells_x+1,1,nn,gg])


cdef void _nearby_sum_4d(double[:,:,:,:]& modified_flux, \
        double[:,:,:,:]& flux_center, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg, ii_idx, jj_idx
    cdef int length_x = modified_flux.shape[0]
    cdef int length_y = modified_flux.shape[1]
    # Zero out modified flux
    flux_center[:,:,:,:] = 0.0
    # Iterate over all cells, angles, and groups
    for ii in range(length_x):
        # Get correct index - x
        ii_idx = _find_outer_index(ii, length_x)

        for jj in range(length_y):
            # Get correct index - y
            jj_idx = _find_outer_index(jj, length_y)

            # Iterate over angles and groups
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    flux_center[ii_idx,jj_idx,nn,gg] += modified_flux[ii,jj,nn,gg]


cdef double[:,:] _nearby_sum_2d(double[:,:]& modified_arr, params info):
    # Initialize iterables
    cdef int ii, jj, ii_idx, jj_idx
    cdef int length_x = modified_arr.shape[0]
    cdef int length_y = modified_arr.shape[1]
    # Create (cells_x * cells_y) array
    arr = array_2d(info.cells_x, info.cells_y)
    # Iterate over all cells
    for ii in range(length_x):
        # Get correct index - x
        ii_idx = _find_outer_index(ii, length_x)

        for jj in range(length_y):
            # Get correct index - y
            jj_idx = _find_outer_index(jj, length_y)

            # Sum array
            arr[ii_idx,jj_idx] += modified_arr[ii,jj]

    return arr


cdef double[:,:] _nearby_sum_bc(double[:,:]& modified_arr, params info):
    # Initialize iterables
    cdef int ii, ii_idx
    cdef int length = modified_arr.shape[1]
    # Create (2 * cells_y) array
    arr = array_2d(2, length - 2)
    # Iterate over all cells, angles, and groups
    for ii in range(length):
        # Get correct index
        ii_idx = _find_outer_index(ii, length)
        # Sum array
        arr[0,ii_idx] += modified_arr[0,ii]
        arr[1,ii_idx] += modified_arr[1,ii]

    return arr


cdef int _find_outer_index(int iterable, int length):
    if (iterable < 2):
        return 0
    elif (iterable > (length - 3)):
        return length - 3
    return iterable - 1


cdef int[:,:] _nearby_adjust_medium(int[:,:] medium_map, params info):
    # Initialize iterables
    cdef int ii, jj, ii_idx, jj_idx
    # Create new medium map
    dd2 = cvarray((info.cells_x + 2, info.cells_y + 2), itemsize=sizeof(int), format="i")
    cdef int[:,:] modified_medium_map = dd2
    # Iterate over all cells
    for ii in range(info.cells_x):
        # Medium edges
        modified_medium_map[ii+1,0] = medium_map[ii,0]
        modified_medium_map[ii+1,info.cells_y+1] = medium_map[ii,info.cells_y-1]

        # Center fluxes
        for jj in range(info.cells_y):
            modified_medium_map[ii+1,jj+1] = medium_map[ii,jj]

    # Medium edges
    for jj in range(info.cells_y):
        modified_medium_map[0,jj+1] = medium_map[0,jj]
        modified_medium_map[info.cells_x+1,jj+1] = medium_map[info.cells_x-1,jj]

    # Corner average (0, 0)
    modified_medium_map[0,0] = medium_map[1,0]
    # Corner average (-1, -1)
    modified_medium_map[info.cells_x+1,info.cells_y+1] \
                                    = medium_map[info.cells_x-1,info.cells_y-1]
    # Corner average (0, -1)
    modified_medium_map[0,info.cells_y+1] = medium_map[1,info.cells_y-1]
    # Corner average (-1, 0)
    modified_medium_map[info.cells_x+1,0] = medium_map[info.cells_x-1,0]

    return modified_medium_map


cdef void _nearby_angular_to_scalar(double[:,:,:,:]& angular, \
        double[:,:,:,:]& scalar, double[:]& angle_w, params info):

    # Initialize iterables
    cdef int ii, jj, nn, gg

    # Zero out scalar flux term
    scalar[:,:,:,:] = 0.0

    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    scalar[ii,jj,0,gg] += angle_w[nn] * angular[ii,jj,nn,gg]

################################################################################
# Nearby Problems Criticality functions
################################################################################

cdef void _nearby_fission_source(double[:,:,:]& flux, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& fission_source, \
        double[:,:,:,:]& residual, int[:,:]& medium_map, params info, \
        double keff):

    # Initialize iterables
    cdef int ii, jj, mat, nn, ig, og, nn_r
    cdef double one_group

    # Zero out previous power iteration
    fission_source[:,:,:,:] = 0.0

    # Iterate over all cells, angles, and groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                # loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                one_group = 0.0
                for ig in range(info.groups):
                    one_group += flux[ii,jj,ig] / keff * xs_fission[mat,og,ig]
                # for nn in range(info.angles * info.angles):
                #     # Add nearby residual
                #     nn_r = 0 if residual.shape[2] == 1 else nn
                #     fission_source[ii,jj,nn,og] += one_group + residual[ii,jj,nn_r,og]

                # Add nearby residual
                fission_source[ii,jj,0,og] += one_group + residual[ii,jj,0,og]


cdef void _nearby_critical_on_scatter(double[:,:,:]& residual, \
        double[:,:]& int_angular, double[:,:]& int_dx_angular, \
        double[:,:]& int_dy_angular, double[:,:]& xs_total, \
        int[:,:]& medium_map, double angle_x, double angle_y, \
        double angle_w, int gg0, int gg1, params info):

    # Initialize iterables
    cdef int ii, jj, mat

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            residual[ii,jj,gg0] += angle_w * ((angle_x * int_dx_angular[ii,jj]) \
                            + (angle_y * int_dy_angular[ii,jj]) \
                            + (int_angular[ii,jj] * xs_total[mat,gg1]))


cdef void _nearby_critical_off_scatter(double[:,:,:]& residual, \
    double[:,:,:]& scalar_flux, double[:,:,:]& xs_scatter, \
    double[:,:,:]& xs_fission, double[:,:,:]& fission_source, \
    double[:]& nearby_array, int[:,:]& medium_map, double[:]& delta_x, \
    double[:]& delta_y, double[:]& angle_w, params info):

    # Initialize iterables
    cdef int ii, jj, mat, og, ig

    # Initialize off-scattering term
    cdef float off_scatter
    cdef double fission_source_group

    # Initialize critical nearby terms
    cdef double keff_numerator = 0.0
    cdef double keff_denominator = 0.0
    cdef double nearby_rate = 0.0

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over groups
            for og in range(info.groups):
                fission_source_group = 0.0
                off_scatter = 0.0
                for ig in range(info.groups):
                    fission_source_group += scalar_flux[ii,jj,ig] * xs_fission[mat,og,ig]
                    off_scatter += scalar_flux[ii,jj,ig] * xs_scatter[mat,og,ig]

                # Create curve fit source with only one angle
                fission_source[ii,jj,og] = fission_source_group

                # Update off-scatter residual
                residual[ii,jj,og] -= (off_scatter)

                # Update nearby fission rate
                nearby_rate += fission_source_group / (delta_x[ii] * delta_y[jj])

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    keff_numerator += angle_w[nn] * fission_source_group
                    keff_denominator += angle_w[nn] * (residual[ii,jj,og])

    # Nearby Reaction Rate and Curve Fit Flux
    nearby_array[0] = nearby_rate
    nearby_array[1] = keff_numerator / keff_denominator


cdef void _nearby_critical_residual_source(double[:,:,:]& residual, \
    double[:,:,:]& fission_source, double curve_fit_keff, params info):

    # Initialize iterables
    cdef int ii, jj, gg

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            # Iterate over groups
            for gg in range(info.groups):
                residual[ii,jj,gg] -= fission_source[ii,jj,gg] / curve_fit_keff


cdef double _nearby_keffective(double[:,:,:]& flux, double rate, params info):
    cdef int ii, jj, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                keff += rate * flux[ii,jj,gg]
    return keff

################################################################################
# Hybrid Method Time Dependent Problems
################################################################################

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
        for jj in range(info_u.cells_y):
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

################################################################################
# Variable Hybrid Time Dependent Problems
################################################################################

cdef void _vhybrid_source_c(double[:,:,:]& flux_u, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& source_c, int[:,:]& medium_map,  int[:]& edges_gidx_c, \
        params info_u, params info_c):

    # Initialize iterables
    cdef int ii, jj, mat, gg, og, ig
    cdef double source

    # Zero out previous source
    source_c[:,:,:,:] = 0.0

    # Iterate over all spatial cells
    for ii in range(info_u.cells_x): #, nogil=True):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for gg in range(info_c.groups):
                source = 0.0
                for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                    for ig in range(info_u.groups):
                        source += flux_u[ii,jj,ig] * xs_scatter[mat,og,ig]
                source_c[ii,jj,0,gg] += source


cdef void _coarsen_flux(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        int[:]& edges_gidx_c, params info_c):

    # Initialize iterables
    cdef int ii, jj, og, ig
    cdef double tmp_flux

    # Zero out previous flux
    flux_c[:,:,:] = 0.0

    # Iterate over spatial cells and energy groups
    for ii in range(info_c.cells_x): #, nogil=True):
        for jj in range(info_c.cells_y):
            for og in range(info_c.groups):
                tmp_flux = 0.0
                for ig in range(edges_gidx_c[og], edges_gidx_c[og+1]):
                    tmp_flux += flux_u[ii,jj,ig]
                flux_c[ii,jj,og] += tmp_flux


cdef void _variable_off_scatter(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:]& xs_matrix, double[:,:]& off_scatter, \
        int group, double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, \
        int out_idx2, params info):

    # Initialize iterables
    cdef int gg, in_idx1, in_idx2, ii, jj, mat, og, ig
    cdef double prod_tmp, delta_coarse

    # Zero out previous values
    off_scatter[:,:] = 0.0

    # Iterate over collided groups
    for gg in range(info.groups): #, nogil=True):

        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])

        if gg < group:
            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    mat = medium_map[ii,jj]
                    prod_tmp = 0.0
                    for og in range(out_idx1, out_idx2):
                        for ig in range(in_idx1, in_idx2):
                            prod_tmp += xs_matrix[mat, og, ig] * delta_coarse \
                                        * (edges_g[ig+1] - edges_g[ig]) * flux[ii,jj,gg]
                    off_scatter[ii,jj] += prod_tmp

        elif gg > group:
            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    mat = medium_map[ii,jj]
                    prod_tmp = 0.0
                    for og in range(out_idx1, out_idx2):
                        for ig in range(in_idx1, in_idx2):
                            prod_tmp += xs_matrix[mat, og, ig] * delta_coarse \
                                        * (edges_g[ig+1] - edges_g[ig]) * flux_old[ii,jj,gg]
                    off_scatter[ii,jj] += prod_tmp


cdef void _vhybrid_source_total(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        double[:,:,:]& xs_matrix_u, double[:,:,:,:]& source, int[:,:]& medium_map, \
        double[:]& edges_g, int[:]& edges_gidx_c, params info_u, params info_c):

    # Initialize iterables
    cdef int ii, jj, mat, nn, ig, og, idx1, idx2
    cdef double one_group, delta_coarse

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x): #, nogil=True):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]

            # Combine fluxes
            for og in range(info_c.groups):
                idx1 = edges_gidx_c[og]
                idx2 = edges_gidx_c[og+1]
                delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])

                for ig in range(idx1, idx2):
                    flux_u[ii,jj,ig] = flux_u[ii,jj,ig] + flux_c[ii,jj,og] * delta_coarse \
                                    * (edges_g[ig+1] - edges_g[ig])

            # Add flux-xs product to source
            for og in range(info_u.groups):
                one_group = 0.0
                for ig in range(info_u.groups):
                    one_group += flux_u[ii,jj,ig] * xs_matrix_u[mat,og,ig]
                for nn in range(info_u.angles * info_u.angles):
                    source[ii,jj,nn,og] += one_group


################################################################################
# Anisotropic Scattering (spherical harmonic moments)
################################################################################
# Flux moments are (cells_x, cells_y, groups, moments) with
# phi_k = sum_n w_n Y_k(Omega_n) psi_n. Scatter moments are (materials, groups,
# groups, moments), where each harmonic k uses the Legendre moment of its
# degree l, without (2l + 1), which is in P_source[k, n] = (2l + 1) Y_k(Omega_n).
# For a single moment the isotropic function is called so the results are
# unchanged.

cdef void _xs_matrix_aniso(double[:,:,:,:]& mat1, double[:,:,:,:]& mat2, \
        double[:,:,:]& mat3, params info):
    # Fission is isotropic, so it only adds to the k = 0 moment
    _shared_xs_matrix_aniso(mat1, mat2, mat3, info)


cdef void _group_moments(double[:]& group_moments, double[:,:,:,:]& flux, \
        double[:,:,:,:]& xs_matrix, int mat, int ii, int jj, int group, \
        params info) noexcept nogil:
    # Scatter moments into one group: sum over groups of sigma_l * phi_k
    cdef int kk, ig
    for kk in range(info.moments):
        group_moments[kk] = 0.0
        for ig in range(info.groups):
            group_moments[kk] += flux[ii,jj,ig,kk] * xs_matrix[mat,group,ig,kk]


cdef void _off_scatter_moments(double[:,:,:,:]& flux, double[:,:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:,:]& off_scatter, params info, int group):
    # Gauss-Seidel off-group scatter moments (cells_x x cells_y x moments)
    cdef int ii, jj, mat, og, kk
    off_scatter[:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for kk in range(info.moments):
                for og in range(0, group):
                    off_scatter[ii,jj,kk] += xs_matrix[mat,group,og,kk] \
                                             * flux[ii,jj,og,kk]
                for og in range(group + 1, info.groups):
                    off_scatter[ii,jj,kk] += xs_matrix[mat,group,og,kk] \
                                             * flux_old[ii,jj,og,kk]


cdef void _off_scatter_jacobi_moments(double[:,:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:,:,:]& off_scatter_all, params info, int group) noexcept nogil:
    # Jacobi off-group scatter moments, see _off_scatter_jacobi
    cdef int ii, jj, mat, og, kk
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for kk in range(info.moments):
                off_scatter_all[group,ii,jj,kk] = 0.0
                for og in range(info.groups):
                    if og != group:
                        off_scatter_all[group,ii,jj,kk] += \
                                xs_matrix[mat,group,og,kk] * flux_old[ii,jj,og,kk]


cdef void _source_total_aniso(double[:,:,:,:]& source, double[:,:,:,:]& flux, \
        double[:,:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, double[:,:]& P_source, params info, \
        int[:,:] q_index=None):
    # Create sum_k (2l + 1) Y_k (sigma_s + sigma_f)_l * phi_k + external
    cdef int ii, jj, nn, og, og_q, mat
    if info.moments == 1:
        _source_total(source, flux[:,:,:,0], xs_matrix[:,:,:,0], medium_map, \
                      external, info, q_index)
        return
    cdef int[:,:] q_cols = _source_columns(external.shape[2], q_index, info)
    group_moments = array_1d(info.moments)
    source[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                _group_moments(group_moments, flux, xs_matrix, mat, ii, jj, og, info)
                for nn in range(info.angles * info.angles):
                    source[ii,jj,nn,og] = _legendre_sum(group_moments, P_source, \
                                nn, info) + external[ii,jj,q_cols[nn,og],og_q]


cdef void _angular_to_moments(double[:,:,:,:]& angular_flux, \
        double[:,:,:,:]& flux, double[:,:]& P_weights, params info):
    # phi_k = sum_n w_n Y_k(Omega_n) psi_n
    cdef int ii, jj, nn, gg, kk
    if info.moments == 1:
        _angular_to_scalar(angular_flux, flux[:,:,:,0], P_weights[0], info)
        return
    flux[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    for kk in range(info.moments):
                        flux[ii,jj,gg,kk] += angular_flux[ii,jj,nn,gg] \
                                             * P_weights[kk,nn]


cdef void _angular_edge_to_moments(double[:,:,:,:]& psi_x, \
        double[:,:,:,:]& psi_y, double[:,:,:,:]& flux, double[:,:]& P_weights, \
        params info):
    # Cell center moments from angular flux at cell edges
    cdef int ii, jj, nn, gg, kk
    cdef double psi
    if info.moments == 1:
        _angular_edge_to_scalar(psi_x, psi_y, flux[:,:,:,0], P_weights[0], info)
        return
    flux[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    psi = 0.25 * (psi_x[ii,jj,nn,gg] + psi_x[ii+1,jj,nn,gg] \
                                + psi_y[ii,jj,nn,gg] + psi_y[ii,jj+1,nn,gg])
                    for kk in range(info.moments):
                        flux[ii,jj,gg,kk] += psi * P_weights[kk,nn]


cdef void _time_source_star_cn_aniso(double[:,:,:,:]& psi_x, \
        double[:,:,:,:]& psi_y, double[:,:,:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& external_prev, \
        double[:,:,:,:]& external, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:,:]& P_source, double constant, params info):
    # Anisotropic scattering of the previous time step, see _time_source_star_cn
    cdef int ii, jj, mat, nn, og, nn_q, og_q
    cdef double psi, dpsi_x, dpsi_y
    if info.moments == 1:
        _time_source_star_cn(psi_x, psi_y, phi[:,:,:,0], xs_total, \
                xs_scatter[:,:,:,0], velocity, q_star, external_prev, external, \
                medium_map, delta_x, delta_y, angle_x, angle_y, constant, info)
        return
    group_moments = array_1d(info.moments)
    q_star[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                og_q = 0 if external.shape[3] == 1 else og
                _group_moments(group_moments, phi, xs_scatter, mat, ii, jj, og, info)
                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    # Calculate angular flux center
                    psi = 0.25 * (psi_x[ii,jj,nn,og] + psi_x[ii+1,jj,nn,og] \
                               + psi_y[ii,jj,nn,og] + psi_y[ii,jj+1,nn,og])
                    # Calculate cell flux derivative
                    dpsi_x = (psi_x[ii+1,jj,nn,og] - psi_x[ii,jj,nn,og]) / delta_x[ii]
                    dpsi_y = (psi_y[ii,jj+1,nn,og] - psi_y[ii,jj,nn,og]) / delta_y[jj]
                    q_star[ii,jj,nn,og] += _legendre_sum(group_moments, P_source, \
                                                         nn, info)
                    q_star[ii,jj,nn,og] += external[ii,jj,nn_q,og_q] \
                            + external_prev[ii,jj,nn_q,og_q] \
                            - angle_x[nn] * dpsi_x - angle_y[nn] * dpsi_y + psi \
                            * (constant / (velocity[og] * info.dt) - xs_total[mat,og])


cdef void _time_right_side_aniso(double[:,:,:,:]& q_star, \
        double[:,:,:,:]& flux, double[:,:,:,:]& xs_scatter, \
        int[:,:]& medium_map, double[:,:]& P_source, params info):
    # Add the anisotropic (sigma_s + sigma_f) * phi source to q_star
    cdef int ii, jj, nn, og, mat
    if info.moments == 1:
        _time_right_side(q_star, flux[:,:,:,0], xs_scatter[:,:,:,0], \
                         medium_map, info)
        return
    group_moments = array_1d(info.moments)
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                _group_moments(group_moments, flux, xs_scatter, mat, ii, jj, og, info)
                for nn in range(info.angles * info.angles):
                    q_star[ii,jj,nn,og] += _legendre_sum(group_moments, P_source, \
                                                         nn, info)


cdef void _normalize_moments(double[:,:,:,:]& flux, params info):
    # Normalize all moments by the L2 norm of the scalar flux
    cdef int ii, jj, gg, kk
    cdef double norm = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for gg in range(info.groups):
                norm += flux[ii,jj,gg,0] * flux[ii,jj,gg,0]
    norm = sqrt(norm)
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for gg in range(info.groups):
                for kk in range(info.moments):
                    flux[ii,jj,gg,kk] /= norm


cdef void _source_total_critical_aniso(double[:,:,:,:]& source, \
        double[:,:,:,:]& flux, double[:,:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:,:]& medium_map, double keff, \
        double[:,:]& P_source, params info):
    # Create sum_k (2l + 1) Y_k sigma_s,l phi_k + sigma_f * phi / keff
    cdef int ii, jj, nn, ig, og, mat
    cdef double fission
    group_moments = array_1d(info.moments)
    source[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                _group_moments(group_moments, flux, xs_scatter, mat, ii, jj, og, info)
                fission = 0.0
                for ig in range(info.groups):
                    fission += flux[ii,jj,ig,0] * xs_fission[mat,og,ig] / keff
                for nn in range(info.angles * info.angles):
                    source[ii,jj,nn,og] = _legendre_sum(group_moments, P_source, \
                                                        nn, info) + fission


cdef void _hybrid_source_collided_aniso(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:,:]& source_c, \
        int[:,:]& medium_map, int[:]& coarse_idx, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x J x N'^2 x G')
    cdef int ii, jj, mat, nn, og
    if info_u.moments == 1:
        _hybrid_source_collided(flux_u[:,:,:,0], xs_scatter[:,:,:,0], source_c, \
                                medium_map, coarse_idx, info_u)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info_u.groups):
                _group_moments(group_moments, flux_u, xs_scatter, mat, ii, jj, \
                               og, info_u)
                for nn in range(info_c.angles * info_c.angles):
                    source_c[ii,jj,nn,coarse_idx[og]] += _legendre_sum( \
                                            group_moments, P_source_c, nn, info_u)


cdef void _hybrid_source_total_aniso(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& flux_c, double[:,:,:,:]& xs_matrix, \
        double[:,:,:,:]& source, int[:,:]& medium_map, int[:]& coarse_idx, \
        double[:]& factor_u, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, jj, mat, nn, og, kk
    if info_u.moments == 1:
        _hybrid_source_total(flux_u[:,:,:,0], flux_c[:,:,:,0], \
                    xs_matrix[:,:,:,0], source, medium_map, coarse_idx, \
                    factor_u, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info_u.groups):
                for kk in range(info_u.moments):
                    flux_u[ii,jj,og,kk] = flux_u[ii,jj,og,kk] \
                            + flux_c[ii,jj,coarse_idx[og],kk] * factor_u[og]
            for og in range(info_u.groups):
                _group_moments(group_moments, flux_u, xs_matrix, mat, ii, jj, \
                               og, info_u)
                for nn in range(info_u.angles * info_u.angles):
                    source[ii,jj,nn,og] += _legendre_sum(group_moments, \
                                                         P_source_u, nn, info_u)


cdef void _vhybrid_source_c_aniso(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:,:]& source_c, \
        int[:,:]& medium_map, int[:]& edges_gidx_c, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x J x N'^2 x G')
    cdef int ii, jj, mat, nn, gg, og
    if info_u.moments == 1:
        _vhybrid_source_c(flux_u[:,:,:,0], xs_scatter[:,:,:,0], source_c, \
                          medium_map, edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for gg in range(info_c.groups):
                for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                    _group_moments(group_moments, flux_u, xs_scatter, mat, ii, \
                                   jj, og, info_u)
                    for nn in range(info_c.angles * info_c.angles):
                        source_c[ii,jj,nn,gg] += _legendre_sum(group_moments, \
                                                    P_source_c, nn, info_u)


cdef void _coarsen_moments(double[:,:,:,:]& flux_u, double[:,:,:,:]& flux_c, \
        int[:]& edges_gidx_c, params info_c):
    # Sum the fine group moments in each coarse group
    cdef int ii, jj, og, ig, kk
    flux_c[:,:,:,:] = 0.0
    for ii in range(info_c.cells_x):
        for jj in range(info_c.cells_y):
            for og in range(info_c.groups):
                for ig in range(edges_gidx_c[og], edges_gidx_c[og+1]):
                    for kk in range(info_c.moments):
                        flux_c[ii,jj,og,kk] += flux_u[ii,jj,ig,kk]


cdef void _variable_off_scatter_aniso(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, int[:,:]& medium_map, \
        double[:,:,:,:]& xs_matrix, double[:,:,:]& off_scatter, int group, \
        double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, int out_idx2, \
        params info):
    # Collapsed off-group scatter moments, see _variable_off_scatter
    cdef int gg, in_idx1, in_idx2, ii, jj, mat, og, ig, kk
    cdef double prod_tmp, delta_coarse
    off_scatter[:,:,:] = 0.0
    for gg in range(info.groups):
        if gg == group:
            continue
        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                mat = medium_map[ii,jj]
                for kk in range(info.moments):
                    prod_tmp = 0.0
                    for og in range(out_idx1, out_idx2):
                        for ig in range(in_idx1, in_idx2):
                            prod_tmp += xs_matrix[mat,og,ig,kk] * delta_coarse \
                                        * (edges_g[ig+1] - edges_g[ig])
                    if gg < group:
                        off_scatter[ii,jj,kk] += prod_tmp * flux[ii,jj,gg,kk]
                    else:
                        off_scatter[ii,jj,kk] += prod_tmp * flux_old[ii,jj,gg,kk]


cdef void _vhybrid_source_total_aniso(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& flux_c, double[:,:,:,:]& xs_matrix_u, \
        double[:,:,:,:]& source, int[:,:]& medium_map, double[:]& edges_g, \
        int[:]& edges_gidx_c, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, jj, mat, nn, ig, og, kk, idx1, idx2
    cdef double delta_coarse
    if info_u.moments == 1:
        _vhybrid_source_total(flux_u[:,:,:,0], flux_c[:,:,:,0], \
                xs_matrix_u[:,:,:,0], source, medium_map, edges_g, \
                edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info_c.groups):
                idx1 = edges_gidx_c[og]
                idx2 = edges_gidx_c[og+1]
                delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])
                for ig in range(idx1, idx2):
                    for kk in range(info_u.moments):
                        flux_u[ii,jj,ig,kk] = flux_u[ii,jj,ig,kk] \
                                + flux_c[ii,jj,og,kk] * delta_coarse \
                                * (edges_g[ig+1] - edges_g[ig])
            for og in range(info_u.groups):
                _group_moments(group_moments, flux_u, xs_matrix_u, mat, ii, jj, \
                               og, info_u)
                for nn in range(info_u.angles * info_u.angles):
                    source[ii,jj,nn,og] += _legendre_sum(group_moments, \
                                                         P_source_u, nn, info_u)
