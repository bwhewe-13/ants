################################################################################
#                             ___    _   _____________
#                            /   |  / | / /_  __/ ___/
#                           / /| | /  |/ / / /  \__ \
#                          / ___ |/ /|  / / /  ___/ /
#                         /_/  |_/_/ |_/ /_/  /____/
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


from libc.math cimport pow, sqrt

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
from ants.cytools_shared cimport group_convergence as _shared_group_convergence
from ants.cytools_shared cimport int_array_1d as _shared_int_array_1d
from ants.parameters cimport params


################################################################################
# Memoryview functions -- delegates to cytools_shared
################################################################################
cdef double[:] array_1d(int dim1):
    return _shared_array_1d(dim1)


cdef int[:] int_array_1d(int dim1):
    return _shared_int_array_1d(dim1)


cdef double[:,:] array_2d(int dim1, int dim2):
    return _shared_array_2d(dim1, dim2)


cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3):
    return _shared_array_3d(dim1, dim2, dim3)


cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4):
    return _shared_array_4d(dim1, dim2, dim3, dim4)

################################################################################
# Convergence functions
################################################################################
cdef double group_convergence(double[:,:]& arr1, double[:,:]& arr2, params info):
    cdef double[:,:] _arr1 = arr1
    cdef double[:,:] _arr2 = arr2
    return _shared_group_convergence(_arr1, _arr2, info)


cdef double angle_convergence(double[:]& arr1, double[:]& arr2, params info):
    cdef double[:] _arr1 = arr1
    cdef double[:] _arr2 = arr2
    return _shared_angle_convergence(_arr1, _arr2, info)

################################################################################
# Material Interface functions
################################################################################

cdef int[:] _material_index(int[:] medium_map, params info):
    # Initialize iterable
    cdef int ii
    # Initialize placeholder
    cdef int index = 1
    # Create splits of type int32
    cdef int[:] splits = medium_map[:].copy()
    # Add first edge
    splits[0] = 0
    # Iterate over medium map
    for ii in range(1, info.cells_x):
        if (medium_map[ii] != medium_map[ii-1]):
            splits[index] = ii
            index += 1
    # Add final edge
    splits[index] = info.cells_x
    # Return only necessary values
    return splits[:index+1]

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


cdef void _dmd_subtraction(double[:,:,:]& y_minus, double[:,:,:]& y_plus, \
        double[:,:]& flux, double[:,:]& flux_old, int kk, params info):
    # Initialize iterables
    cdef int ii, gg
    for ii in range(info.cells_x):
        for gg in range(info.groups):
            if (kk < info.dmd_snapshots - 1):
                y_minus[ii,gg,kk] = (flux[ii,gg] - flux_old[ii,gg])

            if (kk > 0):
                y_plus[ii,gg,kk-1] = (flux[ii,gg] - flux_old[ii,gg])


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


cdef void _off_scatter_jacobi(double[:,:]& flux_old, int[:]& medium_map, \
        double[:,:,:]& xs_matrix, double[:,:]& off_scatter_all, \
        params info, int group) noexcept nogil:
    # Jacobi variant: uses flux_old for ALL off-diagonal groups so that each
    # group's scattering source is independent of the sweep order.  This
    # removes the sequential Gauss-Seidel data dependency and allows groups
    # to be swept in parallel.  Both methods converge to the same fixed point.
    cdef int ii, mat, og

    for ii in range(info.cells_x):
        off_scatter_all[group, ii] = 0.0
        mat = medium_map[ii]
        for og in range(info.groups):
            if og != group:
                off_scatter_all[group, ii] += xs_matrix[mat, group, og] * flux_old[ii, og]


cdef void _source_total(double[:,:,:]& source, double[:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:]& medium_map, \
        double[:,:,:]& external, params info, int[:,:] q_index=None):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, nn, ig, og, og_q, mat
    cdef double one_group
    # External source column of each angle and group
    cdef int[:,:] q_cols = _source_columns(external.shape[1], q_index, info)
    # Zero out previous values
    source[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        for og in range(info.groups):
            og_q = 0 if external.shape[2] == 1 else og
            one_group = 0.0
            for ig in range(info.groups):
                one_group += flux[ii,ig] * xs_matrix[mat,og,ig]

            for nn in range(info.angles):
                source[ii,nn,og] += one_group
                source[ii,nn,og] += external[ii,q_cols[nn,og],og_q]


cdef int[:,:] _source_columns(int columns, int[:,:] q_index, params info):
    # External source column for each (angle, group), see cytools_shared
    return _shared_source_columns(columns, q_index, info.angles, info.groups)


cdef void _angular_to_scalar(double[:,:,:]& angular_flux, \
        double[:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x + info.flux_at_edges):
        for nn in range(info.angles):
            for gg in range(info.groups):
                scalar_flux[ii,gg] += angular_flux[ii,nn,gg] * angle_w[nn]


cdef void _angular_edge_to_scalar(double[:,:,:]& angular_flux, \
        double[:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for nn in range(info.angles):
            for gg in range(info.groups):
                scalar_flux[ii,gg] += 0.5 * angle_w[nn] * (angular_flux[ii,nn,gg] \
                                        + angular_flux[ii+1,nn,gg])


################################################################################
# Time Dependent functions
################################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info):
    cdef double[:,:] _xs_total = xs_total
    cdef double[:] _velocity = velocity
    _shared_total_velocity(_xs_total, _velocity, constant, info)


cdef void _time_source_star_bdf1(double[:,:,:]& flux, double[:,:,:]& q_star, \
        double[:,:,:]& external, double[:]& velocity, params info):
    # Combining the source (I x N x G) with the angular flux (I x N x G)
    # Initialize iterables
    cdef int ii, nn, gg, nn_q, gg_q
    # Zero out previous values
    q_star[:,:,:] = 0.0
    for gg in range(info.groups):
        gg_q = 0 if external.shape[2] == 1 else gg
        for nn in range(info.angles):
            nn_q = 0 if external.shape[1] == 1 else nn
            for ii in range(info.cells_x):
                # loc = gg + info.groups * (nn + ii * info.angles)
                q_star[ii,nn,gg] = external[ii,nn_q,gg_q] + flux[ii,nn,gg] \
                                    * 1 / (velocity[gg] * info.dt)


cdef void _time_source_star_cn(double[:,:,:]& psi_edges, double[:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:]& q_star, double[:,:,:]& external_prev, \
        double[:,:,:]& external, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double constant, params info):
    # Combining the external (I x N x G) with the angular flux (I x N x G)
    # external_prev is time step \ell, external is time step \ell + 1
    # Initialize iterables
    cdef int ii, mat, nn, og, ig, nn_q, og_q
    # Initialize angular flux center estimates
    cdef double psi, dpsi, one_group
    # Zero out previous values
    q_star[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            og_q = 0 if external.shape[2] == 1 else og
            one_group = 0.0
            for ig in range(info.groups):
                one_group += phi[ii,ig] * xs_scatter[mat,og,ig]
            for nn in range(info.angles):
                nn_q = 0 if external.shape[1] == 1 else nn
                # Calculate angular flux center
                psi = 0.5 * (psi_edges[ii,nn,og] + psi_edges[ii+1,nn,og])
                # Calculate cell flux derivative
                dpsi = (psi_edges[ii+1,nn,og] - psi_edges[ii,nn,og]) / delta_x[ii]
                # loc = og + info.groups * (nn + ii * info.angles)
                # Add scalar flux density of previous time step
                q_star[ii,nn,og] += one_group
                q_star[ii,nn,og] += external[ii,nn_q,og_q] - angle_x[nn] * dpsi \
                                + psi * (constant / (velocity[og] * info.dt) \
                                - xs_total[mat,og]) + external_prev[ii,nn_q,og_q]


cdef void _time_source_star_bdf2(double[:,:,:]& flux_1, \
        double[:,:,:]& flux_2, double[:,:,:]& q_star, \
        double[:,:,:]& external, double[:]& velocity, params info):
    # Combining the source (I x N x G) with the angular flux (I x N x G)
    # flux_1 is time step \ell - 1, flux_2 is time step \ell - 2
    # Initialize iterables
    cdef int ii, nn, gg, nn_q, gg_q
    # Zero out previous values
    q_star[:,:,:] = 0.0
    for gg in range(info.groups):
        gg_q = 0 if external.shape[2] == 1 else gg
        for nn in range(info.angles):
            nn_q = 0 if external.shape[1] == 1 else nn
            for ii in range(info.cells_x):
                # loc = gg + info.groups * (nn + ii * info.angles)
                q_star[ii,nn,gg] = external[ii,nn_q,gg_q] \
                        + flux_1[ii,nn,gg] * 2 / (velocity[gg] * info.dt) \
                        - flux_2[ii,nn,gg] * 1 / (2 * velocity[gg] * info.dt)


cdef void _time_source_star_tr_bdf2(double[:,:,:]& flux_1, double[:,:,:]& flux_2, \
        double[:,:,:]& q_star, double[:,:,:]& external, double[:]& velocity, \
        double gamma, params info):
    # flux_1 is time step \ell (edges)
    # flux_2 is time step \ell + gamma (centers)

    # Initialize iterables
    cdef int ii, nn, gg, nn_q, gg_q
    # Zero out previous values
    q_star[:,:,:] = 0.0
    # Iterate over cells, angles, groups
    for gg in range(info.groups):
        gg_q = 0 if external.shape[2] == 1 else gg
        for nn in range(info.angles):
            nn_q = 0 if external.shape[1] == 1 else nn
            for ii in range(info.cells_x):
                # loc = gg + info.groups * (nn + ii * info.angles)
                q_star[ii,nn,gg] = external[ii,nn_q,gg_q] + flux_2[ii,nn,gg] \
                        * 1 / (gamma * (1 - gamma) * velocity[gg] * info.dt) \
                        - 0.5 * (flux_1[ii,nn,gg] + flux_1[ii+1,nn,gg]) \
                        * (1 - gamma) / (gamma * velocity[gg] * info.dt)


cdef void _time_right_side(double[:,:,:]& q_star, double[:,:]& flux, \
        double[:,:,:]& xs_scatter, int[:]& medium_map, params info):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Initialize iterables
    cdef int ii, nn, ig, og, mat
    cdef double one_group
    # Iterate over dimensions
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            # loc = og + info.groups * (nn + ii * info.angles)
            one_group = 0.0
            for ig in range(info.groups):
                one_group += flux[ii,ig] * xs_scatter[mat,og,ig]
            for nn in range(info.angles):
                q_star[ii,nn,og] += one_group

################################################################################
# Criticality functions
################################################################################

cdef double[:,:,:] _fission_matrix(object fission, object chi):
    return _shared_fission_matrix(fission, chi)


cdef void _normalize_flux(double[:,:]& flux, params info):
    cdef double[:,:] _flux = flux
    _shared_normalize_flux(_flux, info)


cdef void _fission_source(double[:,:]& flux, double[:,:,:]& xs_fission, \
        double[:,:,:]& source, int[:]& medium_map, params info, double keff):
    # Calculate the fission source (I x G) for the power iteration
    # (keff^{-1} * sigma_f * phi)
    # Initialize iterables
    cdef int ii, mat, ig, og#, loc
    # Zero out previous power source
    source[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            # loc = og + ii * info.groups
            for ig in range(info.groups):
                source[ii,0,og] += flux[ii,ig] * xs_fission[mat,og,ig]
            source[ii,0,og] /= keff


cdef double _update_keffective(double[:,:] flux_new, double[:,:] flux_old, \
        double[:,:,:] xs_fission, int[:] medium_map, params info, double keff):
    return _shared_update_keffective(flux_new, flux_old, xs_fission, medium_map, info, keff)


cdef void _source_total_critical(double[:,:,:]& source, double[:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:]& medium_map, double keff, params info):
    # Create (sigma_s + sigma_f) * phi + external function

    # Initialize iterables
    cdef int ii, ig, og, mat#, loc

    # Zero out previous values
    source[:,:,:] = 0.0

    # Iterate over all cells, groups
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            # loc = og + info.groups * ii
            for ig in range(info.groups):
                source[ii,0,og] += (flux[ii,ig] * xs_fission[mat,og,ig]) / keff \
                                 + (flux[ii,ig] * xs_scatter[mat,og,ig])


################################################################################
# Nearby Problems
################################################################################

cdef void _nearby_flux_to_scalar(double[:,:]& scalar_flux, \
        double[:]& angular_spatial, double angle_w, int gg, params info):
    # Initialize iterables
    cdef int ii
    # Iterate over all spatial cells
    for ii in range(info.cells_x):
        scalar_flux[ii,gg] += angular_spatial[ii] * angle_w


cdef void _nearby_off_scatter(double[:,:]& residual, \
    double[:,:]& scalar_flux, double[:,:,:]& xs_scatter, \
    double[:,:,:]& xs_fission, int[:]& medium_map, params info):

    # Initialize iterables
    cdef int ii, mat, og, ig

    # Initialize off-scattering term
    cdef double off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        # Iterate over groups
        for og in range(info.groups):
            off_scatter = 0.0
            for ig in range(info.groups):
                off_scatter += scalar_flux[ii,ig] \
                            * (xs_scatter[mat,og,ig] + xs_fission[mat,og,ig])
            residual[ii,og] -= (off_scatter)


cdef void _nearby_on_scatter(double[:,:]& residual, double[:]& int_angular, \
        double[:]& int_dx_angular, double[:,:]& xs_total, \
        double[:,:]& external, int[:]& medium_map, double[:]& delta_x, \
        double angle_x, double angle_w, int gg0, int gg1, params info):

    # Initialize iterables
    cdef int ii, mat

    # Allow for energy independent sources
    cdef int gg_q = 0 if external.shape[1] == 1 else gg1

    for ii in range(info.cells_x):
        mat = medium_map[ii]
        residual[ii,gg0] += angle_w * ((angle_x * int_dx_angular[ii]) \
                        + (int_angular[ii] * xs_total[mat,gg1]) \
                        - (external[ii,gg_q] * delta_x[ii]))


cdef void _nearby_fission_source(double[:,:]& flux, double[:,:,:]& xs_fission, \
        double[:,:,:]& source, double[:,:,:]& residual, int[:]& medium_map, \
        params info, double keff):
    # Initialize iterables
    cdef int ii, mat, nn, ig, og
    cdef double one_group
    # Zero out previous power iteration
    source[:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            # loc = og + info.groups * (nn + ii * info.angles)
            one_group = 0.0
            for ig in range(info.groups):
                one_group += flux[ii,ig] * xs_fission[mat,og,ig] / keff
            for nn in range(info.angles):
                # Add nearby residual
                source[ii,nn,og] += one_group + residual[ii,nn,og]


cdef double _nearby_keffective(double[:,:]& flux, double rate, params info):
    cdef int ii, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            keff += rate * flux[ii, gg]
    return keff

################################################################################
# Hybrid Method Time Dependent Problems
################################################################################

cdef void _hybrid_source_collided(double[:,:]& flux_u, double[:,:,:]& xs_scatter, \
        double[:,:,:]& source_c, int[:]& medium_map, int[:]& coarse_idx, \
        params info_u, params info_c):
    # Initialize iterables
    cdef int ii, mat, og, ig
    # Zero out previous source
    source_c[:,:,:] = 0.0
    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_u.groups):
            # loc = coarse_idx[og] + ii * info_c.groups
            for ig in range(info_u.groups):
                source_c[ii,0,coarse_idx[og]] += flux_u[ii,ig] * xs_scatter[mat,og,ig]


cdef void _hybrid_source_total(double[:,:]& flux_u, double[:,:]& flux_c, \
        double[:,:,:]& xs_matrix, double[:,:,:]& source, int[:]& medium_map, \
        int[:]& coarse_idx, double[:]& factor_u,  params info_u, params info_c):
    # Initialize iterables
    cdef int ii, mat, nn, ig, og
    cdef double one_group
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_u.groups):
            flux_u[ii,og] = flux_u[ii,og] + flux_c[ii,coarse_idx[og]] * factor_u[og]
        for og in range(info_u.groups):
            one_group = 0.0
            for ig in range(info_u.groups):
                one_group += flux_u[ii,ig] * xs_matrix[mat,og,ig]
            for nn in range(info_u.angles):
                source[ii,nn,og] += one_group

################################################################################
# Variable Hybrid Time Dependent Problems
################################################################################

cdef void _vhybrid_parameters(int[:] coarse_idx, double[:] factor, \
        double[:] edges_g, int[:] edges_gidx_c, int groups_c):

    cdef int ii, jj
    cdef int count = 0

    for ii in range(groups_c):
        for jj in range(edges_gidx_c[ii], edges_gidx_c[ii+1]):
            coarse_idx[jj] = count
            factor[jj] = (edges_g[jj + 1] - edges_g[jj]) \
                        / (edges_g[edges_gidx_c[ii + 1]] - edges_g[edges_gidx_c[ii]])
        count += 1


cdef void _vhybrid_velocity(double[:] star_coef_c, double[:] velocity_u, \
        int[:] edges_gidx_c, double constant, params info_c):

    cdef int ii, jj
    cdef double fine_group_vel

    for ii in range(info_c.groups):
        fine_group_vel = 0.0
        for jj in range(edges_gidx_c[ii], edges_gidx_c[ii+1]):
            fine_group_vel += velocity_u[jj]
        star_coef_c[ii] = constant * (edges_gidx_c[ii+1] - edges_gidx_c[ii]) \
                            / (fine_group_vel * info_c.dt)


cdef void _vhybrid_source_c(double[:,:]& flux_u, double[:,:,:]& xs_scatter, \
        double[:,:,:]& source_c, int[:]& medium_map,  int[:]& edges_gidx_c, \
        params info_u, params info_c):

    # Initialize iterables
    cdef int ii, mat, gg, og, ig
    cdef double source

    # Zero out previous source
    source_c[:,:,:] = 0.0

    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for gg in range(info_c.groups):
            source = 0.0
            for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                for ig in range(info_u.groups):
                    source += flux_u[ii,ig] * xs_scatter[mat,og,ig]
            source_c[ii,0,gg] = source


cdef void _coarsen_flux(double[:,:]& flux_u, double[:,:]& flux_c, \
        int[:]& edges_gidx_c, params info_c):

    # Initialize iterables
    cdef int ii, og, ig
    cdef double tmp_flux

    # Zero out previous flux
    flux_c[:,:] = 0.0

    # Iterate over spatial cells and energy groups
    for ii in range(info_c.cells_x):
        for og in range(info_c.groups):
            tmp_flux = 0.0
            for ig in range(edges_gidx_c[og], edges_gidx_c[og+1]):
                tmp_flux += flux_u[ii, ig]
            flux_c[ii,og] = tmp_flux


cdef void _variable_cross_sections(double[:]& xs_total_c, double[:,:]& xs_total_u, \
        double star_coef_c, double[:]& xs_scatter_c, double[:,:,:]& xs_scatter_u, \
        double[:]& edges_g, int idx1, int idx2, params info_c):

    # Initialize iterables
    cdef int mat, ii, jj
    cdef double delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])

    # Zero out cross sections
    xs_total_c[:] = 0.0
    xs_scatter_c[:] = 0.0

    # Iterate over every material and uncollided energy group
    for mat in range(info_c.materials):
        for ii in range(idx1, idx2):
            xs_total_c[mat] += xs_total_u[mat,ii] * (edges_g[ii+1] - edges_g[ii])

            # Inner iteration for scatter cross section
            for jj in range(idx1, idx2):
                xs_scatter_c[mat] += xs_scatter_u[mat,ii,jj] * (edges_g[jj+1] - edges_g[jj])

        xs_total_c[mat] = xs_total_c[mat] * delta_coarse + star_coef_c
        xs_scatter_c[mat] *= delta_coarse


cdef void _variable_off_scatter(double[:,:]& flux, double[:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& off_scatter, \
        int group, double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, \
        int out_idx2, params info):

    # Initialize iterables
    cdef int gg, in_idx1, in_idx2, ii, mat, og, ig
    cdef double prod_tmp, delta_coarse

    # Zero out previous values
    off_scatter[:] = 0.0

    # Iterate over collided groups
    for gg in range(info.groups):

        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])

        if gg < group:
            for ii in range(info.cells_x):
                mat = medium_map[ii]
                prod_tmp = 0.0
                for og in range(out_idx1, out_idx2):
                    for ig in range(in_idx1, in_idx2):
                        prod_tmp += xs_matrix[mat, og, ig] * delta_coarse \
                                     * (edges_g[ig+1] - edges_g[ig]) * flux[ii,gg]
                off_scatter[ii] += prod_tmp

        elif gg > group:
            for ii in range(info.cells_x):
                mat = medium_map[ii]
                prod_tmp = 0.0
                for og in range(out_idx1, out_idx2):
                    for ig in range(in_idx1, in_idx2):
                        prod_tmp += xs_matrix[mat, og, ig] * delta_coarse \
                                    * (edges_g[ig+1] - edges_g[ig]) * flux_old[ii,gg]
                off_scatter[ii] += prod_tmp


cdef void _vhybrid_source_total(double[:,:]& flux_u, double[:,:]& flux_c, \
        double[:,:,:]& xs_matrix_u, double[:,:,:]& source, int[:]& medium_map, \
        double[:]& edges_g, int[:]& edges_gidx_c, params info_u, params info_c):

    # Initialize iterables
    cdef int ii, mat, nn, ig, og, idx1, idx2
    cdef double one_group, delta_coarse

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]

        for og in range(info_c.groups):
            idx1 = edges_gidx_c[og]
            idx2 = edges_gidx_c[og+1]
            delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])

            for ig in range(idx1, idx2):
                flux_u[ii,ig] = flux_u[ii,ig] + flux_c[ii,og] * delta_coarse \
                                * (edges_g[ig+1] - edges_g[ig])

        for og in range(info_u.groups):
            one_group = 0.0
            for ig in range(info_u.groups):
                one_group += flux_u[ii,ig] * xs_matrix_u[mat,og,ig]
            for nn in range(info_u.angles):
                source[ii,nn,og] += one_group


################################################################################
# Anisotropic Scattering (Legendre moments)
################################################################################
# Flux moments are (cells_x, groups, moments) with phi_l = sum_n w_n P_l psi_n.
# Scatter moments are (materials, groups, groups, moments) without (2l + 1),
# which is in P_source[l, n] = (2l + 1) * P_l(mu_n). For a single moment the
# isotropic function is called so the results are unchanged.

cdef void _xs_matrix_aniso(double[:,:,:,:]& mat1, double[:,:,:,:]& mat2, \
        double[:,:,:]& mat3, params info):
    # Fission is isotropic, so it only adds to the l = 0 moment
    cdef int ll, ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig,0] = (mat2[mat,og,ig,0] + mat3[mat,og,ig])
                for ll in range(1, info.moments):
                    mat1[mat,og,ig,ll] = mat2[mat,og,ig,ll]


cdef void _group_moments(double[:]& group_moments, double[:,:,:]& flux, \
        double[:,:,:,:]& xs_matrix, int mat, int cell, int group, \
        params info) noexcept nogil:
    # Scatter moments into one group: sum over groups of sigma_l * phi_l
    cdef int ll, ig
    for ll in range(info.moments):
        group_moments[ll] = 0.0
        for ig in range(info.groups):
            group_moments[ll] += flux[cell,ig,ll] * xs_matrix[mat,group,ig,ll]


cdef double _legendre_sum(double[:]& group_moments, double[:,:]& P_source, \
        int angle, params info) noexcept nogil:
    # Expand scatter moments at one ordinate: sum_l (2l + 1) P_l(mu_n) s_l
    cdef int ll
    cdef double one_angle = 0.0
    for ll in range(info.moments):
        one_angle += P_source[ll,angle] * group_moments[ll]
    return one_angle


cdef void _off_scatter_moments(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:]& off_scatter, params info, int group):
    # Gauss-Seidel off-group scatter moments (cells_x x moments)
    cdef int ii, mat, og, ll
    off_scatter[:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for ll in range(info.moments):
            for og in range(0, group):
                off_scatter[ii,ll] += xs_matrix[mat,group,og,ll] * flux[ii,og,ll]
            for og in range(group + 1, info.groups):
                off_scatter[ii,ll] += xs_matrix[mat,group,og,ll] * flux_old[ii,og,ll]


cdef void _off_scatter_jacobi_moments(double[:,:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:,:]& off_scatter_all, params info, int group) noexcept nogil:
    # Jacobi off-group scatter moments, see _off_scatter_jacobi
    cdef int ii, mat, og, ll
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for ll in range(info.moments):
            off_scatter_all[group,ii,ll] = 0.0
            for og in range(info.groups):
                if og != group:
                    off_scatter_all[group,ii,ll] += xs_matrix[mat,group,og,ll] \
                                                    * flux_old[ii,og,ll]


cdef void _source_total_aniso(double[:,:,:]& source, double[:,:,:]& flux, \
        double[:,:,:,:]& xs_matrix, int[:]& medium_map, \
        double[:,:,:]& external, double[:,:]& P_source, params info, \
        int[:,:] q_index=None):
    # Create sum_l (2l + 1) P_l (sigma_s + sigma_f)_l * phi_l + external
    cdef int ii, nn, og, og_q, mat
    if info.moments == 1:
        _source_total(source, flux[:,:,0], xs_matrix[:,:,:,0], medium_map, \
                      external, info, q_index)
        return
    cdef int[:,:] q_cols = _source_columns(external.shape[1], q_index, info)
    group_moments = array_1d(info.moments)
    source[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            og_q = 0 if external.shape[2] == 1 else og
            _group_moments(group_moments, flux, xs_matrix, mat, ii, og, info)
            for nn in range(info.angles):
                source[ii,nn,og] = _legendre_sum(group_moments, P_source, nn, info) \
                                    + external[ii,q_cols[nn,og],og_q]


cdef void _angular_to_moments(double[:,:,:]& angular_flux, \
        double[:,:,:]& flux, double[:,:]& P_weights, params info):
    # phi_l = sum_n w_n P_l(mu_n) psi_n
    cdef int ii, nn, gg, ll
    if info.moments == 1:
        _angular_to_scalar(angular_flux, flux[:,:,0], P_weights[0], info)
        return
    flux[:,:,:] = 0.0
    for ii in range(info.cells_x + info.flux_at_edges):
        for nn in range(info.angles):
            for gg in range(info.groups):
                for ll in range(info.moments):
                    flux[ii,gg,ll] += angular_flux[ii,nn,gg] * P_weights[ll,nn]


cdef void _angular_edge_to_moments(double[:,:,:]& angular_flux, \
        double[:,:,:]& flux, double[:,:]& P_weights, params info):
    # Cell center moments from angular flux at cell edges
    cdef int ii, nn, gg, ll
    cdef double psi
    if info.moments == 1:
        _angular_edge_to_scalar(angular_flux, flux[:,:,0], P_weights[0], info)
        return
    flux[:,:,:] = 0.0
    for ii in range(info.cells_x):
        for nn in range(info.angles):
            for gg in range(info.groups):
                psi = 0.5 * (angular_flux[ii,nn,gg] + angular_flux[ii+1,nn,gg])
                for ll in range(info.moments):
                    flux[ii,gg,ll] += psi * P_weights[ll,nn]


cdef void _time_source_star_cn_aniso(double[:,:,:]& psi_edges, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:]& q_star, \
        double[:,:,:]& external_prev, double[:,:,:]& external, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:,:]& P_source, double constant, params info):
    # Anisotropic scattering of the previous time step, see _time_source_star_cn
    cdef int ii, mat, nn, og, nn_q, og_q
    cdef double psi, dpsi
    if info.moments == 1:
        _time_source_star_cn(psi_edges, phi[:,:,0], xs_total, xs_scatter[:,:,:,0], \
                velocity, q_star, external_prev, external, medium_map, \
                delta_x, angle_x, constant, info)
        return
    group_moments = array_1d(info.moments)
    q_star[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            og_q = 0 if external.shape[2] == 1 else og
            _group_moments(group_moments, phi, xs_scatter, mat, ii, og, info)
            for nn in range(info.angles):
                nn_q = 0 if external.shape[1] == 1 else nn
                # Calculate angular flux center
                psi = 0.5 * (psi_edges[ii,nn,og] + psi_edges[ii+1,nn,og])
                # Calculate cell flux derivative
                dpsi = (psi_edges[ii+1,nn,og] - psi_edges[ii,nn,og]) / delta_x[ii]
                q_star[ii,nn,og] += _legendre_sum(group_moments, P_source, nn, info)
                q_star[ii,nn,og] += external[ii,nn_q,og_q] - angle_x[nn] * dpsi \
                                + psi * (constant / (velocity[og] * info.dt) \
                                - xs_total[mat,og]) + external_prev[ii,nn_q,og_q]


cdef void _time_right_side_aniso(double[:,:,:]& q_star, double[:,:,:]& flux, \
        double[:,:,:,:]& xs_scatter, int[:]& medium_map, \
        double[:,:]& P_source, params info):
    # Add the anisotropic (sigma_s + sigma_f) * phi source to q_star
    cdef int ii, nn, og, mat
    if info.moments == 1:
        _time_right_side(q_star, flux[:,:,0], xs_scatter[:,:,:,0], medium_map, info)
        return
    group_moments = array_1d(info.moments)
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            _group_moments(group_moments, flux, xs_scatter, mat, ii, og, info)
            for nn in range(info.angles):
                q_star[ii,nn,og] += _legendre_sum(group_moments, P_source, nn, info)


cdef void _normalize_moments(double[:,:,:]& flux, params info):
    # Normalize all moments by the L2 norm of the scalar flux
    cdef int ii, gg, ll
    cdef double norm = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            norm += pow(flux[ii,gg,0], 2)
    norm = sqrt(norm)
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            for ll in range(info.moments):
                flux[ii,gg,ll] /= norm


cdef void _source_total_critical_aniso(double[:,:,:]& source, \
        double[:,:,:]& flux, double[:,:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:]& medium_map, double keff, \
        double[:,:]& P_source, params info):
    # Create sum_l (2l + 1) P_l sigma_s,l phi_l + sigma_f * phi / keff
    cdef int ii, nn, ig, og, mat
    cdef double fission
    group_moments = array_1d(info.moments)
    source[:,:,:] = 0.0
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        for og in range(info.groups):
            _group_moments(group_moments, flux, xs_scatter, mat, ii, og, info)
            fission = 0.0
            for ig in range(info.groups):
                fission += flux[ii,ig,0] * xs_fission[mat,og,ig] / keff
            for nn in range(info.angles):
                source[ii,nn,og] = _legendre_sum(group_moments, P_source, nn, info) \
                                    + fission


cdef void _hybrid_source_collided_aniso(double[:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:]& source_c, \
        int[:]& medium_map, int[:]& coarse_idx, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x N' x G')
    cdef int ii, mat, nn, og
    if info_u.moments == 1:
        _hybrid_source_collided(flux_u[:,:,0], xs_scatter[:,:,:,0], source_c, \
                                medium_map, coarse_idx, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_u.groups):
            _group_moments(group_moments, flux_u, xs_scatter, mat, ii, og, info_u)
            for nn in range(info_c.angles):
                source_c[ii,nn,coarse_idx[og]] += _legendre_sum(group_moments, \
                                                    P_source_c, nn, info_u)


cdef void _hybrid_source_total_aniso(double[:,:,:]& flux_u, \
        double[:,:,:]& flux_c, double[:,:,:,:]& xs_matrix, \
        double[:,:,:]& source, int[:]& medium_map, int[:]& coarse_idx, \
        double[:]& factor_u, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, mat, nn, og, ll
    if info_u.moments == 1:
        _hybrid_source_total(flux_u[:,:,0], flux_c[:,:,0], xs_matrix[:,:,:,0], \
                    source, medium_map, coarse_idx, factor_u, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_u.groups):
            for ll in range(info_u.moments):
                flux_u[ii,og,ll] = flux_u[ii,og,ll] \
                                + flux_c[ii,coarse_idx[og],ll] * factor_u[og]
        for og in range(info_u.groups):
            _group_moments(group_moments, flux_u, xs_matrix, mat, ii, og, info_u)
            for nn in range(info_u.angles):
                source[ii,nn,og] += _legendre_sum(group_moments, P_source_u, nn, info_u)


cdef void _vhybrid_source_c_aniso(double[:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:]& source_c, \
        int[:]& medium_map, int[:]& edges_gidx_c, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x N' x G')
    cdef int ii, mat, nn, gg, og
    if info_u.moments == 1:
        _vhybrid_source_c(flux_u[:,:,0], xs_scatter[:,:,:,0], source_c, \
                          medium_map, edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for gg in range(info_c.groups):
            for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                _group_moments(group_moments, flux_u, xs_scatter, mat, ii, og, info_u)
                for nn in range(info_c.angles):
                    source_c[ii,nn,gg] += _legendre_sum(group_moments, \
                                                P_source_c, nn, info_u)


cdef void _coarsen_moments(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        int[:]& edges_gidx_c, params info_c):
    # Sum the fine group moments in each coarse group
    cdef int ii, og, ig, ll
    flux_c[:,:,:] = 0.0
    for ii in range(info_c.cells_x):
        for og in range(info_c.groups):
            for ig in range(edges_gidx_c[og], edges_gidx_c[og+1]):
                for ll in range(info_c.moments):
                    flux_c[ii,og,ll] += flux_u[ii,ig,ll]


cdef void _variable_cross_sections_aniso(double[:]& xs_total_c, \
        double[:,:]& xs_total_u, double star_coef_c, double[:,:]& xs_scatter_c, \
        double[:,:,:,:]& xs_scatter_u, double[:]& edges_g, int idx1, int idx2, \
        params info_c):
    # Collapse the within-group scatter moments, see _variable_cross_sections
    cdef int mat, ii, jj, ll
    cdef double delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])
    xs_total_c[:] = 0.0
    xs_scatter_c[:,:] = 0.0
    for mat in range(info_c.materials):
        for ii in range(idx1, idx2):
            xs_total_c[mat] += xs_total_u[mat,ii] * (edges_g[ii+1] - edges_g[ii])
            for jj in range(idx1, idx2):
                for ll in range(info_c.moments):
                    xs_scatter_c[mat,ll] += xs_scatter_u[mat,ii,jj,ll] \
                                            * (edges_g[jj+1] - edges_g[jj])
        xs_total_c[mat] = xs_total_c[mat] * delta_coarse + star_coef_c
        for ll in range(info_c.moments):
            xs_scatter_c[mat,ll] *= delta_coarse


cdef void _variable_off_scatter_aniso(double[:,:,:]& flux, \
        double[:,:,:]& flux_old, int[:]& medium_map, \
        double[:,:,:,:]& xs_matrix, double[:,:]& off_scatter, int group, \
        double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, int out_idx2, \
        params info):
    # Collapsed off-group scatter moments, see _variable_off_scatter
    cdef int gg, in_idx1, in_idx2, ii, mat, og, ig, ll
    cdef double prod_tmp, delta_coarse
    off_scatter[:,:] = 0.0
    for gg in range(info.groups):
        if gg == group:
            continue
        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])
        for ii in range(info.cells_x):
            mat = medium_map[ii]
            for ll in range(info.moments):
                prod_tmp = 0.0
                for og in range(out_idx1, out_idx2):
                    for ig in range(in_idx1, in_idx2):
                        prod_tmp += xs_matrix[mat,og,ig,ll] * delta_coarse \
                                    * (edges_g[ig+1] - edges_g[ig])
                if gg < group:
                    off_scatter[ii,ll] += prod_tmp * flux[ii,gg,ll]
                else:
                    off_scatter[ii,ll] += prod_tmp * flux_old[ii,gg,ll]


cdef void _vhybrid_source_total_aniso(double[:,:,:]& flux_u, \
        double[:,:,:]& flux_c, double[:,:,:,:]& xs_matrix_u, \
        double[:,:,:]& source, int[:]& medium_map, double[:]& edges_g, \
        int[:]& edges_gidx_c, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, mat, nn, ig, og, ll, idx1, idx2
    cdef double delta_coarse
    if info_u.moments == 1:
        _vhybrid_source_total(flux_u[:,:,0], flux_c[:,:,0], xs_matrix_u[:,:,:,0], \
                source, medium_map, edges_g, edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        mat = medium_map[ii]
        for og in range(info_c.groups):
            idx1 = edges_gidx_c[og]
            idx2 = edges_gidx_c[og+1]
            delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])
            for ig in range(idx1, idx2):
                for ll in range(info_u.moments):
                    flux_u[ii,ig,ll] = flux_u[ii,ig,ll] + flux_c[ii,og,ll] \
                                    * delta_coarse * (edges_g[ig+1] - edges_g[ig])
        for og in range(info_u.groups):
            _group_moments(group_moments, flux_u, xs_matrix_u, mat, ii, og, info_u)
            for nn in range(info_u.angles):
                source[ii,nn,og] += _legendre_sum(group_moments, P_source_u, nn, info_u)
