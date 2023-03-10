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

from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
# import numpy as np

cdef params2d _to_params2d(dict params_dict):
    cdef params2d params
    params.cells_x = params_dict["cells_x"]
    params.cells_y = params_dict["cells_y"]
    params.angles = params_dict["angles"]
    params.groups = params_dict["groups"]
    params.materials = params_dict["materials"]
    params.geometry = params_dict["geometry"]
    params.spatial = params_dict["spatial"]
    params.qdim = params_dict["qdim"]
    params.bc_x = params_dict["bc_x"]
    params.bcdim_x = params_dict["bcdim_x"]
    params.bc_y = params_dict["bc_y"]
    params.bcdim_y = params_dict["bcdim_y"]
    params.steps = params_dict["steps"]
    params.dt = params_dict["dt"]
    params.angular = params_dict["angular"]
    params.adjoint = params_dict["adjoint"]
    return params

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
        double[:,:,:] xs_scatter, double[:,:,:] xs_fission, params2d params):
    cdef size_t mat, ig, og
    for mat in range(params.materials):
        for ig in range(params.groups):
            for og in range(params.groups):
                xs_matrix[mat,ig,og] = xs_scatter[mat,ig,og] \
                                        + xs_fission[mat,ig,og]

cdef void combine_total_velocity(double[:,:]& xs_total_star, \
        double[:,:]& xs_total, double[:]& velocity, params2d params):
    cdef size_t mat, group
    for group in range(params.groups):
        for mat in range(params.materials):
            xs_total_star[mat,group] = xs_total[mat,group] \
                                    + 1 / (velocity[group] * params.dt)

cdef void combine_source_flux(double[:,:,:]& flux_last, double[:]& source_star, \
        double[:]& source, double[:]& velocity, params2d params):
    cdef size_t cell, angle, group, start
    cdef size_t stop = params.groups * params.angles
    for group in range(params.groups):
        for angle in range(params.angles):
            for cell in range(params.cells_x * params.cells_y):
                start = group + angle * params.groups
                source_star[start::stop][cell] = source[start::stop][cell] \
                        + flux_last[cell,angle,group] \
                        * 1 / (velocity[group] * params.dt)

cdef double[:] array_1d_ij(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y,), \
                    itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef double[:] array_1d_ijg(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y * params.groups,), \
                    itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef double[:] array_1d_ijng(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y * params.angles \
                  * params.groups,), itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef double[:,:] array_2d_ijg(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y, params.groups), \
                    itemsize=sizeof(double), format="d")
    cdef double[:,:] flux = dd1
    flux[:,:] = 0.0
    return flux

cdef double[:,:] array_2d_ijn(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y, params.angles), \
                    itemsize=sizeof(double), format="d")
    cdef double[:,:] flux = dd1
    flux[:,:] = 0.0
    return flux


cdef double[:,:,:] array_3d_ijng(params2d params):
    dd1 = cvarray((params.cells_x * params.cells_y, params.angles, \
                params.groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux = dd1
    flux[:,:,:] = 0.0
    return flux

cdef double[:,:,:] array_3d_mgg(params2d params):
    dd1 = cvarray((params.materials, params.groups, params.groups), \
                    itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux = dd1
    flux[:,:,:] = 0.0
    return flux

cdef double[:,:,:,:] array_4d_tijng(params2d params):
    dd1 = cvarray((params.steps, params.cells_x * params.cells_y, \
            params.angles, params.groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:,:] flux = dd1
    flux[:,:,:,:] = 0.0
    return flux

cdef double[:] update_y_edge(double[:]& boundary_y, double angle_y, params2d params):
    # This is for converting boundary condition of [2 x I] or [2] into
    # [I] for edge_y
    cdef size_t loc
    loc = 0 if angle_y > 0.0 else 1 # top or bottom
    dd1 = cvarray((params.cells_x,), itemsize=sizeof(double), format="d")
    cdef double[:] edge_y = dd1
    if params.bcdim_y == 0:
        edge_y[:] = boundary_y[loc]
    elif loc == 0:
        edge_y[:] = boundary_y[0:params.cells_x]
    else:
        edge_y[:] = boundary_y[params.cells_x:2*params.cells_x]
    return edge_y

# cdef double[:] group_flux(params2d params):
#     cdef int flux_size = params.cells_x * params.cells_y * params.groups
#     flux_size *= params.angles if params.angular == True else 1
#     dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
#     cdef double[:] flux = dd1
#     flux[:] = 0.0
#     return flux

cdef double group_convergence_scalar(double[:,:]& arr1, double[:,:]& arr2, \
        params2d params):
    cdef size_t cell, group
    cdef size_t cells  = params.cells_x * params.cells_y
    cdef double change = 0.0
    for group in range(params.groups):
        for cell in range(cells):
            change += pow((arr1[cell,group] - arr2[cell,group]) \
                            / arr1[cell,group] / (cells), 2)
    change = sqrt(change)
    return change

cdef double group_convergence_angular(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        double[:]& weight, params2d params):
    cdef size_t group, cell, angle
    cdef double change = 0.0
    cdef double flux_new, flux_old
    for group in range(params.groups):
        for cell in range(params.cells_x * params.cells_y):
            flux_new = 0.0
            flux_old = 0.0
            for angle in range(params.angles):
                flux_new += arr1[cell,angle,group] * weight[angle]
                flux_old += arr2[cell,angle,group] * weight[angle]
            change += pow((flux_new - flux_old) / flux_new \
                            / (params.cells_x * params.cells_y), 2)
    change = sqrt(change)
    return change

cdef double[:] angle_flux(params2d params, bint angular):
    cdef int flux_size
    if angular == True:
        flux_size = params.cells_x * params.cells_y * params.angles
    else:
        flux_size = params.cells_x * params.cells_y
    dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef void angle_angular_to_scalar(double[:,:]& angular, double[:]& scalar, \
        double[:]& weight, params2d params):
    cdef size_t cell, angle
    scalar[:] = 0.0
    for angle in range(params.angles):
        for cell in range(params.cells_x * params.cells_y):
            scalar[cell] += angular[cell,angle] * weight[angle]

cdef double angle_convergence_scalar(double[:]& arr1, double[:]& arr2, \
        params2d params):
    cdef size_t cell
    cdef double change = 0.0
    for cell in range(params.cells_x * params.cells_y):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] \
                        / (params.cells_x * params.cells_y), 2)
    change = sqrt(change)
    return change

cdef double angle_convergence_angular(double[:,:]& arr1, double[:,:]& arr2, \
        double[:]& weight, params2d params):
    cdef size_t cell, angle
    cdef double change = 0.0
    cdef double flux, flux_old
    for cell in range(params.cells_x * params.cells_y):
        flux = 0.0
        flux_old = 0.0
        for angle in range(params.angles):
            flux += arr1[cell,angle] * weight[angle]
            flux_old += arr2[cell,angle] * weight[angle]
        change += pow((flux - flux_old) / flux \
                        / (params.cells_x * params.cells_y), 2)
    change = sqrt(change)
    return change

cdef void off_scatter_scalar(double[:,:]& flux, double[:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
        params2d params, size_t group):
    cdef size_t cell, mat, angle, og
    source[:] = 0.0
    for cell in range(params.cells_x * params.cells_y):
        mat = medium_map[cell]
        for og in range(0, group):
            source[cell] += xs_matrix[mat,group,og] * flux[cell,og]
        for og in range(group+1, params.groups):
            source[cell] += xs_matrix[mat,group,og] * flux_old[cell,og]

cdef void off_scatter_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
        double[:]& weight, params2d params, size_t group):
    cdef size_t cell, mat, angle, og
    source[:] = 0.0
    for cell in range(params.cells_x * params.cells_y):
        mat = medium_map[cell]
        for angle in range(params.angles):
            for og in range(0, group):
                source[cell] += xs_matrix[mat,group,og] * weight[angle] \
                                * flux[cell,angle,og]
            for og in range(group+1, params.groups):
                source[cell] += xs_matrix[mat,group,og] * weight[angle] \
                                * flux_old[cell,angle,og]

cdef void fission_source(double[:]& power_source, double[:,:]& flux, \
        double[:,:,:]& xs_fission, int[:]& medium_map, params2d params, \
        double[:] keff):
    cdef size_t cell, mat, ig, og
    power_source[:] = 0.0
    for cell in range(params.cells_x * params.cells_y):
        mat = medium_map[cell]
        for ig in range(params.groups):
            for og in range(params.groups):
                power_source[ig::params.groups][cell] += (1 / keff[0]) \
                                * flux[cell,og] * xs_fission[mat,ig,og]

cdef void normalize_flux(double[:,:]& flux, params2d params):
    cdef size_t cell, group
    cdef double keff = 0.0
    for group in range(params.groups):
        for cell in range(params.cells_x * params.cells_y):
            keff += pow(flux[cell,group], 2)
    keff = sqrt(keff)
    for group in range(params.groups):
        for cell in range(params.cells_x * params.cells_y):
            flux[cell,group] /= keff

# cdef void normalize_flux(double[:,:,:]& flux, params2d params):
#     cdef size_t xx, yy, gg
#     cdef double keff = 0.0
#     for xx in range(params.cells_x):
#         for yy in range(params.cells_y):
#             for gg in range(params.groups):
#                 keff += pow(flux[xx][yy][gg], 2)
#     keff = sqrt(keff)
#     for xx in range(params.cells_x):
#         for yy in range(params.cells_y):
#             for gg in range(params.groups):
#                 flux[xx][yy][gg] /= keff

cdef double update_keffective(double[:,:]& flux, double[:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_fission, params2d params, \
        double keff_old):
    cdef size_t cell, mat, ig, og
    cdef double rate_new = 0.0
    cdef double rate_old = 0.0
    for cell in range(params.cells_x * params.cells_y):
        mat = medium_map[cell]
        for ig in range(params.groups):
            for og in range(params.groups):
                rate_new += flux[cell,og] * xs_fission[mat,ig,og]
                rate_old += flux_old[cell,og] * xs_fission[mat,ig,og]
    return (rate_new * keff_old) / rate_old

cdef void calculate_source_c(double[:,:]& flux_u, double[:,:,:]& xs_scatter_u, \
        double[:]& source_c, int[:]& medium_map, int[:]& index_c, \
        params2d params_u, params2d params_c):
    cdef size_t cell, mat, ig, og
    # Multiply flux by scatter
    flux = flux_u.copy()
    flux[:,:] = 0.0
    for cell in range(params_u.cells_x * params_u.cells_y):
        mat = medium_map[cell]
        for og in range(params_u.groups):
            for ig in range(params_u.groups):
                flux[cell,ig] += flux_u[cell,og] * xs_scatter_u[mat,ig,og]
    # Shrink to size G hat
    big_to_small(flux, source_c, index_c, params_u, params_c)

cdef void calculate_source_t(double[:,:]& flux_u, double[:,:]& flux_c, \
        double[:,:,:]& xs_scatter_u, double[:]& source_t, \
        int[:]& medium_map, int[:]& index_u, double[:]& factor_u, \
        params2d params_u, params2d params_c):
    cdef size_t cell, mat, angle, group, ig, og
    source_t[:] = 0.0
    # Resize collided flux to size (I x G)
    flux = small_to_big(flux_c, index_u, factor_u, params_u, params_c)
    for cell in range(params_u.cells_x * params_u.cells_y):
        mat = medium_map[cell]
        for og in range(params_u.groups):
            for ig in range(params_u.groups):
                source_t[ig::params_u.groups][cell] += xs_scatter_u[mat,ig,og] \
                                    * (flux_u[cell,og] + flux[cell,og])

cdef void calculate_source_star(double[:,:,:]& flux_last, double[:]& source_star, \
        double[:]& source_t, double[:]& source_u, double[:]& velocity, params2d params):
    cdef size_t cell, angle, group, start
    cdef size_t stop = params.groups * params.angles
    source_star[:] = 0.0
    for group in range(params.groups):
        for angle in range(params.angles):
            for cell in range(params.cells_x * params.cells_y):
                start = group + angle * params.groups
                source_star[start::stop][cell] = source_t[group::params.groups][cell] \
                        + source_u[start::stop][cell] \
                        + flux_last[cell,angle,group] \
                        * 1 / (velocity[group] * params.dt)
                        

cdef void big_to_small(double[:,:]& flux_u, double[:]& flux_c, \
        int[:]& index_c, params2d params_u, params2d params_c):
    cdef size_t cell, group
    flux_c[:] = 0.0
    for cell in range(params_u.cells_x * params_u.cells_y):
        for group in range(params_u.groups):
            flux_c[index_c[group]::params_c.groups][cell] += flux_u[cell,group]
            # flux_c[group::params_c.groups][cell] += flux_u[cell,group]

cdef double[:,:] small_to_big(double[:,:]& flux_c, int[:]& index_u, \
        double[:]& factor_u, params2d params_u, params2d params_c):
    cdef size_t cell, group_u, group_c
    flux_u = array_2d_ijg(params_u)
    for cell in range(params_c.cells_x * params_c.cells_y):
        for group_c in range(params_c.groups):
            for group_u in range(index_u[group_c], index_u[group_c+1]):
                flux_u[cell,group_u] = flux_c[cell,group_c] * factor_u[group_u]
    return flux_u[:,:]