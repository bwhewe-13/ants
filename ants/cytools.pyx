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

# distutils: language=c++
# cython: cdivision=True

from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np

cdef params1d _to_params1d(dict params_dict):
    cdef params1d params
    params.cells = params_dict["cells"]
    params.angles = params_dict["angles"]
    params.groups = params_dict["groups"]
    params.materials = params_dict["materials"]
    params.geometry = params_dict["geometry"] 
    params.spatial = params_dict["spatial"]
    params.qdim = params_dict["qdim"]
    params.bc = params_dict["bc"]
    params.bcdim = params_dict["bcdim"]
    params.angular = params_dict["angular"]
    params.adjoint = params_dict["adjoint"]
    # print(params_dict.keys())
    return params

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission):
    cdef size_t materials = xs_matrix.shape[0]
    cdef size_t groups = xs_matrix.shape[1]
    for mat in range(materials):
        for ing in range(groups):
            for outg in range(groups):
                xs_matrix[mat][ing][outg] = xs_scatter[mat][ing][outg] \
                                            + xs_fission[mat][ing][outg]

cdef double[:] group_flux(params1d params, bint angular):
    cdef int flux_size
    if angular == True:
        flux_size = params.cells * params.angles * params.groups
    else:
        flux_size = params.cells * params.groups
    dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef double group_convergence(double[:]& arr1, double[:]& arr2, \
                    double[:]& weight, params1d params, bint angular):
    if angular == True:
        return _group_convergence_angular(arr1, arr2, weight, params)
    else:
        return _group_convergence_scalar(arr1, arr2, params)


cdef double _group_convergence_scalar(double[:]& arr1, double[:]& arr2, \
                                params1d params):
    cdef size_t cell
    cdef double change = 0.0
    for cell in range(params.groups * params.cells):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / params.cells, 2)
    change = sqrt(change)
    return change

cdef double _group_convergence_angular(double[:]& arr1, double[:]& arr2, \
                                double[:]& weight, params1d params):
    cdef size_t group, cell, angle
    cdef double change = 0.0
    cdef double flux, flux_old
    for group in range(params.groups):
        for cell in range(params.cells):
            flux = 0
            flux_old = 0
            for angle in range(params.angles):
                flux += arr1[group::params.groups][angle::params.angles][cell] \
                            * weight[angle]
                flux_old += arr2[group::params.groups][angle::params.angles][cell] \
                            * weight[angle]
            change += pow((flux - flux_old) / flux / params.cells, 2)
    change = sqrt(change)
    return change

cdef double[:] angle_flux(params1d params, bint angular):
    cdef int flux_size
    if angular == True:
        flux_size = params.cells * params.angles
    else:
        flux_size = params.cells
    dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
    cdef double[:] flux = dd1
    flux[:] = 0.0
    return flux

cdef void angle_angular_to_scalar(double[:]& arr1, double[:]& arr2, \
                        double[:]& weight, params1d params, bint angular):
    cdef size_t cell, angle
    if angular == True:
        arr1[:] = 0.0
        for angle in range(params.angles):
            for cell in range(params.cells):
                arr1[cell] += weight[angle] * arr2[angle::params.angles][cell]
    else:
        arr1[:] = arr2[:]


cdef double angle_convergence(double[:]& arr1, double[:]& arr2, \
                        double[:]& weight, params1d params, bint angular):
    if angular == True:
        return _angle_convergence_angular(arr1, arr2, weight, params)
    else:
        return _angle_convergence_scalar(arr1, arr2, params)


cdef double _angle_convergence_angular(double[:]& arr1, double[:]& arr2, \
                                double[:]& weight, params1d params):
    cdef size_t cell, angle
    cdef double change = 0.0
    cdef double flux, flux_old
    for cell in range(params.cells):
        flux = 0.0
        flux_old = 0.0
        for angle in range(params.angles):
            flux += arr1[angle::params.angles][cell] * weight[angle]
            flux_old += arr2[angle::params.angles][cell] * weight[angle]
        change += pow((flux - flux_old) / flux / params.cells, 2)
    change = sqrt(change)
    return change

cdef double _angle_convergence_scalar(double[:]& arr1, double[:]& arr2, \
                                params1d params):
    cdef size_t cell
    cdef double change = 0.0
    for cell in range(params.cells):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / params.cells, 2)
    change = sqrt(change)
    return change


cdef void off_scatter_term(double[:]& flux, double[:]& flux_old, \
                    int[:]& medium_map, double[:,:,:]& xs_matrix, \
                    double[:]& source, double[:]& weight, \
                    params1d params, size_t group, bint angular):
    if angular == True:
        _off_scatter_angular(flux, flux_old, medium_map, xs_matrix, source, \
                            weight, params, group)
    else:
        _off_scatter_scalar(flux, flux_old, medium_map, xs_matrix, source, \
                            params, group)


cdef void _off_scatter_scalar(double[:]& flux, double[:]& flux_old, \
                            int[:]& medium_map, double[:,:,:]& xs_matrix, \
                            double[:]& source, params1d params, size_t group):
    cdef size_t cell, mat, angle, og
    source[:] = 0.0
    for cell in range(params.cells):
        mat = medium_map[cell]
        for og in range(0, group):
            source[cell] += xs_matrix[mat,group,og] * flux[og::params.groups][cell]
        for og in range(group+1, params.groups):
            source[cell] += xs_matrix[mat,group,og] * flux_old[og::params.groups][cell]

cdef void _off_scatter_angular(double[:]& flux, double[:]& flux_old, \
                            int[:]& medium_map, double[:,:,:]& xs_matrix, \
                            double[:]& source, double[:]& weight, \
                            params1d params, size_t group):
    cdef size_t cell, mat, angle, og
    source[:] = 0.0
    for cell in range(params.cells):
        mat = medium_map[cell]
        for angle in range(params.angles):
            for og in range(0, group):
                source[cell] += xs_matrix[mat,group,og] * weight[angle] \
                        * flux[og::params.groups][angle::params.angles][cell]
            for og in range(group+1, params.groups):
                source[cell] += xs_matrix[mat,group,og] * weight[angle] \
                        * flux_old[og::params.groups][angle::params.angles][cell]

