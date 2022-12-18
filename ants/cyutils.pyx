########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np

cdef void power_iteration_source(double[:] power_source, double[:,:] flux, \
                                 int[:] medium_map, double[:,:,:] xs_fission, 
                                 double keff):
    power_source[:] = 0
    cdef int cells = medium_map.shape[0]
    cdef int groups = flux.shape[1]
    cdef int material
    for cell in range(cells):
        material = medium_map[cell]
        for ingroup in range(groups):
            for outgroup in range(groups):
                power_source[ingroup::groups][cell] += flux[cell][outgroup] \
                                * xs_fission[material][ingroup][outgroup] / keff

cdef void mnp_power_iteration_source(double[:] power_source, double[:,:] flux, \
                                    int[:] medium_map, double[:,:,:] xs_fission, \
                                    int angles, double keff):
    power_source[:] = 0
    cdef int cells = medium_map.shape[0]
    cdef int groups = flux.shape[1]
    cdef int material, angle
    for cell in range(cells):
        material = medium_map[cell]
        for angle in range(angles):
            for ig in range(groups):
                for og in range(groups):
                    power_source[ig::groups][angle::angles][cell] += flux[cell][og] \
                                    * xs_fission[material][ig][og] / keff

cdef void add_manufactured_source(double[:] power_source, double[:] mnp_source):
    cdef int cells = power_source.shape[0]
    cdef int cell
    for cell in range(cells):
        power_source[cell] += mnp_source[cell]

cdef double multiply_manufactured_flux(double[:,:] flux, double keff):
    cdef int cells = flux.shape[0]
    cdef int groups = flux.shape[1]
    cdef double half_keff = 0.0
    for cell in range(cells):
        for group in range(groups):
            half_keff += keff * flux[cell][group]
    return half_keff

cdef void normalize_flux(double[:,:] flux):
    cdef double keff = 0.0
    cdef int cells = flux.shape[0]
    cdef int groups = flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            keff += pow(flux[cell][group], 2)
    keff = sqrt(keff)
    for group in range(groups):
        for cell in range(cells):
            flux[cell][group] /= keff
    # return keff

cdef double update_keffective(double[:,:] flux, double[:,:] flux_old, \
                            int[:] medium_map, double[:,:,:] xs_fission, \
                            double keff_old):
    cdef double fission_rate = 0.0
    cdef double fission_rate_old = 0.0
    cdef int cells = flux.shape[0]
    cdef int groups = flux.shape[1]
    cdef int cell, material, ig, og
    for cell in range(cells):
        material = medium_map[cell]
        for ig in range(groups):
            for og in range(groups):
                fission_rate += flux[cell][og] * xs_fission[material][ig][og]
                fission_rate_old += flux_old[cell][og] * xs_fission[material][ig][og] / keff_old
    # print(fission_rate, fission_rate_old)
    return fission_rate / fission_rate_old

# cdef double fission_rate(double[:,:] flux, int[:] medium_map, \
#                                 double[:,:,:] xs_fission):
#     cdef double fission_rate = 0.0
#     cdef int cells = flux.shape[0]
#     cdef int groups = flux.shape[1]
#     cdef int cell, material, ig, og
#     for cell in range(cells):
#         material = medium_map[cell]
#         for ig in range(groups):
#             for og in range(groups):
#                 fission_rate += flux[cell][og] * xs_fission[material][ig][og]
#     return fission_rate

cdef void divide_by_keff(double[:,:] flux, double keff):
    cdef int cells = flux.shape[0]
    cdef int groups = flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            flux[cell][group] /= keff

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission):
    cdef size_t materials = xs_matrix.shape[0]
    cdef size_t groups = xs_matrix.shape[1]
    for mat in range(materials):
        for ing in range(groups):
            for outg in range(groups):
                xs_matrix[mat][ing][outg] = xs_scatter[mat][ing][outg] \
                                            + xs_fission[mat][ing][outg]

cdef void off_scatter_source(double[:,:]& flux, double[:,:]& flux_old, \
                             int[:]& medium_map, double[:,:,:]& xs_matrix, \
                             double[:]& source, int group):
    cdef int groups = flux.shape[1]
    cdef int cells = medium_map.shape[0]
    source[:] = 0
    for cell in range(cells):
        material = medium_map[cell]
        for outgroup in range(0, group):
            source[cell] += xs_matrix[material, group, outgroup] \
                            * flux[cell, outgroup]
        for outgroup in range(group+1, groups):
            source[cell] += xs_matrix[material, group, outgroup] \
                            * flux_old[cell, outgroup]

cdef void off_scatter_source_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
                        int[:]& medium_map, double[:,:,:]& xs_matrix, \
                        double[:]& source, size_t group, double[:]& weight):
    cdef size_t groups = flux.shape[2]
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = weight.shape[0]
    source[:] = 0
    for cell in range(cells):
        material = medium_map[cell]
        for angle in range(angles):
            for outgroup in range(0, group):
                source[cell] += xs_matrix[material, group, outgroup] \
                            * flux[cell, angle, outgroup] * weight[angle]
            for outgroup in range(group+1, groups):
                source[cell] += xs_matrix[material, group, outgroup] \
                            * flux_old[cell, angle, outgroup] * weight[angle]

cdef double scalar_convergence(double [:,:]& arr1, double [:,:]& arr2):
    cdef size_t cells, groups
    cells = arr1.shape[0]
    groups = arr1.shape[1]
    cdef double change = 0.0
    for group in range(groups):
        for cell_x in range(cells):
            change += pow((arr1[cell_x][group] - arr2[cell_x][group]) \
                        / arr1[cell_x][group] / cells, 2)
    change = sqrt(change)
    return change

cdef double group_scalar_convergence(double [:]& arr1, double [:]& arr2):
    cdef int cells = arr1.shape[0]
    cdef double change = 0.0
    for cell in range(cells):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / cells, 2)
    change = sqrt(change)
    return change

cdef double angular_convergence(double [:,:,:]& arr1, double [:,:,:]& arr2, \
                                double[:]& weight):
    cdef size_t cells, angles, groups
    cells = arr1.shape[0]
    angles = arr1.shape[1]
    groups = arr1.shape[2]
    cdef double change = 0.0
    cdef double scalar_flux, scalar_flux_old
    for group in range(groups):
        for cell in range(cells):
            scalar_flux = 0
            scalar_flux_old = 0
            for angle in range(angles):
                scalar_flux += weight[angle] * arr1[cell][angle][group]
                scalar_flux_old += weight[angle] * arr2[cell][angle][group]
            change += pow((scalar_flux - scalar_flux_old) / \
                            scalar_flux / cells, 2)
    change = sqrt(change)
    return change

cdef double group_angular_convergence(double[:,:]& arr1, double [:,:]& arr2, \
                                      double [:]& weight):
    cdef size_t cells, angles
    cells = arr1.shape[0]
    angles = arr1.shape[1]
    cdef double change = 0.0
    cdef double scalar_flux, scalar_flux_old
    for cell in range(cells):
        scalar_flux = 0
        scalar_flux_old = 0
        for angle in range(angles):
            scalar_flux += weight[angle] * arr1[cell][angle]
            scalar_flux_old += weight[angle] * arr2[cell][angle]
        change += pow((scalar_flux - scalar_flux_old) / \
                        scalar_flux / cells, 2)
    change = sqrt(change)
    return change

cdef void angular_to_scalar(double[:,:]& scalar_flux, \
                        double[:,:,:]& angular_flux, double[:]& weight):
    cdef size_t cells = angular_flux.shape[0]
    cdef size_t angles = angular_flux.shape[1]
    cdef size_t groups = angular_flux.shape[2]
    for group in range(groups):
        for angle in range(angles):
            for cell in range(cells):
                scalar_flux[cell][group] += \
                        weight[angle] * angular_flux[cell][angle][group]

cdef void group_angular_to_scalar(double[:]& scalar_flux, \
                    double[:,:]& angular_flux, double[:]& angle_weight):
    cdef size_t cells, angles, cell, angle
    cells = angular_flux.shape[0]
    angles = angular_flux.shape[1]
    scalar_flux[:] = 0
    for angle in range(angles):
        for cell in range(cells):
            scalar_flux[cell] += angle_weight[angle] * angular_flux[cell][angle]

cdef void time_coef(double[:]& temporal_coef, double[:]& velocity, \
                     double time_step_size):
    cdef size_t groups = velocity.shape[0]
    for group in range(groups):
        temporal_coef[group] = 1 / (velocity[group] * time_step_size)
