########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Multigroup spatial sweeps for different spatial discretizations.
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

import ants.constants as constants
from x_sweeps cimport scalar_sweep, angular_sweep

# from libcpp cimport float
from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np

def source_iteration(int [:] medium_map, \
                    double [:, :] xs_total, \
                    double [:, :, :] xs_scatter, \
                    double [:, :, :] xs_fission, \
                    double [:, :, :] external_source, \
                    int [:] point_source_loc, \
                    double [:, :, :] point_source, \
                    double [:] spatial_coef, \
                    double [:] angle_weight,\
                    int spatial = 2, int boundary = 0, \
                    bint angular=False):
    if angular == True:
        return angular_flux(medium_map, xs_total, \
                xs_scatter, xs_fission, external_source, point_source_loc, \
                point_source, spatial_coef, angle_weight, spatial=spatial, \
                boundary=boundary)
    else:
        return scalar_flux(medium_map, xs_total, \
                xs_scatter, xs_fission, external_source, point_source_loc, \
                point_source, spatial_coef, angle_weight, spatial=spatial, \
                boundary=boundary)
    

cdef double[:,:] scalar_flux(int [:] medium_map, \
                            double [:,:] xs_total, \
                            double [:,:,:] xs_scatter, \
                            double [:,:,:] xs_fission, \
                            double [:,:,:] external_source, \
                            int [:] point_source_loc, \
                            double [:,:,:] point_source, \
                            double [:] spatial_coef, \
                            double [:] angle_weight,\
                            int spatial = 2, int boundary = 0):
    
    cdef size_t cells_x = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t cell, group

    arr2d_1 = cvarray((cells_x, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] scalar_flux_old = arr2d_1
    scalar_flux_old[:,:] = 0

    arr2d_2 = cvarray((cells_x, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] scalar_flux = arr2d_2

    arr1d = cvarray((cells_x,), itemsize=sizeof(double), format="d")    
    cdef double[:] one_group_flux_old = arr1d
    one_group_flux_old[:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        scalar_flux[:, :] = 0
        for group in range(groups):
            one_group_flux_old[:] = scalar_flux_old[:,group]
            scalar_flux[:,group] = scalar_sweep(one_group_flux_old, \
                            medium_map, xs_total[:,group], \
                            xs_scatter[:,group,group], \
                            external_source[:,:,group], \
                            point_source_loc, point_source[:,:,group], \
                            spatial_coef, angle_weight, spatial, boundary)
        change = scalar_convergence(scalar_flux, scalar_flux_old)
        # print("Out Count", count, "Change", change)
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old[:,:] = scalar_flux[:,:]
    return np.asarray(scalar_flux)


cdef double[:,:,:] angular_flux(int [:] medium_map, \
                            double [:,:] xs_total, \
                            double [:,:,:] xs_scatter, \
                            double [:,:,:] xs_fission, \
                            double [:,:,:] external_source, \
                            int [:] point_source_loc, \
                            double [:,:,:] point_source, \
                            double [:] spatial_coef, \
                            double [:] angle_weight,\
                            int spatial = 2, int boundary = 0):
    cdef size_t cells_x = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = spatial_coef.shape[0]
    cdef size_t cell, group, angle

    arr3d_1 = cvarray((cells_x, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux_old = arr3d_1
    angular_flux_old[:,:,:] = 0

    arr3d_2 = cvarray((cells_x, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux = arr3d_2

    arr2d = cvarray((cells_x, angles), itemsize=sizeof(double), format="d")    
    cdef double[:,:] one_group_flux_old = arr2d
    one_group_flux_old[:,:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angular_flux[:,:,:] = 0
        for group in range(groups):
            one_group_flux_old[:,:] = angular_flux_old[:,:,group]
            angular_flux[:,:,group] = angular_sweep(one_group_flux_old, \
                            medium_map, xs_total[:,group], \
                            xs_scatter[:,group,group], \
                            external_source[:,:,group], \
                            point_source_loc, point_source[:,:,group], \
                            spatial_coef, angle_weight, spatial, boundary)
            # print("In group loop",np.sum(angular_flux), np.sum(angular_flux_old))
        change = angular_convergence(angular_flux, angular_flux_old, angle_weight)
        # print("Out Count", count, "Change", change)
        # print()
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        angular_flux_old[:,:,:] = angular_flux[:,:,:]
    return np.asarray(angular_flux)


cdef double scalar_convergence(double [:,:]& arr1, double [:,:]& arr2):
    cdef size_t cells_x, groups
    cells_x = arr1.shape[0]
    groups = arr1.shape[1]
    cdef double change = 0.0
    for group in range(groups):
        for cell_x in range(cells_x):
            change += pow((arr1[cell_x][group] - arr2[cell_x][group]) \
                        / arr1[cell_x][group] / cells_x, 2)
    change = sqrt(change)
    return change


cdef double angular_convergence(double [:,:,:]& angular_flux, \
                                double [:,:,:]& angular_flux_old, \
                                double[:]& angle_weight):
    cdef size_t cells, angles, groups
    cells = angular_flux.shape[0]
    angles = angular_flux.shape[1]
    groups = angular_flux.shape[2]
    cdef double change = 0.0
    cdef double scalar_flux, scalar_flux_old
    for group in range(groups):
        for cell in range(cells):
            scalar_flux = 0
            scalar_flux_old = 0
            for angle in range(angles):
                scalar_flux += angle_weight[angle] * angular_flux[cell][angle][group]
                scalar_flux_old += angle_weight[angle] * angular_flux_old[cell][angle][group]
            change += pow((scalar_flux - scalar_flux_old) / \
                            scalar_flux / cells, 2)
    change = sqrt(change)
    return change

