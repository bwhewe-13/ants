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
from ants.cyants.x_sweeps cimport x_scalar_sweep, x_angular_sweep, x_time_sweep

# from libcpp cimport float
from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np
from tqdm import tqdm

def source_iteration(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double [:] point_source,
                    double[:] spatial_coef, double[:] angle_weight, \
                    int[:] params, bint angular=False):
    if angular == True:
        return mg_angular_flux(medium_map, xs_total, \
                xs_scatter, xs_fission, external_source, \
                point_source, spatial_coef, angle_weight, params)
    else:
        return mg_scalar_flux(medium_map, xs_total, \
                xs_scatter, xs_fission, external_source, \
                point_source, spatial_coef, angle_weight, params)


cdef double[:,:] mg_scalar_flux(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double[:] point_source, \
                    double[:] spatial_coef, double[:] angle_weight, \
                    int[:] params):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t cell, group
    cdef size_t ex_group_idx, ps_group_idx

    arr2d_1 = cvarray((cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] scalar_flux_old = arr2d_1
    scalar_flux_old[:,:] = 0

    arr2d_2 = cvarray((cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] scalar_flux = arr2d_2

    arr1d = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] one_group_flux_old = arr1d
    one_group_flux_old[:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        scalar_flux[:, :] = 0
        for group in range(groups):
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[5] == 1 else group
            one_group_flux_old[:] = scalar_flux_old[:,group]
            scalar_flux[:,group] = x_scalar_sweep(one_group_flux_old, \
                            medium_map, xs_total[:,group], \
                            xs_scatter[:,group,group], \
                            external_source, \
                            point_source[ps_group_idx::params[6]], \
                            spatial_coef, angle_weight, params, \
                            ex_group_idx)
        change = scalar_convergence(scalar_flux, scalar_flux_old)
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old[:,:] = scalar_flux[:,:]
    return np.asarray(scalar_flux)

 
cdef double[:,:,:] mg_angular_flux(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double[:] point_source, \
                    double[:] spatial_coef, double[:] angle_weight, \
                    int[:] params):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = spatial_coef.shape[0]
    cdef size_t cell, group, angle
    cdef size_t ex_group_idx, ps_group_idx

    arr3d_1 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux_old = arr3d_1
    angular_flux_old[:,:,:] = 0

    arr3d_2 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux = arr3d_2

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")    
    cdef double[:,:] one_group_flux_old = arr2d
    one_group_flux_old[:,:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angular_flux[:,:,:] = 0
        for group in range(groups):
            one_group_flux_old[:,:] = angular_flux_old[:,:,group]
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[5] == 1 else group            
            angular_flux[:,:,group] = x_angular_sweep(one_group_flux_old, \
                            medium_map, xs_total[:,group], \
                            xs_scatter[:,group,group], \
                            external_source, \
                            point_source[ps_group_idx::params[6]], \
                            spatial_coef, angle_weight, params, \
                            ex_group_idx)
        change = angular_convergence(angular_flux, angular_flux_old, angle_weight)
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        angular_flux_old[:,:,:] = angular_flux[:,:,:]
    return np.asarray(angular_flux)


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


def time_dependent(int[:] medium_map, double[:,:] xs_total, \
                   double[:,:,:] xs_matrix, \
                   double[:] external_source, double[:] point_source, \
                   double[:] spatial_coef, \
                   double[:] angle_weight, double[:] velocity, \
                   int[:] params, double time_step_size):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]

    arr1d = cvarray((groups,), itemsize=sizeof(double), format="d")
    cdef double[:] temporal_coef = arr1d
    _time_coef(temporal_coef, velocity, time_step_size)

    arr3d_1 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux = arr3d_1
    angular_flux[:,:,:] = 0

    arr3d_2 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux_last = arr3d_2
    angular_flux_last[:,:,:] = 0

    arr3d_3 = cvarray((params[8], cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] time_step_flux = arr3d_3
    time_step_flux[:,:,:] = 0

    cdef double time_const 
    if params[7] == 1:
        time_const = 0.5
    else:
        time_const = 0.75
    for step in tqdm(range(params[8]), desc="Time Steps"): 
        angular_flux = time_source_iteration(angular_flux_last, medium_map, \
                xs_total, xs_matrix, external_source, \
                point_source, spatial_coef, angle_weight, \
                temporal_coef, params, time_const)
        angular_to_scalar(time_step_flux, angular_flux, angle_weight, step)
        angular_flux_last[:,:,:] = angular_flux[:,:,:]
    return np.asarray(time_step_flux)


cdef void _time_coef(double[:]& temporal_coef, double[:]& velocity, \
                     double time_step_size):
    cdef size_t groups = temporal_coef.shape[0]
    for group in range(groups):
        temporal_coef[group] = 1 / (velocity[group] * time_step_size)


cdef double[:,:,:] time_source_iteration(double[:,:,:]& angular_flux_last, \
                    int[:]& medium_map, double[:,:]& xs_total, \
                    double[:,:,:]& xs_matrix, \
                    double[:]& external_source, \
                    double[:]& point_source, double[:]& spatial_coef, \
                    double[:]& angle_weight, double[:]& temporal_coef, \
                    int[:]& params, double time_const):
    
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, ps_group_idx

    arr3d = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux_next = arr3d
    angular_flux_next[:,:,:] = 0

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")    
    cdef double[:,:] one_group_flux = arr2d
    one_group_flux[:,:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angular_flux_next[:,:,:] = 0
        for group in range(groups):
            one_group_flux[:,:] = angular_flux_last[:,:,group]
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[5] == 1 else group
            angular_flux_next[:,:,group] = x_time_sweep(one_group_flux, \
                            medium_map, xs_total[:,group], \
                            xs_matrix[:,group,group], \
                            external_source, \
                            point_source[ps_group_idx::params[6]], \
                            spatial_coef, angle_weight, params, \
                            temporal_coef[group], \
                            time_const, ex_group_idx)
        change = angular_convergence(angular_flux_next, angular_flux_last, angle_weight)
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        angular_flux_last[:,:,:] = angular_flux_next[:,:,:]
    return angular_flux_next


cdef void angular_to_scalar(double[:,:,:]& time_step_flux, \
            double[:,:,:]& angular_flux, double[:]& angle_weight, \
            size_t time_step):
    cdef size_t cell, angle, group
    cdef size_t cells = angular_flux.shape[0]
    cdef size_t angles = angular_flux.shape[1]
    cdef size_t groups = angular_flux.shape[2]
    for group in range(groups):
        for angle in range(angles):
            for cell in range(cells):
                time_step_flux[time_step][cell][group] += angle_weight[angle] \
                                           * angular_flux[cell][angle][group]
