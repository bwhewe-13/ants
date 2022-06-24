########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Mono-energetic spatial sweeps for different spatial discretizations.
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

import ants.constants as constants

# from libcpp cimport float
from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
# import numpy as np


cdef double[:] x_scalar_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
                            double[:]& xs_total, double[:]& xs_scatter, \
                            double[:]& external_source, \
                            double[:]& point_source, double[:]& spatial_coef, \
                            double[:]& angle_weight, int[:]& params, \
                            size_t ex_group_idx): 
    cdef int cells = medium_map.shape[0]
    cdef int angles = angle_weight.shape[0]
    cdef int cell, angle
    cdef size_t ex_angle_idx

    arr1d = cvarray((cells,), itemsize=sizeof(double), format="d")
    cdef double[:] scalar_flux = arr1d
    scalar_flux[:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        scalar_flux[:] = 0
        for angle in range(angles):
            ex_angle_idx = 0 if params[4] == 1 else angle
            if params[2] == 0:
                vacuum(scalar_flux, scalar_flux_old, medium_map, \
                       xs_total, xs_scatter, external_source, \
                       point_source[angle], spatial_coef[angle], \
                       angle_weight[angle], params, ex_group_idx, \
                       ex_angle_idx)
            elif params[2] == 1:
                reflected(scalar_flux, scalar_flux_old, medium_map, \
                          xs_total, xs_scatter, external_source, \
                          point_source[angle], spatial_coef[angle], \
                          angle_weight[angle], params, ex_group_idx, \
                          ex_angle_idx)
            else:
                print("You are wrong")
        change = scalar_convergence(scalar_flux, scalar_flux_old)
        # print("In Count", count, "Change", change)
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old[:] = scalar_flux[:]
    return scalar_flux[:]


cdef double[:,:] x_angular_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
                                double[:]& xs_total, double[:]& xs_scatter, \
                                double[:]& external_source, \
                                double[:]& point_source, double[:]& spatial_coef, \
                                double[:]& angle_weight, int[:]& params, \
                                size_t ex_group_idx): 
    cdef int cells = medium_map.shape[0]
    cdef int angles = angle_weight.shape[0]
    cdef int cell, angle
    cdef size_t ex_angle_idx

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")
    cdef double[:,:] angular_flux = arr2d
    angular_flux[:,:] = 0

    arr1d_1 = cvarray((angles,), itemsize=sizeof(double), format="d")
    cdef double[:] dummy_angle_weight = arr1d_1
    dummy_angle_weight[:] = 1

    arr1d_2 = cvarray((cells,), itemsize=sizeof(double), format="d")
    cdef double[:] scalar_flux = arr1d_2
    scalar_flux[:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angular_flux[:,:] = 0
        angular_to_scalar(scalar_flux, angular_flux_old, angle_weight)
        for angle in range(angles):
            ex_angle_idx = 0 if params[4] == 1 else angle
            if params[2] == 0:
                vacuum(angular_flux[:,angle], scalar_flux, medium_map, \
                       xs_total, xs_scatter, external_source, \
                       point_source[angle], spatial_coef[angle], \
                       dummy_angle_weight[angle], params, ex_group_idx, \
                       ex_angle_idx)
            elif params[2] == 1:
                reflected(angular_flux[:,angle], scalar_flux, medium_map, \
                          xs_total, xs_scatter, external_source, \
                          point_source[angle], spatial_coef[angle], \
                          dummy_angle_weight[angle], params, \
                          ex_group_idx, ex_angle_idx)
            else:
                print("You are wrong")            
        change = angular_convergence(angular_flux, angular_flux_old, angle_weight)
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        angular_flux_old[:,:] = angular_flux[:,:]
    return angular_flux[:,:]


cdef double[:,:] x_time_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
                              double[:]& xs_total, double[:]& xs_matrix, \
                              double[:]& external_source, \
                              double[:]& point_source, double[:]& spatial_coef, \
                              double[:]& angle_weight, int[:]& params, \
                              double temporal_coef, double time_const, \
                              size_t ex_group_idx):    
    cdef int cells = medium_map.shape[0]
    cdef int angles = angle_weight.shape[0]
    cdef size_t ex_angle_idx, cell, angle

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")
    cdef double[:,:] angular_flux = arr2d
    angular_flux[:,:] = 0

    arr2d_1 = cvarray((cells, angles), itemsize=sizeof(double), format="d")
    cdef double[:,:] angular_flux_last = arr2d_1
    angular_flux_last[:,:] = angular_flux_old[:,:]

    arr1d_1 = cvarray((angles,), itemsize=sizeof(double), format="d")
    cdef double[:] dummy_angle_weight = arr1d_1
    dummy_angle_weight[:] = 1

    arr1d_2 = cvarray((cells,), itemsize=sizeof(double), format="d")
    cdef double[:] scalar_flux = arr1d_2
    scalar_flux[:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angular_flux[:,:] = 0
        angular_to_scalar(scalar_flux, angular_flux_old, angle_weight)
        for angle in range(angles):
            ex_angle_idx = 0 if params[4] == 1 else angle
            time_vacuum(angular_flux[:,angle], scalar_flux, \
                    angular_flux_last[:,angle], medium_map, xs_total, \
                    xs_matrix, external_source, point_source[angle], \
                    spatial_coef[angle], dummy_angle_weight[angle], \
                    params, temporal_coef, time_const, ex_group_idx, ex_angle_idx)
        change = angular_convergence(angular_flux, angular_flux_old, angle_weight)
        # print("In Count", count, "Change", change)
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        angular_flux_old[:,:] = angular_flux[:,:]
    return angular_flux[:,:]


cdef void vacuum(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                int[:]& medium_map, double[:]& xs_total, \
                double[:]& xs_scatter, double[:]& external_source, \
                double point_source, double spatial_coef, \
                double angle_weight, int[:]& params, size_t ex_group_idx, \
                size_t ex_angle_idx):
    cdef double edge_one = 0
    cdef double edge_two = 0
    cdef int material, cell
    cdef int cells = medium_map.shape[0]
    if spatial_coef > 0:
        for cell in range(cells):
            material = medium_map[cell]
            if cell == params[5]:
                edge_one += point_source
            edge_two = (xs_scatter[material] * scalar_flux_old[cell] \
                        + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                        + edge_one * (abs(spatial_coef) \
                        - 0.5 * xs_total[material])) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
            if params[1] == 2:
                scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
            elif params[1] == 1:
                scalar_flux[cell] += angle_weight * edge_two
            edge_one = edge_two
    elif spatial_coef < 0:
        for cell in range(cells-1, -1, -1):
            material = medium_map[cell]
            if (cell + 1) == params[5]:
                edge_two += point_source
            edge_one = (xs_scatter[material] * scalar_flux_old[cell] \
                        + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                        + edge_two * (abs(spatial_coef) \
                        - 0.5 * xs_total[material])) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
            if params[1] == 2:
                scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
            elif params[1] == 1:
                scalar_flux[cell] += angle_weight * edge_one
            edge_two = edge_one


cdef void reflected(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                    int[:]& medium_map, double[:]& xs_total, \
                    double[:]& xs_scatter, double[:]& external_source, \
                    double point_source, double spatial_coef, \
                    double angle_weight, int[:]& params, \
                    size_t ex_group_idx, size_t ex_angle_idx):
    cdef double edge_one = 0
    cdef double edge_two = 0
    cdef int material, cell
    cdef int cells = medium_map.shape[0]

    for cell in range(cells):
        material = medium_map[cell]
        if cell == params[5]:
            edge_one += point_source
        edge_two = (xs_scatter[material] * scalar_flux_old[cell] \
                    + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                    + edge_one * (abs(spatial_coef) \
                    - 0.5 * xs_total[material])) \
                    * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
        if params[1] == 2:
            scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
        elif params[1] == 1:
            scalar_flux[cell] += angle_weight * edge_two
        edge_one = edge_two

    for cell in range(cells-1, -1, -1):
        edge_two = edge_one
        material = medium_map[cell]
        if (cell + 1) == params[5]:
            edge_two += point_source
        edge_one = (xs_scatter[material] * scalar_flux_old[cell] \
                    + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                    + edge_two * (abs(spatial_coef) \
                    - 0.5 * xs_total[material])) \
                    * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
        if params[1] == 2:
            scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
        elif params[1] == 1:
            scalar_flux[cell] += angle_weight * edge_one


cdef void time_vacuum(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                        double [:]& angular_flux_last, int[:]& medium_map, \
                        double[:]& xs_total, double[:]& xs_matrix, \
                        double[:]& external_source, \
                        double point_source, double spatial_coef, \
                        double angle_weight, int[:]& params, \
                        double temporal_coef, \
                        double time_const, size_t ex_group_idx, \
                        size_t ex_angle_idx):

    cdef double edge_one = 0
    cdef double edge_two = 0
    cdef int material, cell
    cdef int cells = medium_map.shape[0]
    if spatial_coef > 0:
        for cell in range(cells):
            material = medium_map[cell]
            if cell == params[5]:
                edge_one += point_source
            edge_two = (xs_matrix[material] * scalar_flux_old[cell] \
                        + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                         + angular_flux_last[cell] * temporal_coef + edge_one * (abs(spatial_coef) \
                        - 0.5 * xs_total[material] - time_const * temporal_coef)) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material] + time_const * temporal_coef)
            if params[1] == 2:
                scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
            elif params[1] == 1:
                scalar_flux[cell] += angle_weight * edge_two
            edge_one = edge_two
    elif spatial_coef < 0:
        for cell in range(cells-1, -1, -1):
            material = medium_map[cell]
            if (cell + 1) == params[5]:
                edge_two += point_source
            edge_one = (xs_matrix[material] * scalar_flux_old[cell] \
                        + external_source[ex_group_idx + ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                         + angular_flux_last[cell] * temporal_coef + edge_two * (abs(spatial_coef) \
                        - 0.5 * xs_total[material] - time_const * temporal_coef)) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material] + time_const * temporal_coef)
            if params[1] == 2:
                scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
            elif params[1] == 1:
                scalar_flux[cell] += angle_weight * edge_one
            edge_two = edge_one


cdef void angular_to_scalar(double[:]& scalar_flux, 
                    double[:,:]& angular_flux, double[:]& angle_weight):
    cdef size_t cells, angles, cell, angle
    cells = angular_flux.shape[0]
    angles = angular_flux.shape[1]
    scalar_flux[:] = 0
    for angle in range(angles):
        for cell in range(cells):
            scalar_flux[cell] += angle_weight[angle] * angular_flux[cell][angle]


cdef double scalar_convergence(double [:]& arr1, double [:]& arr2):
    n = arr1.shape[0]
    cdef double change = 0.0
    for cell in range(<int> n):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / n, 2)
    change = sqrt(change)
    return change


cdef double angular_convergence(double[:,:]& angular_flux, 
                double [:,:]& angular_flux_old, double [:]& angle_weight):
    cdef size_t cells, angles
    cells = angular_flux.shape[0]
    angles = angular_flux.shape[1]
    cdef double change = 0.0
    cdef double scalar_flux, scalar_flux_old
    for cell in range(cells):
        scalar_flux = 0
        scalar_flux_old = 0
        for angle in range(angles):
            scalar_flux += angle_weight[angle] * angular_flux[cell][angle]
            scalar_flux_old += angle_weight[angle] * angular_flux_old[cell][angle]
        change += pow((scalar_flux - scalar_flux_old) / \
                        scalar_flux / cells, 2)
    change = sqrt(change)
    return change