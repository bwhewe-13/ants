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

from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE
from ants.cyants.x_sweeps cimport scalar_x_sweep, angular_x_sweep, time_x_sweep
from ants.cyants.r_sweeps cimport r_sweep

# from libcpp cimport float
from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np
from tqdm import tqdm

def criticality(int[:] medium_map, double[:,:] xs_total, \
            double[:,:,:] xs_scatter, double[:,:,:] xs_fission, double[:] mu, \
            double[:] angle_weight, int[:] params, double cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    scalar_flux_old = np.random.rand(cells, groups)
    keff = normalize_flux(scalar_flux_old)
    divide_by_keff(scalar_flux_old, keff)

    power_source = memoryview(np.zeros((cells * groups)))
    scalar_flux = scalar_flux_old.copy()
    point_source = memoryview(np.zeros((angles)))

    if params[0] == 1:
        _spatial_coef(mu, mu, cell_width)

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        power_iteration_source(power_source, scalar_flux_old, \
                               medium_map, xs_fission)
        scalar_flux = scalar_multi_group(scalar_flux, medium_map, \
                    xs_total, xs_scatter, power_source, point_source, \
                    mu, angle_weight, params, cell_width)
        keff = normalize_flux(scalar_flux)
        divide_by_keff(scalar_flux, keff)
        change = scalar_convergence(scalar_flux, scalar_flux_old)
#        print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
#                 '='*35, change, keff))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return np.asarray(scalar_flux), keff


cdef void power_iteration_source(double[:] power_source, double[:,:] flux, \
                                 int[:] medium_map, double[:,:,:] xs_fission):
    power_source[:] = 0
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = flux.shape[1]
    cdef size_t material
    for cell in range(cells):
        material = medium_map[cell]
        for ingroup in range(groups):
            for outgroup in range(groups):
                power_source[ingroup::groups][cell] += flux[cell][outgroup] \
                                * xs_fission[material][ingroup][outgroup]


cdef double normalize_flux(double[:,:] scalar_flux):
    cdef double keff = 0
    cdef size_t cells = scalar_flux.shape[0]
    cdef size_t groups = scalar_flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            keff += pow(scalar_flux[cell][group], 2)
    keff = sqrt(keff)
    return keff


cdef void divide_by_keff(double[:,:] scalar_flux, double keff):
    cdef size_t cells = scalar_flux.shape[0]
    cdef size_t groups = scalar_flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            scalar_flux[cell][group] /= keff


def source_iteration(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double [:] point_source, \
                    double[:] mu, double[:] angle_weight, int[:] params, \
                    double cell_width, bint angular=False):
    cells = medium_map.shape[0]
    groups = xs_total.shape[1]
    angles = mu.shape[0]
    materials = xs_total.shape[0]
    xs_matrix = memoryview(np.zeros((materials, groups, groups)))
    combine_self_scattering(xs_matrix, xs_scatter, xs_fission)

    spatial_coef = memoryview(np.zeros((angles)))
    if params[0] == 1:
        _spatial_coef(spatial_coef, mu, cell_width)
    elif params[0] == 2:
        spatial_coef[:] = mu[:]

    if angular == True:
        angular_flux_old = memoryview(np.zeros((cells, angles, groups)))
        return angular_multi_group(angular_flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, point_source, \
                            spatial_coef, angle_weight, params, cell_width)
    else:
        scalar_flux_old = memoryview(np.zeros((cells, groups)))
        return scalar_multi_group(scalar_flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, point_source, \
                            spatial_coef, angle_weight, params, cell_width)


cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission):
    cdef size_t materials = xs_matrix.shape[0]
    cdef size_t groups = xs_matrix.shape[1]
    for mat in range(materials):
        for ing in range(groups):
            for outg in range(groups):
                xs_matrix[mat][ing][outg] = xs_scatter[mat][ing][outg] \
                                            + xs_fission[mat][ing][outg]


cdef void _spatial_coef(double[:]& spatial_coef, double[:]& mu, \
                        double cell_width):
    cdef size_t angles = mu.shape[0]
    for angle in range(angles):
        spatial_coef[angle] = mu[angle] / cell_width


cdef double[:,:] scalar_multi_group(double[:,:] scalar_flux_old, \
            int[:] medium_map, double[:,:] xs_total, double[:,:,:] xs_matrix,\
            double[:] external_source, double[:] point_source, double[:] mu, \
            double[:] angle_weight, int[:] params, double cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, ps_group_idx

    arr2d = cvarray((cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] scalar_flux = arr2d

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
            ps_group_idx = 0 if params[6] == 1 else group
            one_group_flux_old[:] = scalar_flux_old[:,group]
            if params[0] == 1:
                # print("Here to slab")
                scalar_flux[:,group] = scalar_x_sweep(one_group_flux_old, \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    external_source, point_source[ps_group_idx::params[6]], \
                    mu, angle_weight, params, ex_group_idx)
            elif params[0] == 2:
                # print("Here to sphere")
                scalar_flux[:,group] = r_sweep(one_group_flux_old, \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    external_source, point_source[ps_group_idx::params[6]], \
                    mu, angle_weight, params, cell_width, ex_group_idx)
        change = scalar_convergence(scalar_flux, scalar_flux_old)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        scalar_flux_old[:,:] = scalar_flux[:,:]
    return np.asarray(scalar_flux)

 
cdef double[:,:,:] angular_multi_group(double[:,:,:] angular_flux_old, \
            int[:] medium_map, double[:,:] xs_total, double[:,:,:] xs_matrix, \
            double[:] external_source, double[:] point_source, double[:] mu, \
            double[:] angle_weight, int[:] params, double cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    cdef size_t ex_group_idx, ps_group_idx

    arr3d = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] angular_flux = arr3d

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
            ps_group_idx = 0 if params[6] == 1 else group            
            angular_flux[:,:,group] = angular_x_sweep(one_group_flux_old, \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    external_source, point_source[ps_group_idx::params[6]], \
                    mu, angle_weight, params, ex_group_idx)
        change = angular_convergence(angular_flux, angular_flux_old, angle_weight)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
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
    cdef size_t groups = velocity.shape[0]
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
            angular_flux_next[:,:,group] = time_x_sweep(one_group_flux, \
                            medium_map, xs_total[:,group], \
                            xs_matrix[:,group,group], \
                            external_source, \
                            point_source[ps_group_idx::params[6]], \
                            spatial_coef, angle_weight, params, \
                            temporal_coef[group], \
                            time_const, ex_group_idx)
        change = angular_convergence(angular_flux_next, angular_flux_last, angle_weight)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        angular_flux_last[:,:,:] = angular_flux_next[:,:,:]
    return angular_flux_next


cdef void angular_to_scalar(double[:,:,:]& time_step_flux, \
            double[:,:,:]& angular_flux, double[:]& angle_weight, \
            size_t time_step):
    # cdef size_t cell, angle, group
    cdef size_t cells = angular_flux.shape[0]
    cdef size_t angles = angular_flux.shape[1]
    cdef size_t groups = angular_flux.shape[2]
    for group in range(groups):
        for angle in range(angles):
            for cell in range(cells):
                time_step_flux[time_step][cell][group] += angle_weight[angle] \
                                           * angular_flux[cell][angle][group]
