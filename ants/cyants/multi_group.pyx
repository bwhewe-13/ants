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
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                double[:] mu, double[:] angle_weight, int[:] params, \
                double[:] cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    flux_old = np.random.rand(cells, groups)
    keff = normalize_flux(flux_old)
    divide_by_keff(flux_old, keff)

    power_source = memoryview(np.zeros((cells * groups)))
    flux = flux_old.copy()
    point_source = memoryview(np.zeros((angles)))

    # if params[0] == 1:
    #     _spatial_coef(mu, mu, cell_width)

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        power_iteration_source(power_source, flux_old, medium_map, xs_fission)
        flux = scalar_multi_group(flux, medium_map, xs_total, xs_scatter, \
                power_source, point_source, mu, angle_weight, params, cell_width)
        keff = normalize_flux(flux)
        divide_by_keff(flux, keff)
        change = scalar_convergence(flux, flux_old)
        # print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
        #         '='*35, change, keff))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old = flux.copy()
    return np.asarray(flux), keff


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


cdef double normalize_flux(double[:,:] flux):
    cdef double keff = 0
    cdef size_t cells = flux.shape[0]
    cdef size_t groups = flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            keff += pow(flux[cell][group], 2)
    keff = sqrt(keff)
    return keff


cdef void divide_by_keff(double[:,:] flux, double keff):
    cdef size_t cells = flux.shape[0]
    cdef size_t groups = flux.shape[1]
    for group in range(groups):
        for cell in range(cells):
            flux[cell][group] /= keff


def source_iteration(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double [:] point_source, \
                    double[:] mu, double[:] angle_weight, int[:] params, \
                    double[:] cell_width, bint angular=False):
    cells = medium_map.shape[0]
    groups = xs_total.shape[1]
    angles = mu.shape[0]
    materials = xs_total.shape[0]
    xs_matrix = memoryview(np.zeros((materials, groups, groups)))
    combine_self_scattering(xs_matrix, xs_scatter, xs_fission)

    # spatial_coef = memoryview(np.zeros((angles)))
    # if params[0] == 1:
    #     _spatial_coef(spatial_coef, mu, cell_width)
    # elif params[0] == 2:
    #     spatial_coef[:] = mu[:]

    if angular == True:
        flux_old = memoryview(np.zeros((cells, angles, groups)))
        return angular_multi_group(flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, point_source, \
                            mu, angle_weight, params, cell_width)
    else:
        flux_old = memoryview(np.zeros((cells, groups)))
        return scalar_multi_group(flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, point_source, \
                            mu, angle_weight, params, cell_width)


cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission):
    cdef size_t materials = xs_matrix.shape[0]
    cdef size_t groups = xs_matrix.shape[1]
    for mat in range(materials):
        for ing in range(groups):
            for outg in range(groups):
                xs_matrix[mat][ing][outg] = xs_scatter[mat][ing][outg] \
                                            + xs_fission[mat][ing][outg]


# cdef void _spatial_coef(double[:]& spatial_coef, double[:]& mu, \
#                         double cell_width):
#     cdef size_t angles = mu.shape[0]
#     for angle in range(angles):
#         spatial_coef[angle] = mu[angle] / cell_width


cdef void off_scatter_source(double[:,:]& flux, double[:,:]& flux_old, \
                             int[:]& medium_map, double[:,:,:]& xs_matrix, \
                             double[:]& source, size_t group):
    cdef size_t groups = flux.shape[1]
    cdef size_t cells = medium_map.shape[0]
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


cdef double[:,:] scalar_multi_group(double[:,:]& flux_old, int[:] medium_map, \
            double[:,:] xs_total, double[:,:,:] xs_matrix,\
            double[:] external_source, double[:] point_source, double[:] mu, \
            double[:] angle_weight, int[:] params, double[:] cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, ps_group_idx
    arr2d = cvarray((cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] flux = arr2d

    arr1d_1 = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] one_group_flux_old = arr1d_1
    one_group_flux_old[:] = 0

    arr1d_2 = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] off_scatter = arr1d_2
    
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:, :] = 0
        for group in range(groups):
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[6] == 1 else group
            one_group_flux_old[:] = flux_old[:,group]
            off_scatter_source(flux, flux_old, medium_map, xs_matrix, \
                                off_scatter, group)
            if params[0] == 1:
                flux[:,group] = scalar_x_sweep(one_group_flux_old, medium_map, \
                    xs_total[:,group], xs_matrix[:,group,group], off_scatter, \
                    external_source, point_source[ps_group_idx::params[6]], \
                    mu, angle_weight, params, cell_width, ex_group_idx)
            elif params[0] == 2:
                flux[:,group] = r_sweep(one_group_flux_old, medium_map, \
                    xs_total[:,group], xs_matrix[:,group,group], off_scatter, \
                    external_source, point_source[ps_group_idx::params[6]], \
                    mu, angle_weight, params, cell_width, ex_group_idx)
        change = scalar_convergence(flux, flux_old)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    return np.asarray(flux)

 
cdef double[:,:,:] angular_multi_group(double[:,:,:] flux_old, \
            int[:] medium_map, double[:,:] xs_total, double[:,:,:] xs_matrix, \
            double[:] external_source, double[:] point_source, double[:] mu, \
            double[:] angle_weight, int[:] params, double[:] cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    cdef size_t ex_group_idx, ps_group_idx

    arr3d = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux = arr3d

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")    
    cdef double[:,:] one_group_flux_old = arr2d
    one_group_flux_old[:,:] = 0

    arr1d = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] off_scatter = arr1d

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:,:] = 0
        for group in range(groups):
            one_group_flux_old[:,:] = flux_old[:,:,group]
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[6] == 1 else group
            off_scatter_source_angular(flux, flux_old, medium_map, \
                            xs_matrix, off_scatter, group, angle_weight)
            flux[:,:,group] = angular_x_sweep(one_group_flux_old, \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    off_scatter, external_source, \
                    point_source[ps_group_idx::params[6]], mu, angle_weight, \
                    params, cell_width, ex_group_idx)
        change = angular_convergence(flux, flux_old, angle_weight)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    return np.asarray(flux)


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
                   double[:,:,:] xs_matrix, double[:] external_source, \
                   double[:] point_source, double[:] spatial_coef, \
                   double[:] angle_weight, double[:] velocity, \
                   int[:] params, double time_step_size):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]

    arr1d = cvarray((groups,), itemsize=sizeof(double), format="d")
    cdef double[:] temporal_coef = arr1d
    _time_coef(temporal_coef, velocity, time_step_size)

    arr3d_1 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux = arr3d_1
    flux[:,:,:] = 0

    arr3d_2 = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux_last = arr3d_2
    flux_last[:,:,:] = 0

    arr3d_3 = cvarray((params[8], cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] time_step_flux = arr3d_3
    time_step_flux[:,:,:] = 0

    cdef double time_const 
    if params[7] == 1:
        time_const = 0.5
    else:
        time_const = 0.75
    for step in tqdm(range(params[8]), desc="Time Steps"): 
        flux = time_source_iteration(flux_last, medium_map, xs_total, \
                xs_matrix, external_source, point_source, spatial_coef, \
                angle_weight, temporal_coef, params, time_const)
        angular_to_scalar(time_step_flux[step], flux, angle_weight)
        flux_last[:,:,:] = flux[:,:,:]
    return np.asarray(time_step_flux)


cdef void _time_coef(double[:]& temporal_coef, double[:]& velocity, \
                     double time_step_size):
    cdef size_t groups = velocity.shape[0]
    for group in range(groups):
        temporal_coef[group] = 1 / (velocity[group] * time_step_size)


cdef double[:,:,:] time_source_iteration(double[:,:,:]& flux_last, \
                    int[:]& medium_map, double[:,:]& xs_total, \
                    double[:,:,:]& xs_matrix, double[:]& external_source, \
                    double[:]& point_source, double[:]& spatial_coef, \
                    double[:]& angle_weight, double[:]& temporal_coef, \
                    int[:]& params, double time_const):
    
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, ps_group_idx

    arr3d = cvarray((cells, angles, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] flux_next = arr3d
    flux_next[:,:,:] = 0

    arr2d = cvarray((cells, angles), itemsize=sizeof(double), format="d")    
    cdef double[:,:] one_group_flux = arr2d
    one_group_flux[:,:] = 0

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux_next[:,:,:] = 0
        for group in range(groups):
            one_group_flux[:,:] = flux_last[:,:,group]
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[5] == 1 else group
            flux_next[:,:,group] = time_x_sweep(one_group_flux, medium_map, \
                        xs_total[:,group], xs_matrix[:,group,group], \
                        external_source, point_source[ps_group_idx::params[6]], \
                        spatial_coef, angle_weight, params, temporal_coef[group], \
                        time_const, ex_group_idx)
        change = angular_convergence(flux_next, flux_last, angle_weight)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_last[:,:,:] = flux_next[:,:,:]
    return flux_next


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
