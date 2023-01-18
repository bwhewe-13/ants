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

# cython: profile=True

from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE
from ants cimport x_sweeps, r_sweeps, cyutils

from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np
from tqdm import tqdm

def criticality(int[:] medium_map, double[:,:] xs_total, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                double[:] mu, double[:] angle_weight, int[:] params, \
                double[:] cell_width):
    # Initialize components
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    flux_old = np.random.rand(cells, groups)
    cdef double keff = 0.95
    # keff = cyutils.normalize_flux(flux_old)
    cyutils.normalize_flux(flux_old)
    # cyutils.divide_by_keff(flux_old, keff)
    power_source = memoryview(np.zeros((cells * groups)))
    flux = flux_old.copy()
    boundary = memoryview(np.zeros((angles)))
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        cyutils.power_iteration_source(power_source, flux_old, medium_map, \
                                        xs_fission, keff)
        flux = scalar_multi_group(flux, medium_map, xs_total, xs_scatter, \
                power_source, boundary, mu, angle_weight, params, cell_width)
        keff = cyutils.update_keffective(flux, flux_old, medium_map, xs_fission, keff)
        cyutils.normalize_flux(flux_old)
        change = cyutils.scalar_convergence(flux, flux_old)
        print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
                '='*35, change, keff))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old = flux.copy()
    return np.asarray(flux), keff

def mnp_criticality(int[:] medium_map, double[:,:] xs_total, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                double[:] mu, double[:] angle_weight, int[:] params, \
                double[:] cell_width, double[:] mnp_source, double mnp_keff):
    # mnp_keff is the keff / Sum (sigmaf * phi)
    # Initialize components
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    flux_old = np.random.rand(cells, groups)
    cdef double keff = cyutils.multiply_manufactured_flux(flux_old, mnp_keff)
    power_source = memoryview(np.zeros((cells * angles * groups)))
    flux = flux_old.copy()
    boundary = memoryview(np.zeros((angles)))
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        cyutils.mnp_power_iteration_source(power_source, flux_old, medium_map, \
                                            xs_fission, angles, keff)
        cyutils.add_manufactured_source(power_source, mnp_source)
        flux = scalar_multi_group(flux, medium_map, xs_total, xs_scatter, \
                power_source, boundary, mu, angle_weight, params, cell_width)
        keff = cyutils.update_keffective(flux, flux_old, medium_map, xs_fission, keff)
        change = cyutils.scalar_convergence(flux, flux_old)
        # print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
        #         '='*35, change, keff))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old = flux.copy()
    return np.asarray(flux), keff


def source_iteration(int[:] medium_map, double[:,:] xs_total, \
                    double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                    double[:] external_source, double [:] boundary, \
                    double[:] mu, double[:] angle_weight, int[:] params, \
                    double[:] cell_width, bint angular=False):
    cells = medium_map.shape[0]
    groups = xs_total.shape[1]
    angles = mu.shape[0]
    materials = xs_total.shape[0]
    xs_matrix = memoryview(np.zeros((materials, groups, groups)))
    cyutils.combine_self_scattering(xs_matrix, xs_scatter, xs_fission)
    if angular == True:
        flux_old = memoryview(np.zeros((cells, angles, groups)))
        return angular_multi_group(flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, boundary, \
                            mu, angle_weight, params, cell_width)
    else:
        flux_old = memoryview(np.zeros((cells, groups)))
        return scalar_multi_group(flux_old, medium_map, xs_total, \
                            xs_matrix, external_source, boundary, \
                            mu, angle_weight, params, cell_width)


cdef double[:,:] scalar_multi_group(double[:,:]& flux_old, int[:] medium_map, \
            double[:,:] xs_total, double[:,:,:] xs_matrix,\
            double[:] external_source, double[:] boundary, double[:] mu, \
            double[:] angle_weight, int[:] params, double[:] cell_width):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, bc_group_idx
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
            bc_group_idx = 0 if params[6] == 1 else group
            one_group_flux_old[:] = flux_old[:,group]
            cyutils.off_scatter_source(flux, flux_old, medium_map, xs_matrix, \
                                off_scatter, group)
            if params[0] == 1:
                flux[:,group] = x_sweeps.scalar_sweep(one_group_flux_old, medium_map, \
                    xs_total[:,group], xs_matrix[:,group,group], off_scatter, \
                    external_source, boundary[bc_group_idx::params[6]], \
                    mu, angle_weight, params, cell_width, ex_group_idx)
            elif params[0] == 2:
                flux[:,group] = r_sweeps.r_sweep(one_group_flux_old, medium_map, \
                    xs_total[:,group], xs_matrix[:,group,group], off_scatter, \
                    external_source, boundary[bc_group_idx::params[6]], \
                    mu, angle_weight, params, cell_width, ex_group_idx)
        change = cyutils.scalar_convergence(flux, flux_old)
        # print("Source Iteration {}\n{}\nChange {} Flux {}".format(count, \
        #         "="*35, change, np.sum(np.asarray(flux))))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    # print(count, change)
    return np.asarray(flux)

 
cdef double[:,:,:] angular_multi_group(double[:,:,:] flux_old, \
            int[:] medium_map, double[:,:] xs_total, double[:,:,:] xs_matrix, \
            double[:] external_source, double[:] boundary, double[:] mu, \
            double[:] angle_weight, int[:] params, double[:] cell_width):

    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    cdef size_t ex_group_idx, bc_group_idx

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
            bc_group_idx = 0 if params[6] == 1 else group
            cyutils.off_scatter_source_angular(flux, flux_old, medium_map, \
                            xs_matrix, off_scatter, group, angle_weight)
            flux[:,:,group] = x_sweeps.angular_sweep(one_group_flux_old, \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    off_scatter, external_source, \
                    boundary[bc_group_idx::params[6]], mu, angle_weight, \
                    params, cell_width, ex_group_idx)
        change = cyutils.angular_convergence(flux, flux_old, angle_weight)
        # print("Source Iteration {}\n{}\nChange {} Flux {}".format(count, \
        # "="*35, change, np.sum(np.asarray(flux))))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    # print(count, change)        
    return np.asarray(flux)


def time_dependent(int[:] medium_map, double[:,:] xs_total, \
                   double[:,:,:] xs_matrix, double[:] external_source, \
                   double[:] boundary, double[:] spatial_coef, \
                   double[:] angle_weight, double[:] velocity, \
                   int[:] params, double time_step_size):
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]

    arr1d = cvarray((groups,), itemsize=sizeof(double), format="d")
    cdef double[:] temporal_coef = arr1d
    cyutils.time_coef(temporal_coef, velocity, time_step_size)

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
                xs_matrix, external_source, boundary, spatial_coef, \
                angle_weight, temporal_coef, params, time_const)
        cyutils.angular_to_scalar(time_step_flux[step], flux, angle_weight)
        flux_last[:,:,:] = flux[:,:,:]
    return np.asarray(time_step_flux)


cdef double[:,:,:] time_source_iteration(double[:,:,:]& flux_last, \
                    int[:]& medium_map, double[:,:]& xs_total, \
                    double[:,:,:]& xs_matrix, double[:]& external_source, \
                    double[:]& boundary, double[:]& spatial_coef, \
                    double[:]& angle_weight, double[:]& temporal_coef, \
                    int[:]& params, double time_const):
    
    cdef size_t cells = medium_map.shape[0]
    cdef size_t angles = angle_weight.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t ex_group_idx, bc_group_idx

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
            bc_group_idx = 0 if params[5] == 1 else group
            flux_next[:,:,group] = x_sweeps.time_sweep(one_group_flux, medium_map, \
                        xs_total[:,group], xs_matrix[:,group,group], \
                        external_source, boundary[bc_group_idx::params[6]], \
                        spatial_coef, angle_weight, params, temporal_coef[group], \
                        time_const, ex_group_idx)
        change = cyutils.angular_convergence(flux_next, flux_last, angle_weight)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_last[:,:,:] = flux_next[:,:,:]
    return flux_next
