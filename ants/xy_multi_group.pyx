########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE
from ants cimport cyutils, xy_sweeps

from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np

cdef double[:,:] scalar_multi_group(double[:,:]& flux_old, int[:] medium_map, \
                    double[:,:] xs_total, double[:,:,:] xs_matrix, \
                    double[:] external_source, double[:] boundary, \
                    double[:] mu, double[:] eta, double[:] angle_weight, \
                    int[:] params, double[:] delta_x, double[:] delta_y):
    cdef int cells = medium_map.shape[0]
    cdef int groups = xs_total.shape[1]
    cdef int gg_idx, bc_group_idx
    arr2d = cvarray((cells, groups), itemsize=sizeof(double), format="d")
    cdef double[:,:] flux = arr2d

    arr1d_1 = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] one_group_flux_old = arr1d_1
    one_group_flux_old[:] = 0

    arr1d_2 = cvarray((cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] off_scatter = arr1d_2
    
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:, :] = 0
        for group in range(groups):
            gg_idx = 0 if params[4] == 1 else group
            bc_group_idx = 0 if params[7] == 1 else group
            one_group_flux_old[:] = flux_old[:,group]
            cyutils.off_scatter_source(flux, flux_old, medium_map, xs_matrix, \
                                off_scatter, group)
            # Rectangle params[0] == 1
            flux[:,group] = xy_sweeps.scalar_single_sweep(one_group_flux_old, \
                        medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                        off_scatter, external_source, boundary[bc_group_idx::params[7]], \
                        mu, eta, angle_weight, params, delta_x, delta_y, gg_idx)
        change = cyutils.scalar_convergence(flux, flux_old)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        # print("Source Iteration {}\n{}\nChange {} Sum {}".format(count, \
        #         "="*35, change, np.sum(flux)))
        count += 1
        flux_old[:,:] = flux[:,:]
    return np.asarray(flux)

def criticality(int[:] medium_map, double[:,:] xs_total, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission, \
                double[:] mu, double[:] eta, double[:] angle_weight, \
                int[:] params, double[:] delta_x, double[:] delta_y):

    cdef int cells = medium_map.shape[0]
    cdef int groups = xs_total.shape[1]
    cdef int angles = mu.shape[0]
    flux_old = np.random.rand(cells, groups)
    keff = cyutils.normalize_flux(flux_old)
    cyutils.divide_by_keff(flux_old, keff)

    power_source = memoryview(np.zeros((cells * groups)))
    flux = flux_old.copy()
    boundary = memoryview(np.zeros((angles)))

    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    while not (converged):
        cyutils.power_iteration_source(power_source, flux_old, medium_map, xs_fission, 1.0)
        flux = scalar_multi_group(flux, medium_map, xs_total, xs_scatter, \
                                  power_source, boundary, mu, eta, \
                                  angle_weight, params, delta_x, delta_y)
        keff = cyutils.normalize_flux(flux)
        cyutils.divide_by_keff(flux, keff)
        change = cyutils.scalar_convergence(flux, flux_old)
        print("Power Iteration {}\n{}\nChange {} Keff {}".format(count, \
                "="*35, change, keff))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old = flux.copy()
    return np.asarray(flux), keff
