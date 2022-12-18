########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Criticality Multigroup Neutron Transport Problems
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

# from ants.csweeps1d cimport slab1d
# from ants cimport cyutils
# from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE

from ants cimport power_iteration as pi
from ants cimport cytools as tools
from ants.cytools cimport params1d
from ants cimport cytools_crit as ctools

# from libc.math cimport sqrt, pow
# from cython.view cimport array as cvarray
import numpy as np

def power_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
                    double[:,:,:] xs_fission, int[:] medium_map, \
                    double[:] cell_width, double[:] mu, \
                    double[:] angle_w, dict params_dict):
    cdef double keff[1]
    keff[0] = 0.5
    # Convert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Initialize components
    flux_old = np.random.rand(params.cells * params.groups)
    ctools.normalize_flux(flux_old, params)
    power_source = memoryview(np.zeros((params.cells * params.groups)))
    flux = pi.multigroup(flux_old, xs_total, xs_scatter, xs_fission, \
                    power_source, medium_map, cell_width, mu, \
                    angle_w, params, keff)
    return np.asarray(flux).reshape(params.cells, params.groups), keff[0]


# def adjoint(double[:,:] xs_total, double[:,:,:] xs_scatter, \
#                     double[:,:,:] xs_fission, int[:] medium_map, \
#                     double[:] cell_width, double[:] mu, \
#                     double[:] angle_w, dict params_dict):
#     cdef double keff[1]
#     keff[0] = 0.5
#     # Convert dictionary to type params1d
#     params = tools._to_params1d(params_dict)
#     # Initialize components
#     flux_old = np.random.rand(params.cells * params.groups)
#     ctools.normalize_flux(flux_old, params)
#     power_source = memoryview(np.zeros((params.cells * params.groups)))
#     flux = pi.adjoint(flux_old, xs_total, xs_scatter, xs_fission, \
#                     power_source, medium_map, cell_width, mu, \
#                     angle_w, params, keff)
#     return np.asarray(flux).reshape(params.cells, params.groups), keff[0]

# cdef double[:] pi_multi(double[:] flux_old, double[:,:] xs_total, \
#                         double[:,:,:] xs_scatter, double[:,:,:] xs_fission, 
#                         double[:] power_source, int[:] medium_map, \
#                         double[:] cell_width, double[:] mu, \
#                         double[:] angle_weight, paramsc1d params, \
#                         double[:] keff):
#     # Initialize flux
#     cdef double[:] flux = cvarray((params.cells * params.groups,), \
#                                     itemsize=sizeof(double), format="d")
#     flux[:] = flux_old[:]
#     # Set convergence limits
#     cdef bint converged = False
#     cdef size_t count = 1
#     cdef double change = 0.0
#     while not (converged):
#         pi_source(power_source, flux_old, xs_fission, medium_map, \
#                                 params, keff)
#         flux = source_iteration(flux, xs_total, xs_scatter, \
#                         power_source, medium_map, cell_width, mu, \
#                         angle_weight, params)
#         keff[0] = update_keffective(flux, flux_old, medium_map, \
#                                     xs_fission, params, keff[0])
#         normalize_flux(flux, params)
#         change = scalar_convergence(flux, flux_old, params)
#         print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
#                 '='*35, change, keff[0]))
#         converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
#         count += 1
#         flux_old[:] = flux[:]
#     return flux[:]
    
# cdef paramsc1d dict_to_paramsc1d(dict params_dict):
#     cdef paramsc1d params
#     params.cells = params_dict["cells"]
#     params.angles = params_dict["angles"]
#     params.groups = params_dict["groups"]
#     params.geometry = params_dict["geometry"] 
#     params.spatial = params_dict["spatial"]
#     params.qdim = params_dict["qdim"]
#     return params

# cdef void normalize_flux(double[:]& flux, paramsc1d params):
#     cdef size_t cell
#     cdef double keff = 0.0
#     for cell in range(params.cells * params.groups):
#         keff += pow(flux[cell], 2)
#     keff = sqrt(keff)
#     for cell in range(params.cells * params.groups):
#         flux[cell] /= keff

# cdef double update_keffective(double[:] flux, double[:] flux_old, \
#                             int[:] medium_map, double[:,:,:] xs_fission, \
#                             paramsc1d params, double keff_old):
#     cdef double frate = 0.0
#     cdef double frate_old = 0.0
#     cdef size_t cell, mat, ig, og
#     for cell in range(params.cells):
#         mat = medium_map[cell]
#         for ig in range(params.groups):
#             for og in range(params.groups):
#                 frate += flux[og::params.groups][cell] * xs_fission[mat][ig][og]
#                 frate_old += flux_old[og::params.groups][cell] * xs_fission[mat][ig][og]
#     return (frate * keff_old) / frate_old

# cdef void pi_source(double[:] power_source, double[:] flux, \
#                     double[:,:,:] xs_fission, int[:] medium_map, \
#                     paramsc1d params, double[:] keff):
#     power_source[:] = 0.0
#     cdef size_t cell, mat, ig, og
#     for cell in range(params.cells):
#         mat = medium_map[cell]
#         for ig in range(params.groups):
#             for og in range(params.groups):
#                 power_source[ig::params.groups][cell] += (1 / keff[0]) \
#                                     * flux[og::params.groups][cell] \
#                                     * xs_fission[mat][ig][og] 


# cdef double[:] source_iteration(double[:]& flux_old, double[:,:] xs_total, \
#                                 double[:,:,:] xs_matrix, double[:] power_source, \
#                                 int[:] medium_map, double[:] cell_width, \
#                                 double[:] mu, double[:] angle_weight, \
#                                 paramsc1d params):
#     # Initialize components
#     cdef size_t q_idx1, q_idx2, group
#     cdef double[:] flux = cvarray((params.cells * params.groups,), \
#                                     format="d", itemsize=sizeof(double))
#     cdef double[:] off_scatter = cvarray((params.cells,), format="d", \
#                                     itemsize=sizeof(double))
#     # Set convergence limits
#     cdef bint converged = False
#     cdef size_t count = 1
#     cdef double change = 0.0
#     while not (converged):
#         flux[:] = 0.0
#         for group in range(params.groups):
#             q_idx1 = 0 if params.qdim == 1 else group
#             q_idx2 = 1 if params.qdim == 1 else params.groups
#             off_scatter_scalar(flux, flux_old, medium_map, xs_matrix, \
#                                 off_scatter, params, group)
#             if params.geometry == 1:
#                 flux[group::params.groups] = slab1d( \
#                         flux_old[group::params.groups], xs_total[:,group], \
#                         xs_matrix[:,group,group], off_scatter, \
#                         power_source[q_idx1::q_idx2], medium_map, cell_width, \
#                         mu, angle_weight, params)
#         change = scalar_convergence(flux, flux_old, params)
#         converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
#         count += 1
#         flux_old[:] = flux[:]
#     return flux[:]

# cdef void off_scatter_scalar(double[:]& flux, double[:]& flux_old, \
#                             int[:]& medium_map, double[:,:,:]& xs_matrix, \
#                             double[:]& source, paramsc1d params, size_t group):
#     cdef size_t cell, mat, angle, og
#     source[:] = 0.0
#     for cell in range(params.cells):
#         mat = medium_map[cell]
#         for og in range(0, group):
#             source[cell] += xs_matrix[mat, group, og] \
#                             * flux[og::params.groups][cell]
#         for og in range(group+1, params.groups):
#             source[cell] += xs_matrix[mat, group, og] \
#                             * flux_old[og::params.groups][cell]

# cdef double scalar_convergence(double[:]& arr1, double[:]& arr2, \
#                                 paramsc1d params):
#     cdef size_t cell
#     cdef double change = 0.0
#     for cell in range(params.groups * params.cells):
#         change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / params.cells, 2)
#     change = sqrt(change)
#     return change

# cdef void mnp_power_source(double[:] power_source, double[:,:] flux, \
#                             int[:] medium_map, double[:,:,:] xs_fission, \
#                             paramsc1d params, double keff):
#     power_source[:] = 0.0
#     cdef size_t cell, mat, angle, ig, og
#     for cell in range(params.cells):
#         mat = medium_map[cell]
#         for angle in range(params.angles):
#             for ig in range(params.groups):
#                 for og in range(params.groups):
#                     power_source[ig::params.groups][angle::params.angles][cell] \
#                         += flux[cell][og] * xs_fission[mat][ig][og] / keff