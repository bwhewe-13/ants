########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional DJINN Criticality Multigroup Neutron Transport Problems
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

import numpy as np
from libc.math cimport isnan, isinf

from ants.spatial_sweep_1d cimport discrete_ordinates
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters
from ants.constants import *


# def collection(double[:,:] xs_total, double[:,:,:] xs_scatter, \
#         double[:,:,:] xs_fission, int[:] medium_map, double[:] delta_x, \
#         double[:] angle_x, double[:] angle_w, dict params_dict, str file):
#     # Convert dictionary to type params1d
#     info = parameters._to_params(params_dict)
#     parameters._check_critical1d_power_iteration(info)
#     # Initialize keff
#     cdef double keff[1]
#     keff[0] = 0.95
#     # Initialize and normalize flux
#     flux_old = np.random.rand(info.cells_x, info.groups)
#     tools._normalize_flux(flux_old, info)
#     # Solve using the power iteration
#     flux = multigroup_collect(flux_old, xs_total, xs_scatter, xs_fission, \
#                 medium_map, delta_x, angle_x, angle_w, info, keff, file)
#     # Save relevant information to file
#     np.save(file + "fission_cross_sections", np.asarray(xs_fission))
#     np.save(file + "scatter_cross_sections", np.asarray(xs_scatter))
#     np.save(file + "medium_map", np.asarray(medium_map))
#     np.savez(file + "problem_information", **params_dict)
#     return np.asarray(flux), keff[0]


# cdef double[:,:] multigroup_collect(double[:,:]& flux_guess, double[:,:]& xs_total, \
#         double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, int[:]& medium_map, \
#         double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, params info, \
#         double[:]& keff, str file):
#     # Initialize flux
#     flux = tools.array_2d(info.cells_x, info.groups)
#     flux_old = flux_guess.copy()
#     # Initialize power source
#     source = tools.array_1d(info.cells_x * info.groups)
#     tracked_flux = tools.array_3d(MAX_POWER, info.cells_x, info.groups)
#     # Vacuum boundaries
#     cdef double[2] boundary_x = [0.0, 0.0]
#     # Set convergence limits
#     cdef bint converged = False
#     cdef int count = 1
#     cdef double change = 0.0
#     while not (converged):
#         # Update power source term
#         tools._fission_source(flux_old, xs_fission, source, medium_map, \
#                               info, keff[0])
#         # Solve for scalar flux
#         flux = source_iteration_collect(flux_old, xs_total, xs_scatter, \
#                                 source, boundary_x, medium_map, delta_x, \
#                                 angle_x, angle_w, info, count, file)
#         # Update keffective
#         keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
#                                            medium_map, info, keff[0])
#         # Normalize flux
#         tools._normalize_flux(flux, info)
#         # Check for convergence
#         change = tools.group_convergence(flux, flux_old, info)
#         print("Count {}\tKeff {}".format(str(count).zfill(3), keff[0]), end="\r")
#         converged = (change < EPSILON_POWER) or (count >= MAX_POWER)
#         tracked_flux[count-1] = flux[:,:]
#         count += 1
#         flux_old[:,:] = flux[:,:]
#     print("\nConvergence:", change)
#     np.save(file + "flux_fission_model", np.asarray(tracked_flux[:count-1]))
#     return flux[:,:]


# cdef double[:,:] source_iteration_collect(double[:,:]& flux_guess, \
#         double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
#         double[:]& external, double [:]& boundary_x, int[:]& medium_map, \
#         double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
#         params info, int iteration, str file):
#     # Initialize components
#     cdef int gg, qq1, qq2, bc1, bc2
#     # Set indexing
#     qq2 = 1 if info.qdim == 1 else info.groups
#     bc2 = 1 if info.bcdim_x == 1 else info.groups
#     # Initialize flux
#     flux = tools.array_2d(info.cells_x, info.groups)
#     flux_old = flux_guess.copy()
#     flux_1g = tools.array_1d(info.cells_x)
#     # Create off-scattering term
#     off_scatter = tools.array_1d(info.cells_x)
#     tracked_flux = tools.array_3d(MAX_ENERGY, info.cells_x, info.groups)
#     # Set convergence limits
#     cdef bint converged = False
#     cdef size_t count = 1
#     cdef double change = 0.0
#     # Iterate until energy group convergence
#     while not (converged):
#         flux[:,:] = 0.0
#         for gg in range(info.groups):
#             qq1 = 0 if info.qdim == 1 else gg
#             bc1 = 0 if info.bcdim_x == 1 else gg
#             # Select the specific group from last iteration
#             flux_1g[:] = flux_old[:,gg]
#             # Calculate up and down scattering term using Gauss-Seidel
#             tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
#                                off_scatter, info, gg)
#             # Use discrete ordinates for the angular dimension
#             discrete_ordinates(flux[:,gg], flux_1g, xs_total[:,gg], \
#                                xs_scatter[:,gg,gg], off_scatter, external[qq1::qq2], \
#                                boundary_x[bc1::bc2], medium_map, delta_x, \
#                                angle_x, angle_w, info)
#         change = tools.group_convergence(flux, flux_old, info)
#         if isnan(change) or isinf(change):
#             change = 0.5
#         converged = (change < EPSILON_ENERGY) or (count >= MAX_ENERGY)
#         tracked_flux[count-1] = flux[:,:]
#         count += 1
#         flux_old[:,:] = flux[:,:]
#     np.save(file + "flux_scatter_model_{}".format(str(iteration).zfill(3)), \
#             np.asarray(tracked_flux)[:count-1])
#     return flux[:,:]