########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-Dimensional Criticality Multigroup Neutron Transport Problems
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

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants cimport parameters
from ants.constants import *


def power_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, int[:,:] medium_map, double[:] delta_x, \
        double[:] delta_y, double[:] angle_x, double[:] angle_y, \
        double[:] angle_w, dict params_dict):
    # Convert dictionary to type params1d
    info = parameters._to_params(params_dict)
    parameters._check_critical2d_power_iteration(info)
    # Initialize keff
    cdef double keff[1]
    keff[0] = 0.95
    # Initialize and normalize flux
    flux_old = np.random.rand(info.cells_x, info.cells_y, info.groups)
    tools._normalize_flux(flux_old, info)
    # Solve using the power iteration
    flux = multigroup_power(flux_old, xs_total, xs_scatter, xs_fission, \
                            medium_map, delta_x, delta_y, angle_x, \
                            angle_y, angle_w, info, keff)    
    return np.asarray(flux), keff[0]


cdef double[:,:,:] multigroup_power(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params info, double[:]& keff):
    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()
    # Initialize power source
    source = tools.array_1d(info.cells_x * info.cells_y * info.groups)
    # Vacuum boundaries
    cdef double[2] boundary_x = [0.0, 0.0]
    cdef double[2] boundary_y = [0.0, 0.0]
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    while not (converged):
        # Update power source term
        tools._fission_source(flux_old, xs_fission, source, medium_map, \
                              info, keff[0])
        # Solve for scalar flux
        flux = mg.source_iteration(flux_old, xs_total, xs_scatter, source, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
        keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
                                           medium_map, info, keff[0])
        tools._normalize_flux(flux, info)
        change = tools.group_convergence(flux, flux_old, info)
        print("Count {}\tKeff {}".format(str(count).zfill(3), keff[0]), end="\r")
        converged = (change < EPSILON_POWER) or (count >= MAX_POWER)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    print("\nConvergence:", change)
    return flux[:,:,:]
