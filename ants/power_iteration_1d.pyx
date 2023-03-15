########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Power iteration for one-dimensional multigroup neutron transport 
# criticality problems.
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

from ants cimport source_iteration_1d as si
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d
from ants.constants import *


cdef double[:,:] multigroup(double[:,:]& flux_guess, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:]& power_source, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params1d params, \
        double[:]& keff):
    # Initialize flux
    flux = tools.array_2d_ig(params)
    flux_old = flux_guess.copy()
    # Vacuum boundaries
    cdef double[2] boundary = [0.0, 0.0]
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        tools.fission_source(flux_old, xs_fission, power_source, \
                             medium_map, params, keff[0])
        flux = si.multigroup_scalar(flux_old, xs_total, xs_scatter, \
                                    power_source, boundary, medium_map, \
                                    delta_x, angle_x, angle_w, params)
        keff[0] = tools.update_keffective(flux, flux_old, xs_fission, \
                                          medium_map, params, keff[0])
        tools.normalize_flux(flux, params)
        change = tools.group_convergence_scalar(flux, flux_old, params)
        # print("Power Iteration {}\n{}\nChange {} Keff {}".format(count, \
        #         "="*35, change, keff[0]))
        print("Count {}\tKeff {}".format(str(count).zfill(3), keff[0]), end="\r")
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    print()
    return flux[:,:]


cdef double[:,:] nearby(double[:,:]& flux_guess, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, 
        double[:]& power_source, double[:]& nearby_source, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params1d params, double[:]& keff):
    # Initialize flux
    flux = tools.array_2d_ig(params)
    flux_old = flux_guess.copy()
    # Vacuum boundaries
    cdef double[2] boundary = [0.0, 0.0]
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        tools.nearby_fission_source(flux_old, xs_fission, power_source, \
                                    nearby_source, medium_map, params, keff[0])
        flux = si.multigroup_scalar(flux_old, xs_total, xs_scatter, \
                                    power_source, boundary, medium_map, \
                                    delta_x, angle_x, angle_w, params)
        keff[0] = tools.update_keffective(flux, flux_old, xs_fission, \
                                          medium_map, params, keff[0])
        tools.normalize_flux(flux, params)
        change = tools.group_convergence_scalar(flux, flux_old, params)
        # print("Power Iteration {}\n{}\nChange {} Keff {}".format(count, \
        #         "="*35, change, keff[0]))
        print("Count {}\tKeff {}".format(str(count).zfill(3), keff[0]), end="\r")
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    print()
    return flux[:,:]
