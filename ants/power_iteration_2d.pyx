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

from ants cimport source_iteration_2d as si
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d
from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE

cdef double[:,:] multigroup(double[:,:]& flux_guess, double[:,:]& xs_total, \
                    double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, 
                    double[:]& power_source, int[:]& medium_map, \
                    double[:]& delta_x, double[:]& delta_y, \
                    double[:]& angle_x, double[:]& angle_y, \
                    double[:]& angle_w, params2d params, double[:]& keff):
    # Initialize flux
    flux = tools.array_2d_ijg(params)
    flux_old = flux_guess.copy()
    # Vacuum boundaries
    cdef double[2] boundary_x = [0.0, 0.0]
    cdef double[2] boundary_y = [0.0, 0.0]
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        tools.fission_source(power_source, flux_old, xs_fission, \
                            medium_map, params, keff)
        print("here")
        flux = si.multigroup_scalar(flux, xs_total, xs_scatter, power_source, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, params)
        keff[0] = tools.update_keffective(flux, flux_old, medium_map, \
                                          xs_fission, params, keff[0])
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
