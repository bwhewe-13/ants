########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for source_iteration_1d.pyx
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

from ants.cytools_1d cimport params1d

cdef double[:,:,:] multigroup_angular(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& external, double [:]& boundary, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        params1d params)


cdef double[:,:] multigroup_scalar(double[:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& external, double [:]& boundary, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        params1d params)
