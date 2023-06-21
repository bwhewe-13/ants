########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for multi_group_2d.pyx
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

from ants.parameters cimport params


cdef double[:,:,:] source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& external, \
        double [:]& boundary_x, double [:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)