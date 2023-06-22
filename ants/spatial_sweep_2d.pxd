########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for spatial_sweep_2d.pyx
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


cdef void discrete_ordinates(double[:,:]& flux, double[:,:]& flux_old, 
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info)


cdef void _known_sweep(double[:,:,:]& flux, double[:]& xs_total, \
        double[:]& source, double[:]& boundary_x, double[:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, params info)