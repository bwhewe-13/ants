########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for one-Dimensional Time Dependent Source Multigroup 
# Neutron Transport Problems
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

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:]& source, double[:]& boundary, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params1d params)

cdef double[:,:,:,:] hybrid_bdf1(double[:,:]& xs_total_vu, \
        double[:,:]& xs_total_vc, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:]& source_u, double[:]& boundary, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_wu, double[:]& angle_xc, double[:]& angle_wc, \
        int[:]& index_u, int[:]& index_c, double[:]& factor_u, \
        params1d params_u, params1d params_c)