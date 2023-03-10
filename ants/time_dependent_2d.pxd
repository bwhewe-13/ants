########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language = c++
# cython: profile=True

from ants.cytools_2d cimport params2d

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:]& external, double[:]& boundary_x, \
        double[:]& boundary_y, int[:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params2d params)

cdef double[:,:,:,:] hybrid_bdf1(double[:,:]& xs_total_vu, \
        double[:,:]& xs_total_vc, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:]& external_u, double[:]& boundary_x, \
        double[:]& boundary_y, int[:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_xu, double[:]& angle_yu, \
        double[:]& angle_wu, double[:]& angle_xc, double[:]& angle_yc, \
        double[:]& angle_wc, int[:]& index_u, int[:]& index_c, \
        double[:]& factor_u, params2d params_u, params2d params_c)