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


cdef double[:,:,:] multi_group(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef double[:,:,:] source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef double[:,:,:] dynamic_mode_decomp(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef double[:,:,:] variable_source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_u, double[:]& star_coef_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:,:]& external, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        double[:]& edges_g, int[:]& edges_gidx_c, params info)


cdef double[:,:,:,:] _known_source_angular(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef double[:,:,:] _known_source_scalar(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef double[:,:,:,:] _known_source_single(double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, int group, params info)


cdef void _interface_angular(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)


cdef void _interface_scalar(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:]& xs_total, \
        double[:,:,:,:]& source, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info)