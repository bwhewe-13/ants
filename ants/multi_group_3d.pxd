########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for multi_group_3d.pyx
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from ants.parameters cimport params


cdef double[:,:,:,:] source_iteration(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        int[:,:] q_index=*)


cdef double[:,:,:,:] dynamic_mode_decomp(double[:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info, \
        int[:,:] q_index=*)


cdef double[:,:,:,:,:] multi_group_moments(double[:,:,:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, double[:,:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, int[:,:] q_index=*)


cdef void _known_source_angular(double[:,:,:,:,:]& angular_flux, \
        double[:,:]& xs_total, double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        double[:,:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        params info)


cdef void _interface_flux(double[:,:,:,:,:]& flux_edge_x, \
        double[:,:,:,:,:]& flux_edge_y, double[:,:,:,:,:]& flux_edge_z, \
        double[:,:]& xs_total, double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        double[:,:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info)
