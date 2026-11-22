########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for spatial_sweep_3d.pyx
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


cdef void discrete_ordinates(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:]& angle_w, params info)


cdef void discrete_ordinates_nogil(double[:,:,:]& flux, \
        double[:,:,:]& flux_old, double[:]& xs_total, double[:]& xs_scatter, \
        double[:,:,:]& off_scatter, double[:,:,:,:]& external, int[:]& q_col, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info, double[:,:,:,:]& thread_flux, \
        double[:,:,:,:]& reflected_x, double[:,:,:,:]& reflected_y, \
        double[:,:,:,:]& reflected_z, double[:,:,:]& known_x_work, \
        double[:,:,:]& known_y_work, double[:,:,:]& known_z_work) noexcept nogil


cdef void _known_center_sweep(double[:,:,:,:]& flux, double[:]& xs_total, \
        double[:,:,:]& zero_3d, double[:,:,:,:]& source, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        params info)


cdef void _known_interface_sweep(double[:,:,:,:]& flux_edge_x, \
        double[:,:,:,:]& flux_edge_y, double[:,:,:,:]& flux_edge_z, \
        double[:]& xs_total, double[:,:,:,:]& source, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& boundary_z, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:]& angle_w, params info)


cdef void discrete_ordinates_aniso(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, double[:]& xs_total, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:,:]& P_source, double[:,:]& P_weights, \
        params info)


cdef void discrete_ordinates_aniso_nogil(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, double[:]& xs_total, \
        double[:,:]& xs_scatter, double[:,:,:,:]& off_scatter, \
        double[:,:,:,:]& external, int[:]& q_col, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& boundary_z, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double[:,:]& P_source, double[:,:]& P_weights, \
        params info, double[:,:,:,:]& thread_psi, \
        double[:,:,:,:]& thread_source, double[:,:,:,:,:]& thread_flux, \
        double[:,:,:]& zero_3d, double[:]& zero_xs, \
        double[:,:,:,:]& reflected_x, double[:,:,:,:]& reflected_y, \
        double[:,:,:,:]& reflected_z, double[:,:,:]& known_x_work, \
        double[:,:,:]& known_y_work, double[:,:,:]& known_z_work) noexcept nogil
