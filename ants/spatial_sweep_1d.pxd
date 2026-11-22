########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for spatial_sweep_1d.pyx
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


cdef void discrete_ordinates(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:,:]& external, int[:]& q_col, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info)


cdef void discrete_ordinates_nogil(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:,:]& external, int[:]& q_col, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info, double[:,:]& thread_flux, \
        double[:]& edge_out, double[:]& reflector, \
        double[:]& half_angle) noexcept nogil


cdef void _known_sweep(double[:,:]& flux, double[:]& xs_total, \
        double[:]& zero, double[:,:]& source, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info)


cdef void discrete_ordinates_aniso(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:,:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:]& external, int[:]& q_col, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        params info)


cdef void discrete_ordinates_aniso_nogil(double[:,:]& flux, \
        double[:,:]& flux_old, double[:]& xs_total, double[:,:]& xs_scatter, \
        double[:,:]& off_scatter, double[:,:]& external, int[:]& q_col, \
        double[:,:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info, double[:,:]& thread_psi, \
        double[:,:]& thread_source, double[:,:,:]& thread_flux, \
        double[:]& edge_out, double[:]& reflector, double[:]& half_angle, \
        double[:]& P_start, double[:]& zero, \
        double[:]& zero_xs) noexcept nogil
