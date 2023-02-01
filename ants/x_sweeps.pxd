########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for x-sweeps
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

# cython: profile=True

cdef double[:] scalar_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
            double[:]& external_source, double[:]& boundary, \
            double[:]& spatial_coef, double[:]& angle_weight, int[:]& params, \
            double[:]& cell_width, size_t ex_group_idx)

cdef double[:,:] angular_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
            double[:]& external_source, double[:]& boundary, \
            double[:]& spatial_coef, double[:]& angle_weight, int[:]& params, \
            double[:]& cell_width, size_t ex_group_idx)

cdef double[:,:] time_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_matrix, double[:]& external_source, \
            double[:]& boundary, double[:]& spatial_coef, double[:]& angle_weight, \
            int[:]& params, double temporal_coef, double time_const, size_t ex_group_idx)
