########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for xy-sweeps
#
########################################################################

cdef double[:] scalar_quad_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
        double[:]& xs_total, double[:]& xs_matrix, double[:]& off_scatter, \
        double[:]& external_source, double[:]& boundary, double[:]& mu, \
        double[:]& eta, double[:]& angle_weight, int[:]& params, \
        double[:]& delta_x, double[:]& delta_y, size_t gg_idx)

cdef double[:] scalar_single_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
        double[:]& xs_total, double[:]& xs_matrix, double[:]& off_scatter, \
        double[:]& external_source, double[:]& boundary, double[:]& mu, \
        double[:]& eta, double[:]& angle_weight, int[:]& params, \
        double[:]& delta_x, double[:]& delta_y, size_t gg_idx)

# cdef double[:,:] angular_x_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
#             double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
#             double[:]& external_source, double[:]& boundary, \
#             double[:]& spatial_coef, double[:]& angle_weight, \
#             int[:]& params, double[:]& cell_width, size_t gg_idx)

# cdef double[:,:] time_x_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
#             double[:]& xs_total, double[:]& xs_matrix, \
#             double[:]& external_source, \
#             double[:]& boundary, double[:]& spatial_coef, 
#             double[:]& angle_weight, int[:]& params, \
#             double temporal_coef, double time_const, size_t gg_idx)
