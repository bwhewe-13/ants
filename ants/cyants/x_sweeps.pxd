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

cdef double[:] scalar_x_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, \
            double[:]& external_source, double[:]& point_source, \
            double[:]& spatial_coef, double[:]& angle_weight, \
            int[:]& params, size_t ex_group_idx) #, size_t ps_group_idx)

cdef double[:,:] angular_x_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, \
            double[:]& external_source, double[:]& point_source, \
            double[:]& spatial_coef, double[:]& angle_weight, \
            int[:]& params, size_t ex_group_idx) #, size_t ps_group_idx)

cdef double[:,:] time_x_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_matrix, \
            double[:]& external_source, \
            double[:]& point_source, double[:]& spatial_coef, 
            double[:]& angle_weight, int[:]& params, \
            double temporal_coef, double time_const, size_t ex_group_idx)
