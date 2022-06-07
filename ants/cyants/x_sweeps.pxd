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

cdef double[:] scalar_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, \
            double[:,:]& external_source, int[:]& point_source_loc, \
            double[:,:]& point_source, double[:]& spatial_coef, \
            double[:]& angle_weight, int spatial, int boundary)

cdef double[:,:] angular_sweep(double[:,:] angular_flux_old, int[:]& medium_map, \
            double[:]& xs_total, double[:]& xs_scatter, \
            double[:,:]& external_source, int[:]& point_source_loc, \
            double[:,:]& point_source, double[:]& spatial_coef, \
            double[:]& angle_weight, int spatial, int boundary)
