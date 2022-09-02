########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
#
########################################################################

cdef void power_iteration_source(double[:] power_source, double[:,:] flux, \
                                int[:] medium_map, double[:,:,:] xs_fission)

cdef void mms_power_iteration_source(double[:] power_source, double[:,:] flux, \
                        int[:] medium_map, double[:,:,:] xs_fission, int angles)

cdef void add_manufactured_source(double[:] power_source, double[:] mms_source)

cdef double normalize_flux(double[:,:] flux)

cdef void divide_by_keff(double[:,:] flux, double keff)

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission)

cdef void off_scatter_source(double[:,:]& flux, double[:,:]& flux_old, \
                             int[:]& medium_map, double[:,:,:]& xs_matrix, \
                             double[:]& source, int group)

cdef void off_scatter_source_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
                        int[:]& medium_map, double[:,:,:]& xs_matrix, \
                        double[:]& source, size_t group, double[:]& weight)

cdef double scalar_convergence(double [:,:]& arr1, double [:,:]& arr2)

cdef double group_scalar_convergence(double [:]& arr1, double [:]& arr2)

cdef double angular_convergence(double [:,:,:]& arr1, double [:,:,:]& arr2, \
                                double[:]& weight)

cdef double group_angular_convergence(double[:,:]& arr1, double [:,:]& arr2, \
                                double [:]& weight)

cdef void angular_to_scalar(double[:,:]& scalar_flux, \
                        double[:,:,:]& angular_flux, double[:]& weight)

cdef void group_angular_to_scalar(double[:]& scalar_flux, \
                    double[:,:]& angular_flux, double[:]& angle_weight)

cdef void time_coef(double[:]& temporal_coef, double[:]& velocity, \
                     double time_step_size)