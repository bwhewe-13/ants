########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for source_iteration_1d.pyx
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

from ants.cytools_1d cimport params1d

cdef double[:,:] multigroup_scalar(double[:,:]& flux_guess, \
                        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
                        double[:]& source, double [:]& boundary, \
                        int[:]& medium_map, double[:]& delta_x, \
                        double[:]& angle_x, double[:]& angle_w, \
                        params1d params)

cdef double[:,:,:] multigroup_angular(double[:,:,:]& flux_guess, \
                        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
                        double[:]& source, double [:]& boundary, \
                        int[:]& medium_map, double[:]& delta_x, \
                        double[:]& angle_x, double[:]& angle_w, \
                        params1d params)

# cdef void adjoint(double[:] flux, double[:] flux_one_group, double[:] xs_total, \
#             double[:] xs_scatter, double[:] off_scatter, double[:] source, \
#             double[:] boundary, int[:] medium_map, double[:] cell_width, \
#             double[:] mu, double[:] angle_w, params1d params, bint angular)