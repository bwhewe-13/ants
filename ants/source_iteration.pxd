########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for source_iteration.pyx
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.cytools cimport params1d

# cdef double[:] multigroup(double[:] flux_guess, double[:,:] xs_total, \
#                         double[:,:,:] xs_matrix, double[:] external_source, \
#                         double [:] boundary, int[:] medium_map, \
#                         double[:] cell_width, double[:] mu, \
#                         double[:] angle_w, params1d params, bint angular)

cdef double[:] multigroup(double[:] flux_guess, double[:,:] xs_total, \
                        double[:,:,:] xs_matrix, double[:] external_source, \
                        double [:] boundary, int[:] medium_map, \
                        double[:] cell_width, double[:] mu, \
                        double[:] angle_w, params1d params, bint angular)
                        # void (*func)(double[:], double[:], double[:], double[:], \
                        # double[:], double[:], double[:], int[:], \
                        # double[:], double[:], double[:], params1d, bint), \
                        # bint angular)

# cdef void ordinates(double[:] flux, double[:] flux_one_group, double[:] xs_total, \
#                         double[:] xs_scatter, double[:] off_scatter, \
#                         double[:] source, double[:] boundary, int[:] medium_map, \
#                         double[:] cell_width, double[:] mu, double[:] angle_w, \
#                         params1d params, bint angular)

# cdef void adjoint(double[:] flux, double[:] flux_one_group, double[:] xs_total, \
#             double[:] xs_scatter, double[:] off_scatter, double[:] source, \
#             double[:] boundary, int[:] medium_map, double[:] cell_width, \
#             double[:] mu, double[:] angle_w, params1d params, bint angular)