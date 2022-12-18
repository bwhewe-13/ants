########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for power_iteration.pyx
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.cytools cimport params1d

cdef double[:] multigroup(double[:] flux_old, double[:,:] xs_total, \
                        double[:,:,:] xs_scatter, double[:,:,:] xs_fission, 
                        double[:] power_source, int[:] medium_map, \
                        double[:] cell_width, double[:] mu, \
                        double[:] angle_w, params1d params, double[:] keff)

# cdef double[:] adjoint(double[:] flux_guess, double[:,:] xs_total, \
#                         double[:,:,:] xs_scatter, double[:,:,:] xs_fission, 
#                         double[:] power_source, int[:] medium_map, \
#                         double[:] cell_width, double[:] mu, \
#                         double[:] angle_w, params1d params, double[:] keff)