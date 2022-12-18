########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for cytools_crit.pyx
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.cytools cimport params1d

from libc.math cimport sqrt, pow

cdef void fission_source(double[:] power_source, double[:] flux, \
                    double[:,:,:] xs_fission, int[:] medium_map, \
                    params1d params, double[:] keff)

cdef void normalize_flux(double[:]& flux, params1d params)

cdef double update_keffective(double[:] flux, double[:] flux_old, \
                            int[:] medium_map, double[:,:,:] xs_fission, \
                            params1d params, double keff_old)