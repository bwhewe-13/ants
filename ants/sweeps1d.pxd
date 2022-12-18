########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for sweeps1d.pyx
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

# from ants.parameters cimport params1d
from ants.cytools cimport params1d

from libcpp cimport float

cdef double sweep(double flux, double flux_old, double xs_total, \
                double xs_matrix, double off_scatter, double source, \
                double mu, double angle_w, double cell_width, \
                double[:] known_edge, float xs1_const, float xs2_const, \
                params1d params, bint angular)

cdef double find_boundary(double prev_edge, double mu, double[:] boundary, \
                            params1d params)
