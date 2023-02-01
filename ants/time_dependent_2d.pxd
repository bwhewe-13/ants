########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

from ants.cytools_2d cimport params2d

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
                        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
                        double[:]& velocity, double[:]& source, \
                        double[:]& boundary_x, double[:]& boundary_y, \
                        int[:]& medium_map, \
                        double[:]& delta_x, double[:]& delta_y, \
                        double[:]& angle_x, double[:]& angle_y, \
                        double[:]& angle_w, params2d params)
