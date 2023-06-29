########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for cytools_2d.pyx
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

from ants.parameters cimport params

########################################################################
# Memoryview functions
########################################################################
cdef double[:] array_1d(int dim1)

cdef double[:,:] array_2d(int dim1, int dim2)

cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3)

cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4)

########################################################################
# Convergence functions
########################################################################
cdef double group_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info)

cdef double angle_convergence(double[:,:]& arr1, double[:,:]& arr2, params info)

########################################################################
# Multigroup functions
########################################################################
cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, params info)

cdef void _off_scatter(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:]& off_scatter, params info, int group)

cdef void _source_total(double[:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:]& external, params info)

cdef double[:,:,:] _angular_to_scalar(double[:,:,:,:]& angular_flux,
        double[:]& angle_w, params info)

cdef void _initialize_edge_y(double[:]& known_y, double[:]& boundary_y, \
        double[:]& angle_y, double[:]& angle_x, int nn, params info)

cdef void _initialize_edge_x(double[:]& known_x, double[:]& boundary_x, \
        double[:]& angle_x, double[:]& angle_y, int nn, params info)

########################################################################
# Time Dependent functions
########################################################################
cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, params info)

cdef void _time_source_total(double[:]& source, double[:,:,:]& scalar_flux, \
        double[:,:,:,:]& angular_flux, double[:,:,:]& xs_matrix, \
        double[:]& velocity, int[:,:]& medium_map, double[:]& external, \
        params info)

cdef void _time_source_star(double[:,:,:,:]& angular_flux, double[:]& q_star, \
        double[:]& external, double[:]& velocity, params info)

cdef void boundary_decay(double[:]& boundary_x, double[:]& boundary_y, \
        int step, params info)

########################################################################
# Criticality functions
########################################################################
cdef void _normalize_flux(double[:,:,:]& flux, params info)

cdef void _fission_source(double[:,:,:] flux, double[:,:,:] xs_fission, \
        double[:] source, int[:,:] medium_map, params info, double keff)

cdef double _update_keffective(double[:,:,:] flux_new, double[:,:,:] flux_old, \
        double[:,:,:] xs_fission, int[:,:] medium_map, params info, double keff)