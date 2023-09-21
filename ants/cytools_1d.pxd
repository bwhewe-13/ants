########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for cytools_1d.pyx
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
cdef double group_convergence(double[:,:]& arr1, double[:,:]& arr2, params info)

cdef double angle_convergence(double[:]& arr1, double[:]& arr2, params info)

########################################################################
# Material Interface functions
########################################################################
cdef int[:] _material_index(int[:] medium_map, params info)

########################################################################
# Multigroup functions
########################################################################
cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
    double[:,:,:]& mat3, params info)

cdef void _off_scatter(double[:,:]& flux, double[:,:]& flux_old, \
        int[:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:]& off_scatter, params info, int group)

cdef void _source_total(double[:]& source, double[:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:]& medium_map, \
        double[:]& external, params info)

cdef void _angular_to_scalar(double[:,:,:]& angular_flux, \
        double[:,:]& scalar_flux, double[:]& angle_w, params info)

########################################################################
# Time Dependent functions
########################################################################
cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info)

cdef void _time_source_star_bdf1(double[:,:,:]& flux, double[:]& q_star, \
        double[:]& external, double[:]& velocity, params info)

cdef void _time_source_star_bdf2(double[:,:,:]& flux_1, \
        double[:,:,:]& flux_2, double[:]& q_star, double[:]& external, \
        double[:]& velocity, params info)

cdef void _time_source_total_bdf1(double[:]& source, double[:,:]& scalar_flux, \
        double[:,:,:]& angular_flux, double[:,:,:]& xs_matrix, \
        double[:]& velocity, int[:]& medium_map, double[:]& external, \
        params info)

cdef void _time_source_total_bdf2(double[:]& source, double[:,:]& scalar_flux, \
        double[:,:,:]& angular_flux_1, double[:,:,:]& angular_flux_2, \
        double[:,:,:]& xs_matrix, double[:]& velocity, int[:]& medium_map, \
        double[:]& external, params info)

cdef void boundary_decay(double[:]& boundary_x, int step, params info)

########################################################################
# Criticality functions
########################################################################
cdef void _normalize_flux(double[:,:]& flux, params info)

cdef void _fission_source(double[:,:] flux, double[:,:,:] xs_fission, \
        double[:] source, int[:] medium_map, params info, double keff)

cdef double _update_keffective(double[:,:] flux_new, double[:,:] flux_old, \
        double[:,:,:] xs_fission, int[:] medium_map, params info, double keff)

cdef void _source_total_critical(double[:]& source, double[:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:]& medium_map, double keff, params info)

########################################################################
# Nearby Problems Criticality functions
########################################################################
cdef void _nearby_fission_source(double[:,:]& flux, double[:,:,:]& xs_fission, \
        double[:]& source, double[:]& residual, int[:]& medium_map, \
        params info, double keff)

cdef double _nearby_keffective(double[:,:]& flux, double rate, params info)

########################################################################
# Hybrid Method Time Dependent Problems
########################################################################
cdef void _hybrid_source_collided(double[:,:]& flux, double[:,:,:]& xs_scatter, \
        double[:]& source_c, int[:]& medium_map, int[:]& index_c, \
        params info_u, params info_c)

cdef void _hybrid_source_total(double[:,:]& flux_t, double[:,:]& flux_u, \
        double[:,:,:]& xs_matrix, double[:]& source, int[:]& medium_map, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c)

cdef void _expand_hybrid_source(double[:,:]& flux_t, double[:,:]& flux_c, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c)