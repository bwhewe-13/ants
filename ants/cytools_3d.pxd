########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for cytools_3d.pyx
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from ants.parameters cimport params


################################################################################
# Memoryview functions
################################################################################
cdef double[:] array_1d(int dim1)

cdef double[:,:] array_2d(int dim1, int dim2)

cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3)

cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4)

cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5)

cdef double[:,:,:,:,:,:] array_6d(int dim1, int dim2, int dim3, int dim4, \
        int dim5, int dim6)

################################################################################
# Convergence functions
################################################################################
cdef double group_convergence(double[:,:,:,:]& arr1, double[:,:,:,:]& arr2, \
        params info)

cdef double angle_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info) noexcept nogil

################################################################################
# Multigroup functions
################################################################################
cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
        double[:,:,:]& mat3, params info)

cdef double[:,:,:] _fission_matrix(object fission, object chi)

cdef void _dmd_subtraction(double[:,:,:,:,:]& y_minus, \
        double[:,:,:,:,:]& y_plus, double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, int snapshot, params info)

cdef void _off_scatter(double[:,:,:,:]& flux, double[:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:,:]& off_scatter, params info, int group)

cdef void _off_scatter_jacobi(double[:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:,:,:]& off_scatter_all, params info, int group) noexcept nogil

cdef void _source_total(double[:,:,:,:,:]& source, double[:,:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:,:]& medium_map, \
        double[:,:,:,:,:]& external, params info, int[:,:] q_index=*)

cdef int[:,:] _source_columns(int columns, int[:,:] q_index, params info)

################################################################################
# Anisotropic scattering functions
################################################################################
cdef void _xs_matrix_aniso(double[:,:,:,:]& mat1, double[:,:,:,:]& mat2, \
        double[:,:,:]& mat3, params info)

cdef void _off_scatter_moments(double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& flux_old, int[:,:,:]& medium_map, \
        double[:,:,:,:]& xs_matrix, double[:,:,:,:]& off_scatter, \
        params info, int group)

cdef void _off_scatter_jacobi_moments(double[:,:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:,:,:,:]& off_scatter_all, params info, int group) noexcept nogil

cdef void _source_total_aniso(double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& flux, double[:,:,:,:]& xs_matrix, \
        int[:,:,:]& medium_map, double[:,:,:,:,:]& external, \
        double[:,:]& P_source, params info, int[:,:] q_index=*)

cdef void _dmd_subtraction_moments(double[:,:,:,:,:]& y_minus, \
        double[:,:,:,:,:]& y_plus, double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& flux_old, int snapshot, params info)

################################################################################
# Boundary functions
################################################################################
cdef void initialize_known(double[:,:]& known, double[:,:,:]& boundary, \
        double[:,:,:,:]& reflected, double angle, int angle_idx, \
        int bc_lower, int bc_upper) noexcept nogil

cdef void update_reflector(double[:,:]& known_x, double[:,:,:,:]& reflected_x, \
        double[:,:]& known_y, double[:,:,:,:]& reflected_y, \
        double[:,:]& known_z, double[:,:,:,:]& reflected_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        int angle, params info) noexcept nogil
