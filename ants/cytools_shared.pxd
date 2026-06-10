################################################################################
#                            ___    _   _____________
#                           /   |  / | / /_  __/ ___/
#                          / /| | /  |/ / / /  \__ \
#                         / ___ |/ /|  / / /  ___/ /
#                        /_/  |_/_/ |_/ /_/  /____/
#
# Declaration file for cytools_shared.pyx — fused-type shared implementations.
# Import this from cytools_1d.pyx and cytools_2d.pyx.
#
################################################################################

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
# Memoryview allocation functions
################################################################################

cdef double[:] array_1d(int dim1)

cdef int[:] int_array_1d(int dim1)

cdef double[:,:] array_2d(int dim1, int dim2)

cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3)

cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4)

cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, int dim5)

cdef float[:,:,:,:] farray_4d(int dim1, int dim2, int dim3, int dim4)

cdef float[:,:,:,:,:] farray_5d(int dim1, int dim2, int dim3, int dim4, int dim5)

################################################################################
# Fused types
################################################################################

ctypedef fused scalar_flux_nd:
    double[:,:]
    double[:,:,:]

ctypedef fused spatial_nd:
    double[:]
    double[:,:]

ctypedef fused medium_map_nd:
    int[:]
    int[:,:]

################################################################################
# Function declarations
################################################################################

cdef double group_convergence(scalar_flux_nd arr1, scalar_flux_nd arr2,
                               params info)

cdef double angle_convergence(spatial_nd arr1, spatial_nd arr2, params info)

cdef void _normalize_flux(scalar_flux_nd flux, params info)

cdef double _update_keffective(scalar_flux_nd flux_new, scalar_flux_nd flux_old,
                                double[:,:,:] xs_fission,
                                medium_map_nd medium_map,
                                params info, double keff)

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity,
                           double constant, params info)

cdef double[:,:,:] _fission_matrix(object fission, object chi)
