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

cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5)

########################################################################
# Convergence functions
########################################################################
cdef double group_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info)

cdef double angle_convergence(double[:,:]& arr1, double[:,:]& arr2, params info)

########################################################################
# Multigroup functions
########################################################################
cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
    double[:,:,:]& mat3, params info)

cdef void _dmd_subtraction(double[:,:,:,:]& y_minus, double[:,:,:,:]& y_plus, \
        double[:,:,:]& flux, double[:,:,:]& flux_old, int kk, params info)

cdef void _off_scatter(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:]& off_scatter, params info, int group)

cdef void _source_total(double[:,:,:,:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:,:,:,:]& external, params info)

cdef void _angular_to_scalar(double[:,:,:,:]& angular_flux, \
        double[:,:,:]& scalar_flux, double[:]& angle_w, params info)

cdef void _angular_edge_to_scalar(double[:,:,:,:]& psi_x, double[:,:,:,:]& psi_y, \
        double[:,:,:]& scalar_flux, double[:]& angle_w, params info)

cdef void initialize_known_y(double[:]& known_y, double[:,:]& boundary_y, \
        double[:,:,:]& reflected_y, double[:]& angle_y, int angle, params info)

cdef void initialize_known_x(double[:]& known_x, double[:,:]& boundary_x, \
        double[:,:,:]& reflected_x, double[:]& angle_x, int angle, params info)

cdef void update_reflector(double[:]& known_x, double[:,:,:]& reflected_x, \
        double[:]& angle_x, double[:]& known_y, double[:,:,:]& reflected_y, \
        double[:]& angle_y, int angle, params info)

########################################################################
# Time Dependent functions
########################################################################
cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info)

cdef void _time_source_star_bdf1(double[:,:,:,:]& flux, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& external, \
        double[:]& velocity, params info)

cdef void _time_source_total_bdf1(double[:,:,:]& scalar, double[:,:,:,:]& angular, \
        double[:,:,:]& xs_matrix, double[:]& velocity, double[:,:,:,:]& qstar, \
        double[:,:,:,:]& external, int[:,:]& medium_map, params info)

cdef void _time_source_star_cn(double[:,:,:,:]& psi_x, double[:,:,:,:]& psi_y, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external_prev, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double constant, params info)

cdef void _time_source_star_bdf2(double[:,:,:,:]& flux_1, \
        double[:,:,:,:]& flux_2, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external, double[:]& velocity, params info)

cdef void _time_source_total_bdf2(double[:,:,:]& scalar, \
        double[:,:,:,:]& angular_1, double[:,:,:,:]& angular_2, \
        double[:,:,:]& xs_matrix, double[:]& velocity, double[:,:,:,:]& qstar, \
        double[:,:,:,:]& external, int[:,:]& medium_map, params info)

cdef void _time_source_star_tr_bdf2(double[:,:,:,:]& psi_x, \
        double[:,:,:,:]& psi_y, double[:,:,:,:]& flux_2, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& external, \
        double[:]& velocity, double gamma, params info)

cdef void _time_right_side(double[:,:,:,:]& q_star, double[:,:,:]& flux, \
        double[:,:,:]& xs_scatter, int[:,:]& medium_map, params info)

########################################################################
# Criticality functions
########################################################################
cdef void _normalize_flux(double[:,:,:]& flux, params info)

cdef void _fission_source(double[:,:,:]& flux, double[:,:,:]& xs_fission, \
        double[:,:,:,:]& source, int[:,:]& medium_map, params info, \
        double keff)

cdef double _update_keffective(double[:,:,:] flux_new, double[:,:,:] flux_old, \
        double[:,:,:] xs_fission, int[:,:] medium_map, params info, double keff)

cdef void _source_total_critical(double[:,:,:,:]& source, \
        double[:,:,:]& flux, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:,:]& medium_map, double keff, \
        params info)

########################################################################
# Nearby Problems Criticality functions
########################################################################
cdef void _nearby_fission_source(double[:,:,:]& flux, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& source, \
        double[:,:,:,:]& residual, int[:,:]& medium_map, params info, \
        double keff)

cdef double _nearby_keffective(double[:,:,:]& flux, double rate, params info)

########################################################################
# Hybrid Method Time Dependent Problems
########################################################################
cdef void _hybrid_source_collided(double[:,:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:,:]& source_c, \
        int[:,:]& medium_map, int[:]& coarse_idx, params info_u)

cdef void _hybrid_source_total(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        double[:,:,:]& xs_matrix, double[:,:,:,:]& source, int[:,:]& medium_map, \
        int[:]& coarse_idx, double[:]& factor_u, params info_u, params info_c)

# cdef void _hybrid_source_total(double[:,:,:]& flux_t, double[:,:,:]& flux_u, \
#         double[:,:,:]& xs_matrix, double[:,:,:,:]& source, int[:,:]& medium_map, \
#         int[:]& index_u, double[:]& factor_u, params info_u, params info_c)

# cdef void _expand_hybrid_source(double[:,:,:]& flux_t, double[:,:,:]& flux_c, \
#         int[:]& index_u, double[:]& factor_u, params info_u, params info_c)