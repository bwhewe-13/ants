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

cdef struct params2d:
    # """ Parameter data type for fixed source (time independent) problems)
    
    # :param cells_x: Number of spatial cells in horizontal direction (I)
    # :type cells_x: int

    # :param cells_y: Number of spatial cells in vertical direction (J)
    # :type cells_y: int

    # :param angles: Number of angles (N) (must be even)
    # :type angles: int

    # :param groups: Number of energy groups (G)
    # :type groups: int

    # :param materials: Number of materials per problem (M)
    # :type materials: int

    # :param geometry: Geometry type (1: square, 2: triangle)
    # :type geometry: int

    # :param spatial: Spatial discretization type
    # :type spatial: int

    # :param qdim: Number of external source dimensions (0: [0], 1: [(I x J)], 
    #             2: [(I x J) x G], 3: [(I x J) x N x G])
    # :type qdim: int

    # :param bc_x: [left, right] boundary conditions, (0: vacuum, 1: reflect)
    # :type bc_x: int [2]

    # :param bcdim_x: Number of boundary condition dimensions (0: [2], 
    #             1: [2 x J], 2: [2 x J x G], 3: [2 x J x N x G])
    # :type bcdim_x: int

    # :param bc_y: [bottom, top] boundary conditions, (0: vacuum, 1: reflect)
    # :type bc_y: int [2]

    # :param bcdim_y: Number of boundary condition dimensions (0: [2], 
    #             1: [2 x I], 2: [2 x I x G], 3: [2 x I x N x G])
    # :type bcdim_y: int    

    # :param steps: Number of time steps
    # :type steps: int

    # :param dt: Time step width
    # :type dt: double
    # """
    int cells_x
    int cells_y
    int angles
    int groups
    int materials
    int geometry
    int spatial
    int qdim
    int bc_x [2]
    int bcdim_x
    int bc_y [2]
    int bcdim_y
    int steps
    double dt
    bint angular
    bint adjoint

cdef params2d _to_params2d(dict params_dict)

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
        double[:,:,:] xs_scatter, double[:,:,:] xs_fission, params2d params)

cdef void combine_total_velocity(double[:,:]& xs_total_star, \
            double[:,:]& xs_total, double[:]& velocity, params2d params)

cdef void combine_source_flux(double[:,:,:]& flux_last, double[:]& source_star, \
                double[:]& source, double[:]& velocity, params2d params)

cdef double[:] array_1d_ij(params2d params)

cdef double[:] array_1d_ijng(params2d params)

cdef double[:,:] array_2d_ijg(params2d params)

cdef double[:,:] array_2d_ijn(params2d params)

cdef double[:,:,:] array_3d_ijng(params2d params)

cdef double[:,:,:,:] array_4d_tijng(params2d params)

cdef double[:] edge_y_calc(double[:]& boundary_y, double angle_y, params2d params)

cdef double group_convergence_scalar(double[:,:]& arr1, double[:,:]& arr2, \
                                params2d params)

cdef double group_convergence_angular(double[:,:,:]& arr1, double[:,:,:]& arr2, \
                                    double[:]& weight, params2d params)

cdef double[:] angle_flux(params2d params, bint angular)

cdef void angle_angular_to_scalar(double[:,:]& angular, double[:]& scalar, \
                                double[:]& weight, params2d params)

cdef double angle_convergence_scalar(double[:]& arr1, double[:]& arr2, \
                                params2d params)

cdef double angle_convergence_angular(double[:,:]& arr1, double[:,:]& arr2, \
                                    double[:]& weight, params2d params)

cdef void off_scatter_scalar(double[:,:]& flux, double[:,:]& flux_old, \
                            int[:]& medium_map, double[:,:,:]& xs_matrix, \
                            double[:]& source, params2d params, size_t group)

cdef void off_scatter_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
            int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
            double[:]& weight, params2d params, size_t group)

cdef void fission_source(double[:]& power_source, double[:,:]& flux, \
                    double[:,:,:]& xs_fission, int[:]& medium_map, \
                    params2d params, double[:] keff)

cdef void normalize_flux(double[:,:]& flux, params2d params)

cdef double update_keffective(double[:,:]& flux, double[:,:]& flux_old, \
                            int[:]& medium_map, double[:,:,:]& xs_fission, \
                            params2d params, double keff_old)