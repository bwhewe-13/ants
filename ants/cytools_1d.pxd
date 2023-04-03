########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Header file for cytools.pyx
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

cdef struct params1d:
    # """ Parameter data type for fixed source (time independent) problems)
    
    # :param cells: Number of spatial cells (I)
    # :type cells: int

    # :param angles: Number of angles (N) (must be even)
    # :type angles: int

    # :param groups: Number of energy groups (G)
    # :type groups: int

    # :param materials: Number of materials per problem (M)
    # :type materials: int

    # :param geometry: Geometry type (1: slab, 2: sphere)
    # :type geometry: int

    # :param spatial: Spatial discretization type
    # :type spatial: int

    # :param qdim: Number of external source dimensions (0: [0], 1: [I], 
    #             2: [I x G], 3: [I x N x G])
    # :type qdim: int

    # :param bc: [left, right] boundary conditions (0: vacuum, 1: reflect)
    # :type bc: int [2]

    # :param bcdim: Number of boundary condition dimensions (0: [2], 
    #             1: [2 x G], 2: [2 x N x G])
    # :type bcdim: int

    # :param steps: Number of time steps
    # :type steps: int

    # :param dt: Time step width
    # :type dt: double

    # :param bcdecay: Type of boundary source decay
    # :type bcdecay: int
    # """
    int cells
    int angles
    int groups
    int materials
    int geometry
    int spatial
    int qdim
    int bc [2]
    int bcdim
    int steps
    double dt
    bint angular
    bint adjoint
    int bcdecay
    int edges

cdef params1d _to_params1d(dict params_dict)

cdef void combine_self_scattering(double[:,:,:]& xs_matrix, \
                    double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
                    params1d params)

cdef void combine_total_velocity(double[:,:]& xs_total_star, \
            double[:,:]& xs_total, double[:]& velocity, params1d params)

cdef void combine_source_flux(double[:,:,:]& flux_last, double[:]& source_star, \
                double[:]& source, double[:]& velocity, params1d params)

cdef double[:] array_1d(int dim1)

cdef double[:,:] array_2d(int dim1, int dim2)

cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3)

cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4)

cdef double group_convergence_scalar(double[:,:]& arr1, double[:,:]& arr2, \
                                params1d params)

cdef double group_convergence_angular(double[:,:,:]& arr1, \
                double[:,:,:]& arr2, double[:]& weight, params1d params)

cdef double[:] angle_flux(params1d params, bint angular)

cdef void angle_angular_to_scalar(double[:,:]& angular, double[:]& scalar, \
                                  double[:]& weight, params1d params)

cdef double angle_convergence_scalar(double[:]& arr1, double[:]& arr2, \
                                params1d params)

cdef double angle_convergence_angular(double[:,:]& arr1, double[:,:]& arr2, \
                                double[:]& weight, params1d params)

cdef void off_scatter_scalar(double[:,:]& flux, double[:,:]& flux_old, \
            int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
            params1d params, size_t group)

cdef void off_scatter_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
            int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
            double[:]& weight, params1d params, size_t group)

cdef void fission_source(double[:,:] flux, double[:,:,:] xs_fission, \
                    double[:] power_source, int[:] medium_map, \
                    params1d params, double keff)

cdef void normalize_flux(double[:,:]& flux, params1d params)

cdef double update_keffective(double[:,:] flux, double[:,:] flux_old, \
                            double[:,:,:] xs_fission, int[:] medium_map, \
                            params1d params, double keff_old)

cdef void nearby_fission_source(double[:,:]& flux, double[:,:,:]& xs_fission, \
                        double[:]& power_source, double[:]& nearby_source, \
                        int[:]& medium_map, params1d params, double keff)

cdef double nearby_keffective(double[:,:]& flux, double rate, params1d params)

cdef void calculate_source_c(double[:,:]& scalar_flux_u, double[:,:,:]& xs_scatter_u, \
                double[:]& source_c, int[:]& medium_map, int[:]& index_ch, \
                params1d params_u, params1d params_c)

cdef void calculate_source_t(double[:,:]& flux_u, double[:,:]& flux_c, \
                double[:,:,:]& xs_scatter_u, double[:]& source_t, \
                int[:]& medium_map, int[:]& index_u, double[:]& factor_u, \
                params1d params_u, params1d params_c)

cdef void calculate_source_star(double[:,:,:]& flux_last, double[:]& source_star, \
        double[:]& source_t, double[:]& source_u, double[:]& velocity, params1d params)

cdef void big_to_small(double[:,:]& flux_u, double[:]& flux_c, \
                int[:]& index_c, params1d params_u, params1d params_c)

cdef double[:,:] small_to_big(double[:,:]& flux_c, int[:]& index_u, \
            double[:]& factor_u, params1d params_u, params1d params_c)

cdef void boundary_decay(double[:]& boundary, size_t step, params1d params)