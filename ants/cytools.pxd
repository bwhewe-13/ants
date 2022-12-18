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

# distutils: language=c++
# cython: cdivision=True

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
    bint angular
    bint adjoint

cdef params1d _to_params1d(dict params_dict)

cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission)

cdef double[:] group_flux(params1d params, bint angular)

cdef double group_convergence(double[:]& arr1, double[:]& arr2, \
                    double[:]& weight, params1d params, bint angular)

cdef double[:] angle_flux(params1d params, bint angular)

cdef void angle_angular_to_scalar(double[:]& arr1, double[:]& arr2, \
                        double[:]& weight, params1d params, bint angular)

cdef double angle_convergence(double[:]& arr1, double[:]& arr2, \
                        double[:]& weight, params1d params, bint angular)

cdef void off_scatter_term(double[:]& flux, double[:]& flux_old, \
                    int[:]& medium_map, double[:,:,:]& xs_matrix, \
                    double[:]& source, double[:]& weight, \
                    params1d params, size_t group, bint angular)