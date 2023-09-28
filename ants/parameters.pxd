########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for parameters.pyx
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


cdef struct params:
    # """ Parameter data type for fixed source (time independent) problems)

    # :param cells_x: Number of spatial cells in x direction (I)
    # :type cells_x: int

    # :param cells_y: Number of spatial cells in y direction (J)
    # :type cells_y: int

    # :param angles: Number of angles (N) (must be even)
    # :type angles: int

    # :param groups: Number of energy groups (G)
    # :type groups: int

    # :param geometry: Geometry type (1: slab, 2: sphere)
    # :type geometry: int

    # :param spatial: Spatial discretization type
    # :type spatial: int

    # :param qdim: Number of external source dimensions (1: [I],
    #             2: [I x G], 3: [I x N x G])
    # :type qdim: int

    # :param bc: [left, right] boundary conditions (0: vacuum, 1: reflect)
    # :type bc: int [2]

    # :param bcdim: Number of boundary condition dimensions (1: [2],
    #             2: [2 x G], 3: [2 x N x G])
    # :type bcdim: int

    # :param steps: Number of time steps
    # :type steps: int

    # :param dt: Time step width
    # :type dt: double

    # :param bcdecay_x: Type of boundary source decay
    # :type bcdecay_x: int

    # :param bcdecay_y: Type of boundary source decay
    # :type bcdecay_y: int

    # """
    # Collect Spatial cells, angles, energy groups
    int cells_x
    int cells_y
    int angles
    int groups
    int materials
    # Geometry type (slab, sphere)
    int geometry
    # Spatial discretization type
    int spatial
    # External source
    int qdim
    # Boundary parameters
    int bc_x [2]
    int bcdim_x
    int bc_y [2]
    int bcdim_y
    # Time dependent parameters
    int steps
    double dt
    int bcdecay_x
    int bcdecay_y
    # Angular flux option
    bint angular
    # Adjoint option
    bint adjoint
    # Flux at cell edges or centers
    int edges


cdef params _to_params(dict pydic)

########################################################################
# One-dimensional functions
########################################################################
cdef int _check_fixed1d_source_iteration(params info, int xs_length) except -1

cdef int _check_nearby1d_fixed_source(params info, int xs_length) except -1

cdef int _check_nearby1d_criticality(params info) except -1

cdef int _check_timed1d(params info, int xs_length) except -1

cdef int _check_critical1d_power_iteration(params info) except -1

cdef int _check_critical1d_nearby_power(params info) except -1

cdef int _check_hybrid1d_uncollided(params info, int xs_length) except -1

cdef int _check_hybrid1d_collided(params info, int xs_length) except -1

########################################################################
# Two-dimensional functions
########################################################################
cdef int _check_fixed2d_source_iteration(params info, int xs_length) except -1

cdef int _check_nearby2d_fixed_source(params info, int xs_length) except -1

cdef int _check_nearby2d_criticality(params info) except -1

cdef int _check_timed2d_backward_euler(params info, int xs_length) except -1

cdef int _check_critical2d_power_iteration(params info) except -1

cdef int _check_critical2d_nearby_power(params info) except -1

cdef int _check_hybrid2d_bdf1_uncollided(params info, int xs_length) except -1

cdef int _check_hybrid2d_bdf1_collided(params info, int xs_length) except -1