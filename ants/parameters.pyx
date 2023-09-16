########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Creating the params struct and doing type checks
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


cdef params _to_params(dict pydic):
    # Initialize params struct
    cdef params info
    # Collect Spatial cells, angles, energy groups
    info.cells_x = pydic.get("cells_x", 10)
    info.cells_y = pydic.get("cells_y", 1)
    info.angles = pydic.get("angles", 4)
    info.groups = pydic.get("groups", 1)
    info.materials = pydic.get("materials", 1)
    # Geometry type (slab, sphere)
    info.geometry = pydic.get("geometry", 1)
    # Spatial discretization type
    info.spatial = pydic.get("spatial", 2)
    # External source
    info.qdim = pydic.get("qdim", 1)
    # Boundary parameters
    info.bc_x = pydic.get("bc_x", [0, 0])
    info.bcdim_x = pydic.get("bcdim_x", 1)
    info.bc_y = pydic.get("bc_y", [0, 0])
    info.bcdim_y = pydic.get("bcdim_y", 1)
    # Time dependent parameters
    info.steps = pydic.get("steps", 0)
    info.dt = pydic.get("dt", 1.0)
    info.bcdecay_x = pydic.get("bcdecay_x", 0)
    info.bcdecay_y = pydic.get("bcdecay_y", 0)
    # Angular flux option
    info.angular = pydic.get("angular", False)
    # Adjoint option, cross section matrices must be transposed
    info.adjoint = pydic.get("adjoint", False)
    # Flux at cell edges or centers
    info.edges = pydic.get("edges", 0) # 0 = Center, 1 = Edge
    return info


########################################################################
# One-dimensional functions
########################################################################

cdef int _check_fixed1d_source_iteration(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    if info.angular or info.edges:
        assert info.qdim == 3, "Need (I x N x G) fixed source"
    return 0


cdef int _check_nearby1d_fixed_source(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x N x G) source for analysis"
    assert info.bcdim_x == 3, "Need (2 x N x G) boundary for analysis"
    assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_nearby1d_criticality(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 3, "Need (I x N x G) source term"
    assert info.bcdim_x == 1, "No boundary conditions"
    # assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_timed1d_backward_euler(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x N x G) fixed source"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_critical1d_power_iteration(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 2, "Need (I x G) source term"
    # assert info.edges == 0, "Cannot currently use cell edges"
    assert info.bcdim_x == 1, "No boundary conditions"
    return 0


cdef int _check_critical1d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 3, "Need (I x N x G) source term"
    # assert info.edges == 0, "Cannot currently use cell edges"
    assert info.bcdim_x == 1, "No boundary conditions"
    return 0


cdef int _check_hybrid1d_bdf1_uncollided(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x N x G) fixed source"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_hybrid1d_bdf1_collided(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 2, "Need (I x G) fixed source"
    assert info.angular == False, "Scalar Flux is returned"
    assert info.bcdim_x == 1, "No Boundary conditions"
    return 0

########################################################################
# Two-dimensional functions
########################################################################

cdef int _check_fixed2d_source_iteration(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    if info.angular or info.edges:
        assert info.qdim == 3, "Need (I x J x N x G) fixed source"
    return 0


cdef int _check_nearby2d_fixed_source(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x J x N x G) source for analysis"
    assert info.bcdim_x == 4, "Need (2 x J x N x G) boundary for analysis"
    assert info.bcdim_y == 4, "Need (2 x I x N x G) boundary for analysis"
    assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_nearby2d_criticality(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 3, "Need (I x J x N x G) source for analysis"
    assert info.bcdim_x == 1, "No boundary conditions"
    assert info.bcdim_y == 1, "No boundary conditions"
    # assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_timed2d_backward_euler(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x J x N x G) fixed source"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_critical2d_power_iteration(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 2, "Need (I x J x G) source term"
    assert info.edges == 0, "Cannot currently use cell edges"
    assert info.bcdim_x == 1, "No boundary conditions"
    assert info.bcdim_y == 1, "No boundary conditions"
    return 0


cdef int _check_critical2d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.qdim == 3, "Need (I x J x N x G) source term"
    assert info.edges == 0, "Cannot currently use cell edges"
    assert info.bcdim_x == 1, "No boundary conditions"
    assert info.bcdim_y == 1, "No boundary conditions"
    return 0


cdef int _check_hybrid2d_bdf1_uncollided(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 3, "Need (I x J x N^2 x G) fixed source"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_hybrid2d_bdf1_collided(params info, int xs_length) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_length, "Incorrect number of materials"
    assert info.qdim == 2, "Need (I x J x G) fixed source"
    assert info.angular == False, "Scalar Flux is returned"
    assert info.bcdim_x == 1, "No Boundary conditions"
    assert info.bcdim_y == 1, "No Boundary conditions"
    return 0