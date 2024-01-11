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
# cython: profile=True
# distutils: language = c++

from ants.constants import *

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
    
    # Boundary parameters
    info.bc_x = pydic.get("bc_x", [0, 0])
    info.bc_y = pydic.get("bc_y", [0, 0])
    
    # Time dependent parameters
    info.steps = pydic.get("steps", 0)
    info.dt = pydic.get("dt", 1.0)
    
    # Angular flux option
    info.angular = pydic.get("angular", False)

    # Flux at cell edges or centers
    info.edges = pydic.get("edges", 0) # 0 = Center, 1 = Edge
    
    # Multigroup solver (1 = SI, 2 = DMD)
    info.mg = pydic.get("mg", 1)
    
    # DMD parameters
    info.dmd_k = pydic.get("dmd_k", 40)
    info.dmd_r = pydic.get("dmd_r", 2)
    
    # Convergence parameters - iterations
    info.count_nn = pydic.get("count_nn", COUNT_ANGULAR)
    info.count_gg = pydic.get("count_gg", COUNT_ENERGY)
    info.count_keff = pydic.get("count_keff", COUNT_POWER)
    
    # Convergence parameters - difference
    info.change_nn = pydic.get("change_nn", CHANGE_ANGULAR)
    info.change_gg = pydic.get("change_gg", CHANGE_ENERGY)
    info.change_keff = pydic.get("change_keff", CHANGE_POWER)
    return info


########################################################################
# One-dimensional functions
########################################################################

cdef int _check_fixed1d_source_iteration(params info, int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    return 0


cdef int _check_nearby1d_fixed_source(params info, int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_nearby1d_criticality(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    # assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_timed1d(params info, int bc_x_shape, int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_bdf_timed1d(params info, int psi_shape, int q_shape, \
        int bc_x_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed1d(info, bc_x_shape, xs_shape)
    assert psi_shape == info.cells_x, "Need initial flux at cell centers"
    if q_shape > 1:
        assert q_shape == info.steps, \
                "Need time-dependent external source for each time step"
    if bc_x_shape > 1:
        assert bc_x_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    return 0


cdef int _check_cn_timed1d(params info, int psi_shape, int q_shape, \
        int bc_x_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed1d(info, bc_x_shape, xs_shape)
    assert psi_shape == (info.cells_x + 1), "Need initial flux at cell edges"
    if q_shape > 1:
        assert q_shape == (info.steps + 1), "Need time-dependent external " \
                "source for each time step and initial time step"
    if bc_x_shape > 1:
        assert bc_x_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    return 0


cdef int _check_tr_bdf_timed1d(params info, int psi_shape, int q_shape, \
        int bc_x_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed1d(info, bc_x_shape, xs_shape)
    assert psi_shape == (info.cells_x + 1), "Need initial flux at cell edges"
    if q_shape > 1:
        assert q_shape == (info.steps * 2 + 1), "Need time-dependent " \
            "external source for each time step, gamma step, and initial step"
    if bc_x_shape > 1:
        assert bc_x_shape == (info.steps * 2), "Need time-dependent " \
            "boundary source for each time step, gamma step, and initial step"
    return 0


cdef int _check_critical1d_power_iteration(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    # assert info.edges == 0, "Cannot currently use cell edges"
    return 0


cdef int _check_critical1d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    # assert info.edges == 0, "Cannot currently use cell edges"
    return 0


########################################################################
# Two-dimensional functions
########################################################################

cdef int _check_fixed2d_source_iteration(params info, int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    return 0


cdef int _check_nearby2d_fixed_source(params info, int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_nearby2d_criticality(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    # assert info.angular == True, "Need angular flux for analysis"
    return 0


cdef int _check_timed2d(params info, int bc_x_shape, int bc_y_shape, \
        int xs_shape) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.materials == xs_shape, "Incorrect number of materials"
    assert info.steps > 0, "Need at least 1 time step"
    assert info.angular == False, "Scalar flux is returned"
    return 0


cdef int _check_bdf_timed2d(params info, int psi_shape, int q_shape, \
        int bc_x_shape, int bc_y_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed2d(info, bc_x_shape, bc_y_shape, xs_shape)
    assert psi_shape == info.cells_x, "Need initial flux at cell centers"
    if q_shape > 1:
        assert q_shape == info.steps, \
                "Need time-dependent external source for each time step"
    if bc_x_shape > 1:
        assert bc_x_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    if bc_y_shape > 1:
        assert bc_y_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    return 0


cdef int _check_cn_timed2d(params info, int psi_x_shape, int psi_y_shape, \
        int q_shape, int bc_x_shape, int bc_y_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed2d(info, bc_x_shape, bc_y_shape, xs_shape)
    assert psi_x_shape == (info.cells_x + 1), "Need initial flux at cell edges"
    assert psi_y_shape == (info.cells_y + 1), "Need initial flux at cell edges"
    if q_shape > 1:
        assert q_shape == (info.steps + 1), "Need time-dependent external " \
                "source for each time step and initial time step"
    if bc_x_shape > 1:
        assert bc_x_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    if bc_y_shape > 1:
        assert bc_y_shape == info.steps, \
                "Need time-dependent boundary source for each time step"
    return 0


cdef int _check_tr_bdf_timed2d(params info, int psi_x_shape, int psi_y_shape, \
        int q_shape, int bc_x_shape, int bc_y_shape, int xs_shape) except -1:
    # Go through time-dependent default checks
    _check_timed2d(info, bc_x_shape, bc_y_shape, xs_shape)
    assert psi_x_shape == (info.cells_x + 1), "Need initial flux at cell edges"
    assert psi_y_shape == (info.cells_y + 1), "Need initial flux at cell edges"
    if q_shape > 1:
        assert q_shape == (info.steps * 2 + 1), "Need time-dependent " \
            "external source for each time step, gamma step, and initial step"
    if bc_x_shape > 1:
        assert bc_x_shape == (info.steps * 2), "Need time-dependent " \
                "boundary source for each time step, gamma step, and initial step"
    if bc_y_shape > 1:
        assert bc_y_shape == (info.steps * 2), "Need time-dependent " \
                "boundary source for each time step, gamma step, and initial step"
    return 0


cdef int _check_critical2d_power_iteration(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.edges == 0, "Cannot currently use cell edges"
    return 0


cdef int _check_critical2d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.edges == 0, "Cannot currently use cell edges"
    return 0