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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

cdef params _to_params(object pydic):
    # Initialize params struct
    cdef params info

    # Collect Spatial cells, angles, energy groups
    info.cells_x = pydic.cells_x
    info.cells_y = pydic.cells_y
    info.angles = pydic.angles
    info.groups = pydic.groups
    info.materials = pydic.materials

    # Geometry type (slab, sphere)
    info.geometry = pydic.geometry

    # Spatial discretization type
    info.spatial = pydic.space_disc

    # Boundary parameters
    info.bc_x = pydic.bc_x
    info.bc_y = pydic.bc_y

    # Time dependent parameters
    info.steps = pydic.steps
    info.dt = pydic.dt

    # Angular flux option
    info.angular = pydic.angular

    # Flux at cell edges or centers
    info.flux_at_edges = pydic.flux_at_edges

    # OpenMP thread count: 0 means "use all available CPUs"
    import os
    info.num_threads = pydic.num_threads if pydic.num_threads > 0 \
        else os.cpu_count()

    # Parallelism strategy (1 = angle, 2 = group, 3 = both)
    info.parallel_type = pydic.parallel_type

    # Multigroup solver (1 = SI, 2 = DMD)
    info.mg_solver = pydic.mg_solver

    # DMD parameters
    info.dmd_snapshots = pydic.dmd_snapshots
    info.dmd_rank = pydic.dmd_rank

    # Artificial scattering parameters (ray effect mitigation)
    info.sigma_as = pydic.sigma_as
    info.beta_as = pydic.beta_as

    # Convergence parameters - iterations
    info.max_iter_angular = pydic.max_iter_angular
    info.max_iter_energy = pydic.max_iter_energy
    info.max_iter_keff = pydic.max_iter_keff

    # Convergence parameters - difference
    info.tol_angular = pydic.tol_angular
    info.tol_energy = pydic.tol_energy
    info.tol_keff = pydic.tol_keff


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
    # assert info.flux_at_edges == 0, "Cannot currently use cell edges"
    return 0


cdef int _check_critical1d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    # assert info.flux_at_edges == 0, "Cannot currently use cell edges"
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
    assert info.flux_at_edges == 0, "Cannot currently use cell edges"
    return 0


cdef int _check_critical2d_nearby_power(params info) except -1:
    assert info.angles % 2 == 0, "Need an even number of angles"
    assert info.flux_at_edges == 0, "Cannot currently use cell edges"
    return 0
