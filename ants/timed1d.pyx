########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One-Dimensional Time Dependent Source Multigroup NT Problems
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

import numpy as np
from tqdm import tqdm

from libc.math cimport sqrt

from ants cimport cytools_1d as tools
from ants cimport multi_group_1d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import TemporalDiscretization, create_params
from ants.fixed1d import known_flux as steady_state


def time_dependent(materials, sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:] velocity = materials.velocity
    cdef double[:,:,:] initial_flux = sources.initial_flux
    cdef double[:,:,:,:] external = sources.external
    cdef double[:,:,:,:] boundary_x = sources.boundary_x
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Validate half-angle boundary shape
    half_angles = int(np.sum(np.asarray(angle_x) > 0))
    assert boundary_x.shape[2] in (1, half_angles), \
        f"boundary_x angle dimension must be 1 (broadcast) or {half_angles} " \
        f"(incoming angles only), got {boundary_x.shape[2]}"

    # Optionally create a memmap file to stream per-step output
    cdef int steps = info.steps
    cdef int cells_x = info.cells_x
    cdef int groups = info.groups
    flux_file = None
    if time_data.save_to_file is not None:
        flux_file = np.lib.format.open_memmap(
            time_data.save_to_file, mode='w+', dtype=np.float64,
            shape=(steps, cells_x, groups)
        )

    if params.time_disc == TemporalDiscretization.BDF1:
        # Run backward Euler method
        parameters._check_bdf_timed1d(info, initial_flux.shape[0], external.shape[0], \
                                boundary_x.shape[0], xs_total.shape[0])
        flux_final = backward_euler(initial_flux.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x, medium_map, \
                        delta_x, angle_x, angle_w, flux_file, info)
    elif params.time_disc == TemporalDiscretization.CN:
        # Run Crank Nicolson method
        parameters._check_cn_timed1d(info, initial_flux.shape[0], \
                    external.shape[0], boundary_x.shape[0], xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1
        flux_final = crank_nicolson(initial_flux.copy(), xs_total, xs_matrix, \
                         velocity, external, boundary_x.copy(), medium_map, \
                         delta_x, angle_x, angle_w, flux_file, info, info_edge)
    elif params.time_disc == TemporalDiscretization.BDF2:
        # Run BDF2 method
        parameters._check_bdf_timed1d(info, initial_flux.shape[0], \
                    external.shape[0], boundary_x.shape[0], xs_total.shape[0])
        flux_final = bdf2(initial_flux.copy(), xs_total, xs_matrix, \
                           velocity, external, boundary_x.copy(), \
                           medium_map, delta_x, angle_x, angle_w, flux_file, info)
    elif params.time_disc == TemporalDiscretization.TR_BDF2:
        # Run TR-BDF2 method
        parameters._check_tr_bdf_timed1d(info, initial_flux.shape[0], \
                external.shape[0], boundary_x.shape[0], xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux_final = tr_bdf2(initial_flux.copy(), xs_total, xs_matrix, \
                        velocity, external.copy(), boundary_x.copy(), medium_map, \
                        delta_x, angle_x, angle_w, flux_file, info, info_edge)

    return np.asarray(flux_final)



cdef double[:,:] backward_euler(double[:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, object flux_file, params info):

    # Initialize time step, external and boundary indices
    cdef int step, qq, bc

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(flux_last, scalar_flux, angle_w, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF1    ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bc = 0 if boundary_x.shape[0] == 1 else step

        # Expand half-angle boundary to full-angle for sweep
        bc_full = tools._expand_boundary_x(boundary_x[bc], angle_x, info)

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external[qq], \
                                     velocity, info)

        # Solve for the current time step
        mg_result = mg.multi_group(scalar_flux, xs_total_v, xs_scatter, \
                                   q_star, bc_full, medium_map, \
                                   delta_x, angle_x, angle_w, info)
        scalar_flux[:,:] = mg_result[:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux)

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                    bc_full, medium_map, \
                                    delta_x, angle_x, angle_w, info)

    return scalar_flux


cdef double[:,:] crank_nicolson(double[:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, object flux_file, params info, \
        params info_edge):

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, bc

    # Create sigma_t + 2 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 2.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x, info.groups)
    tools._angular_edge_to_scalar(flux_last, scalar_flux, angle_w, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="CN      ", ascii=True):

        # Determine dimensions of external and boundary sources
        qqa = 0 if external.shape[0] == 1 else step # Previous time step
        qq = 0 if external.shape[0] == 1 else step + 1
        bc = 0 if boundary_x.shape[0] == 1 else step

        # Expand half-angle boundary to full-angle for sweep
        bc_full = tools._expand_boundary_x(boundary_x[bc], angle_x, info)

        # Update q_star
        tools._time_source_star_cn(flux_last, scalar_flux, xs_total, \
                        xs_scatter, velocity, q_star, external[qqa], \
                        external[qq], medium_map, delta_x, angle_x, \
                        2.0, info)
        # Solve for the current time step
        mg_result = mg.multi_group(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_full, \
                                medium_map, delta_x, angle_x, angle_w, info)
        scalar_flux[:,:] = mg_result[:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux)
        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info_edge)

    return scalar_flux


cdef double[:,:] bdf2(double[:,:,:]& flux_last_1, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:,:,:,:]& external, \
        double[:,:,:,:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, object flux_file, \
        params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step, external and boundary indices
    cdef int step, qq, bc

    # Combine total cross section and time coefficient (BDF1 time step)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(flux_last_1, scalar_flux, angle_w, info)

    # Create angular flux of previous time steps - Initialize with initial
    flux_last_2 = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bc = 0 if boundary_x.shape[0] == 1 else step

        # Expand half-angle boundary to full-angle for sweep
        bc_full = tools._expand_boundary_x(boundary_x[bc], angle_x, info)

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, \
                                    external[qq], velocity, info)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                        external[qq], velocity, info)

        # Solve for the current time step
        mg_result = mg.multi_group(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_full, \
                                medium_map, delta_x, angle_x, angle_w, info)
        scalar_flux[:,:] = mg_result[:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux)

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_2[:,:,:] = flux_last_1[:,:,:]
        flux_last_1[:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info)

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_v[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v, velocity, 1.5, info)

    return scalar_flux


cdef double[:,:] tr_bdf2(double[:,:,:]& flux_last_ell, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:,:,:,:]& external, \
        double[:,:,:,:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, object flux_file, \
        params info, params info_edge):

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, qqb, bc, bca

    # Initialize gamma
    cdef double gamma = 0.5 # 2 - sqrt(2)

    # Create sigma_t + 2 / (gamma * v * dt) - CN Step
    xs_total_v_cn = tools.array_2d(info.materials, info.groups)
    xs_total_v_cn[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v_cn, velocity, 2.0 / gamma, info)

    # Create sigma_t + (2 - gamma) / ((1 - gamma) * v * dt) - BDF2 Step
    xs_total_v_bdf2 = tools.array_2d(info.materials, info.groups)
    xs_total_v_bdf2[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v_bdf2, velocity, \
                          (2.0 - gamma) / (1.0 - gamma), info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux_ell = tools.array_2d(info.cells_x, info.groups)
    tools._angular_edge_to_scalar(flux_last_ell, scalar_flux_ell, angle_w, info)
    scalar_flux_gamma = tools.array_2d(info.cells_x, info.groups)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="TR-BDF2 ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step * 2 # Ell Step
        qqa = 0 if external.shape[0] == 1 else step * 2 + 1 # Gamma Step
        qqb = 0 if external.shape[0] == 1 else step * 2 + 2 # Ell + 1 Step

        bc = 0 if boundary_x.shape[0] == 1 else step * 2 # Ell Step
        bca = 0 if boundary_x.shape[0] == 1 else step * 2 + 1 # Gamma Step

        # Expand half-angle boundaries to full-angle for sweep
        bc_full = tools._expand_boundary_x(boundary_x[bc], angle_x, info)
        bca_full = tools._expand_boundary_x(boundary_x[bca], angle_x, info)

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn(flux_last_ell, scalar_flux_ell, xs_total, \
                        xs_scatter, velocity, q_star, external[qq], \
                        external[qqa], medium_map, delta_x, angle_x, \
                        2.0 / gamma, info)

        # Solve for the \ell + gamma time step
        scalar_flux_gamma[:,:] = mg.multi_group(
            scalar_flux_ell,
            xs_total_v_cn,
            xs_scatter,
            q_star,
            bc_full,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            info,
        )

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux_gamma, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_gamma = mg._known_source_angular(xs_total_v_cn, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info)
        ################################################################
        # BDF2
        ################################################################
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_last_ell, flux_last_gamma, \
                    q_star, external[qqb], velocity, gamma, info)

        # Solve for the \ell + 1 time step
        mg_result = mg.multi_group(
            scalar_flux_gamma,
            xs_total_v_bdf2,
            xs_scatter,
            q_star,
            bca_full,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            info,
        )
        scalar_flux_ell[:,:] = mg_result[:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux_ell)

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux_ell, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_ell[:,:,:] = mg._known_source_angular(xs_total_v_bdf2, \
                                    q_star, bca_full, medium_map, \
                                    delta_x, angle_x, angle_w, info_edge)

    return scalar_flux_ell


def known_source_calculation(double[:,:,:] flux, materials, sources, geometry, \
        quadrature, solver, time):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:,:,:,:] external = sources.external
    cdef double[:,:,:,:] boundary_x = sources.boundary_x

    # Initialize iterables
    cdef int step, qq, bc

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver, time)
    info = parameters._to_params(params)

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Initialize angular flux
    angular_flux = tools.array_4d(info.steps, info.cells_x, info.angles, info.groups)

    cdef double[:] angle_x = quadrature.angle_x

    # Validate half-angle boundary shape
    half_angles = int(np.sum(np.asarray(angle_x) > 0))
    assert boundary_x.shape[2] in (1, half_angles), \
        f"boundary_x angle dimension must be 1 (broadcast) or {half_angles} " \
        f"(incoming angles only), got {boundary_x.shape[2]}"

    for step in range(info.steps):
        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bc = 0 if boundary_x.shape[0] == 1 else step

        # Expand half-angle boundary to full-angle
        bc_full = tools._expand_boundary_x(boundary_x[bc], angle_x, info)

        angular_flux[step] = steady_state(flux[step], xs_total, xs_matrix, \
                                            external[qq], bc_full, \
                                            geometry, quadrature, params)

    return np.asarray(angular_flux)
