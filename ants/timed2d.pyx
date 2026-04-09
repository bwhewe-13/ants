########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

import numpy as np
from tqdm.auto import tqdm

from libc.math cimport sqrt

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import TemporalDiscretization, create_params


def time_dependent(materials, sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:] velocity = materials.velocity
    cdef double[:,:,:,:,:] external = sources.external
    cdef double[:,:,:,:,:] boundary_x = sources.boundary_x
    cdef double[:,:,:,:,:] boundary_y = sources.boundary_y
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    cdef double[:,:,:,:] initial_flux_x
    cdef double[:,:,:,:] initial_flux_y
    cdef double[:,:,:,:] initial_flux
    if time_data.time_disc == TemporalDiscretization.CN or time_data.time_disc == TemporalDiscretization.TR_BDF2:
        initial_flux_x = sources.initial_flux_x
        initial_flux_y = sources.initial_flux_y
    else:
        initial_flux = sources.initial_flux

    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    if params.time_disc == TemporalDiscretization.BDF1:
        parameters._check_bdf_timed2d(info, initial_flux.shape[0], external.shape[0], \
                        boundary_x.shape[0], boundary_y.shape[0], xs_total.shape[0])
        # Run Backward Euler
        flux = backward_euler(initial_flux.copy(), xs_total, xs_matrix, velocity, \
                    external, boundary_x.copy(), boundary_y.copy(), medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info)

    elif params.time_disc == TemporalDiscretization.CN:
        parameters._check_cn_timed2d(info, initial_flux_x.shape[0], \
                    initial_flux_y.shape[1], external.shape[0], boundary_x.shape[0], \
                    boundary_y.shape[0], xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux = crank_nicolson(initial_flux_x.copy(), initial_flux_y.copy(), \
                    xs_total, xs_matrix, velocity, external, boundary_x.copy(), \
                    boundary_y.copy(), medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, info, info_edge)

    elif params.time_disc == TemporalDiscretization.BDF2:
        parameters._check_bdf_timed2d(info, initial_flux.shape[0], \
                                    external.shape[0], boundary_x.shape[0], \
                                    boundary_y.shape[0], xs_total.shape[0])
        # Run BDF2
        flux = bdf2(initial_flux.copy(), xs_total, xs_matrix, velocity, external, \
                    boundary_x.copy(), boundary_y.copy(), medium_map, delta_x, \
                    delta_y, angle_x, angle_y, angle_w, info)

    elif params.time_disc == TemporalDiscretization.TR_BDF2:
        parameters._check_tr_bdf_timed2d(info, initial_flux_x.shape[0], \
                initial_flux_y.shape[1], external.shape[0], boundary_x.shape[0], \
                boundary_y.shape[0], xs_total.shape[0])

        # Create params with edges for TR/BDF2 method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux = tr_bdf2(initial_flux_x.copy(), initial_flux_y.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x.copy(), \
                        boundary_y.copy(), medium_map, delta_x, delta_y, angle_x, \
                        angle_y, angle_w, info, info_edge)

    return np.asarray(flux)


cdef double[:,:,:,:] backward_euler(double[:,:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:,:,:]& external, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(flux_last, scalar_flux, angle_w, info)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF1    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bcx = 0 if boundary_x.shape[0] == 1 else step
        bcy = 0 if boundary_y.shape[0] == 1 else step

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external[qq], \
                                     velocity, info)

        # Run source iteration
        flux_time[step] = mg.multi_group(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx], \
                                boundary_y[bcy], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, \
                               medium_map, info)

        # Solve for angular flux of previous time step
        flux_last[:,:,:,:] = mg._known_source_angular(xs_total_v, q_star,
                                        boundary_x[bcx], boundary_y[bcy], \
                                        medium_map, delta_x, delta_y, \
                                        angle_x, angle_y, angle_w, info)

    return flux_time[:,:,:,:]


cdef double[:,:,:,:] crank_nicolson(double[:,:,:,:]& flux_last_x, \
        double[:,:,:,:]& flux_last_y, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, \
        params info_edge):
    # flux_last_x = (cells_x + 1, cells_y, angles**2, groups) - x edges
    # flux_last_y = (cells_x, cells_y + 1, angles**2, groups) - y edges

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, bcx, bcy

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 2.0, info)

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y,\
                            info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_edge_to_scalar(flux_last_x, flux_last_y, scalar_flux, \
                                  angle_w, info)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="CN      ", ascii=True):

        # Determine dimensions of external and boundary sources
        qqa = 0 if external.shape[0] == 1 else step # Previous time step
        qq = 0 if external.shape[0] == 1 else step + 1
        bcx = 0 if boundary_x.shape[0] == 1 else step
        bcy = 0 if boundary_y.shape[0] == 1 else step

        # Update q_star
        tools._time_source_star_cn(flux_last_x, flux_last_y, scalar_flux, \
                xs_total, xs_scatter, velocity, q_star, external[qqa], \
                external[qq], medium_map, delta_x, delta_y, angle_x, \
                angle_y, 2.0, info)

        # Run source iteration
        flux_time[step] = mg.multi_group(scalar_flux, xs_total_v, \
                                    xs_scatter, q_star, boundary_x[bcx], \
                                    boundary_y[bcy], medium_map, delta_x, \
                                    delta_y, angle_x, angle_y, angle_w, info)
        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        mg._interface_angular(flux_last_x, flux_last_y, xs_total_v, q_star, \
                boundary_x[bcx], boundary_y[bcy], medium_map, \
                delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)

    return flux_time[:,:,:,:]


cdef double[:,:,:,:] bdf2(double[:,:,:,:]& flux_last_1, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:,:,:,:,:]& external, \
        double[:,:,:,:,:]& boundary_x, double[:,:,:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 1 / (v * dt) (BDF1 time step)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(flux_last_1, scalar_flux, angle_w, info)

    # Create angular flux of previous time steps
    flux_last_2 = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bcx = 0 if boundary_x.shape[0] == 1 else step
        bcy = 0 if boundary_y.shape[0] == 1 else step

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, external[qq], \
                                         velocity, info)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                         external[qq], velocity, info)

        # Run source iteration
        flux_time[step] = mg.multi_group(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx], \
                                boundary_y[bcy], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]

        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                    boundary_x[bcx], boundary_y[bcy], \
                                    medium_map, delta_x, delta_y, angle_x, \
                                    angle_y, angle_w, info)

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_v[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v, velocity, 1.5, info)

    return flux_time[:,:,:,:]


def restart_bdf2(double[:,:,:,:] flux_1, double[:,:,:,:] flux_2, materials, \
        sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:] velocity = materials.velocity
    cdef double[:,:,:,:,:] external = sources.external
    cdef double[:,:,:,:,:] boundary_x = sources.boundary_x
    cdef double[:,:,:,:,:] boundary_y = sources.boundary_y
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w


    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)
    parameters._check_bdf_timed2d(info, flux_1.shape[0], external.shape[0], \
            boundary_x.shape[0], boundary_y.shape[0], xs_total.shape[0])

    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Run BDF2 with 2 known fluxes
    flux = multi_group_bdf2_restart(flux_1.copy(), flux_2.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x.copy(), \
                        boundary_y.copy(), medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, info)

    return np.asarray(flux)


cdef double[:,:,:,:] multi_group_bdf2_restart(double[:,:,:,:]& flux_last_1, \
        double[:,:,:,:]& flux_last_2, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.5, info)

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(flux_last_1, scalar_flux, angle_w, info)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bcx = 0 if boundary_x.shape[0] == 1 else step
        bcy = 0 if boundary_y.shape[0] == 1 else step

        # Run BDF2 on rest of time steps
        tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                     external[qq], velocity, info)

        # Run source iteration
        flux_time[step] = mg.multi_group(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx], \
                                boundary_y[bcy], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]

        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                    boundary_x[bcx], boundary_y[bcy], \
                                    medium_map, delta_x, delta_y, angle_x, \
                                    angle_y, angle_w, info)

    return flux_time[:,:,:,:]


def angular_bdf2(int[:] time_steps, double[:,:,:,:] scalar_flux, \
        double[:,:,:,:] initial_flux, materials, sources, geometry, quadrature, \
        solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:] velocity = materials.velocity
    cdef double[:,:,:,:,:] external = sources.external
    cdef double[:,:,:,:,:] boundary_x = sources.boundary_x
    cdef double[:,:,:,:,:] boundary_y = sources.boundary_y
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)
    parameters._check_bdf_timed2d(info, initial_flux.shape[0], \
                                  external.shape[0], boundary_x.shape[0], \
                                  boundary_y.shape[0], xs_total.shape[0])

    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Run BDF2 with known scalar flux
    flux = multi_group_bdf2_angular(time_steps, scalar_flux.copy(), initial_flux.copy(), \
                xs_total, xs_matrix, velocity, external, boundary_x.copy(), \
                boundary_y.copy(), medium_map, delta_x, delta_y, angle_x, \
                angle_y, angle_w, info, )

    return np.asarray(flux)


cdef double[:,:,:,:,:] multi_group_bdf2_angular(int[:]& time_steps, double[:,:,:,:]& scalar_flux, \
        double[:,:,:,:]& flux_last_1, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy
    cdef int count = 0

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Create angular flux of previous time steps
    flux_last_2 = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_5d(time_steps.shape[0], info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Iterate over time steps
    for step in range(info.steps):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bcx = 0 if boundary_x.shape[0] == 1 else step
        bcy = 0 if boundary_y.shape[0] == 1 else step

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_total_bdf1(scalar_flux[step], flux_last_1, \
                                xs_scatter, velocity, q_star, external[qq],\
                                medium_map, info)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_total_bdf2(scalar_flux[step], flux_last_1, \
                                flux_last_2, xs_scatter, velocity, q_star, \
                                external[qq], medium_map, info)

        # Solve for angular flux of previous time step
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_v, q_star, \
                                            boundary_x[bcx], boundary_y[bcy], \
                                            medium_map, delta_x, delta_y, angle_x, \
                                            angle_y, angle_w, info)

        # Keep specific time steps
        if (step == time_steps[count]):
            flux_time[count] = flux_last_1[:,:,:,:]
            count += 1

        # Exit early
        if (count == time_steps.shape[0]):
            return flux_time[:,:,:,:,:]

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_v[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v, velocity, 1.5, info)

    return flux_time[:,:,:,:,:]


cdef double[:,:,:,:] tr_bdf2(double[:,:,:,:]& flux_ell_x, double[:,:,:,:]& flux_ell_y, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:,:]& external, double[:,:,:,:,:]& boundary_x, \
        double[:,:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, params info_edge):

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, qqb, bcx, bcxa, bcy, bcya

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
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_ell = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_edge_to_scalar(flux_ell_x, flux_ell_y, \
                                  scalar_ell, angle_w, info)
    scalar_gamma = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_edge_to_scalar(flux_ell_x, flux_ell_y, \
                                  scalar_flux, angle_w, info)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="TR-BDF2 ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step * 2 # Ell Step
        qqa = 0 if external.shape[0] == 1 else step * 2 + 1 # Gamma Step
        qqb = 0 if external.shape[0] == 1 else step * 2 + 2 # Ell + 1 Step

        bcx = 0 if boundary_x.shape[0] == 1 else step * 2 # Ell Step
        bcxa = 0 if boundary_x.shape[0] == 1 else step * 2 + 1 # Gamma Step

        bcy = 0 if boundary_y.shape[0] == 1 else step * 2 # Ell Step
        bcya = 0 if boundary_y.shape[0] == 1 else step * 2 + 1 # Gamma Step

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn(flux_ell_x, flux_ell_y, scalar_flux, \
                    xs_total, xs_scatter, velocity, q_star, external[qq], \
                    external[qqa], medium_map, delta_x, delta_y, angle_x, \
                    angle_y, 2.0 / gamma, info)

        # Solve for the \ell + gamma time step
        scalar_flux[:,:,:] = mg.multi_group(scalar_flux, xs_total_v_cn, \
                            xs_scatter, q_star, boundary_x[bcx], \
                            boundary_y[bcy], medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of \ell + gamma time step
        flux_last_gamma = mg._known_source_angular(xs_total_v_cn, q_star, \
                        boundary_x[bcx], boundary_y[bcy], medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info)

        ################################################################
        # BDF2
        ################################################################
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_ell_x, flux_ell_y, flux_last_gamma, \
                            q_star, external[qqb], velocity, gamma, info)

        # Solve for the \ell + 1 time step
        flux_time[step] = mg.multi_group(scalar_flux, xs_total_v_bdf2, \
                                xs_scatter, q_star, boundary_x[bcxa], \
                                boundary_y[bcya], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_ell_x, flux_ell_y, xs_total_v_bdf2, \
                    q_star, boundary_x[bcxa], boundary_y[bcya], medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)

    return flux_time[:,:,:,:]
