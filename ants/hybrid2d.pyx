########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two-Dimensional Hybrid Multigroup Neutron Transport Problems
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

# Uncollided is fine grid (N^2 x G)
# Collided is coarse grid (N'^2 x G')


def time_dependent(materials_u, materials_c, sources, geometry, quadrature_u, \
        quadrature_c, solver, time_data, hybrid_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total_u = materials_u.total
    cdef double[:,:,:] xs_scatter_u = materials_u.scatter
    cdef double[:,:,:] xs_fission_u = materials_u.fission
    cdef double[:] velocity_u = materials_u.velocity
    cdef double[:,:] xs_total_c = materials_c.total
    cdef double[:,:,:] xs_scatter_c = materials_c.scatter
    cdef double[:,:,:] xs_fission_c = materials_c.fission
    cdef double[:] velocity_c = materials_c.velocity
    cdef double[:,:,:,:,:] external_u = sources.external
    cdef double[:,:,:,:,:] boundary_xu = sources.boundary_x
    cdef double[:,:,:,:,:] boundary_yu = sources.boundary_y
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_xu = quadrature_u.angle_x
    cdef double[:] angle_yu = quadrature_u.angle_y
    cdef double[:] angle_wu = quadrature_u.angle_w
    cdef double[:] angle_xc = quadrature_c.angle_x
    cdef double[:] angle_yc = quadrature_c.angle_y
    cdef double[:] angle_wc = quadrature_c.angle_w
    cdef int[:] fine_idx = hybrid_data.fine_idx
    cdef int[:] coarse_idx = hybrid_data.coarse_idx
    cdef double[:] factor = hybrid_data.factor

    # Convert uncollided dictionary to type params
    params_u = create_params(materials_u, quadrature_u, geometry, solver, time_data)
    info_u = parameters._to_params(params_u)

    cdef double[:,:,:,:] initial_flux_x
    cdef double[:,:,:,:] initial_flux_y
    cdef double[:,:,:,:] initial_flux
    if time_data.time_disc == TemporalDiscretization.CN or time_data.time_disc == TemporalDiscretization.TR_BDF2:
        initial_flux_x = sources.initial_flux_x
        initial_flux_y = sources.initial_flux_y
    else:
        initial_flux = sources.initial_flux

    # Convert collided dictionary to type params
    params_c = create_params(materials_c, quadrature_c, geometry, solver, time_data)
    info_c = parameters._to_params(params_c)
    parameters._check_timed2d(info_c, 0, 0, xs_total_c.shape[0])

    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)

    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)

    if params_u.time_disc == TemporalDiscretization.BDF1:
        parameters._check_bdf_timed2d(info_u, initial_flux.shape[0], external_u.shape[0], \
                        boundary_xu.shape[0], boundary_yu.shape[0], xs_total_u.shape[0])
        # Run Backward Euler
        flux = backward_euler(initial_flux.copy(), xs_total_u, xs_total_c, \
                    xs_matrix_u, xs_matrix_c, velocity_u, velocity_c, external_u, \
                    boundary_xu.copy(), boundary_yu.copy(), medium_map, delta_x, \
                    delta_y, angle_xu, angle_xc, angle_yu, angle_yc, angle_wu, \
                    angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)
    elif params_u.time_disc == TemporalDiscretization.CN:
        parameters._check_cn_timed2d(info_u, initial_flux_x.shape[0], \
                    initial_flux_y.shape[1], external_u.shape[0], boundary_xu.shape[0], \
                    boundary_yu.shape[0], xs_total_u.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        flux = crank_nicolson(initial_flux_x.copy(), initial_flux_y.copy(), \
                    xs_total_u, xs_total_c, xs_matrix_u, xs_matrix_c, \
                    velocity_u, velocity_c, external_u, boundary_xu.copy(), \
                    boundary_yu.copy(), medium_map, delta_x, delta_y, angle_xu, \
                    angle_xc, angle_yu, angle_yc, angle_wu, angle_wc, \
                    fine_idx, coarse_idx, factor, info_u, info_c, info_edge)

    elif params_u.time_disc == TemporalDiscretization.BDF2:
        parameters._check_bdf_timed2d(info_u, initial_flux.shape[0], external_u.shape[0], \
                        boundary_xu.shape[0], boundary_yu.shape[0], xs_total_u.shape[0])
        # Run BDF2
        flux = bdf2(initial_flux.copy(), xs_total_u, xs_total_c, xs_matrix_u, \
                    xs_matrix_c, velocity_u, velocity_c, external_u, \
                    boundary_xu.copy(), boundary_yu.copy(), medium_map, delta_x, \
                    delta_y, angle_xu, angle_xc, angle_yu, angle_yc, angle_wu, \
                    angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)

    elif params_u.time_disc == TemporalDiscretization.TR_BDF2:
        parameters._check_tr_bdf_timed2d(info_u, initial_flux_x.shape[0], \
                        initial_flux_y.shape[1], external_u.shape[0], \
                        boundary_xu.shape[0], boundary_yu.shape[0], xs_total_u.shape[0])

        # Create params with edges for TR/BDF2 method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        flux = tr_bdf2(initial_flux_x.copy(), initial_flux_y.copy(), xs_total_u, \
                    xs_total_c, xs_matrix_u, xs_matrix_c, velocity_u, velocity_c, \
                    external_u, boundary_xu.copy(), boundary_yu.copy(), medium_map, \
                    delta_x, delta_y, angle_xu, angle_xc, angle_yu, angle_yc, \
                    angle_wu, angle_wc, fine_idx, coarse_idx, factor, info_u, \
                    info_c, info_edge)

    return np.asarray(flux)


cdef double[:,:,:,:] backward_euler(double[:,:,:,:]& flux_last, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, \
        double[:,:,:,:,:]& external_u, double[:,:,:,:,:]& boundary_xu, \
        double[:,:,:,:,:]& boundary_yu, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_yu, double[:]& angle_yc, \
        double[:]& angle_wu, double[:]& angle_wc, int[:]& fine_idx, \
        int[:]& coarse_idx, double[:]& factor, params info_u, \
        params info_c):

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    tools._angular_to_scalar(flux_last, flux_u, angle_wu, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_4d(info_c.cells_x, info_c.cells_y, 1, info_c.groups)
    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF1*    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bcx = 0 if boundary_xu.shape[0] == 1 else step
        bcy = 0 if boundary_yu.shape[0] == 1 else step

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external_u[qq], \
                                     velocity_u, info_u)
        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bcx], boundary_yu[bcy], boundary_c, \
                      medium_map, delta_x, delta_y, angle_xu, angle_xc, \
                      angle_yu, angle_yc, angle_wu, angle_wc, fine_idx, \
                      coarse_idx, factor, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                            boundary_xu[bcx], boundary_yu[bcy], medium_map, \
                            delta_x, delta_y, angle_xu, angle_yu, angle_wu, \
                            info_u)

        # Step 5: Update and repeat
        tools._angular_to_scalar(flux_last, flux_time[step], angle_wu, info_u)

    return flux_time[:,:,:,:]



cdef double[:,:,:,:] crank_nicolson(double[:,:,:,:]& flux_last_x, \
        double[:,:,:,:]& flux_last_y, double[:,:]& xs_total_u, \
        double[:,:]& xs_total_c, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:,:,:,:,:]& external_u, \
        double[:,:,:,:,:]& boundary_xu, double[:,:,:,:,:]& boundary_yu, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_yu, \
        double[:]& angle_yc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        params info_u, params info_c, params info_edge):
    # flux_last_x = (cells_x + 1, cells_y, angles**2, groups) - x edges
    # flux_last_y = (cells_x, cells_y + 1, angles**2, groups) - y edges

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, bcx, bcy

    # Create sigma_t + 2 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 2.0, info_u)

    # Create sigma_t + 2 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 2.0, info_c)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    tools._angular_edge_to_scalar(flux_last_x, flux_last_y, flux_u, \
                                  angle_wu, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_4d(info_c.cells_x, info_c.cells_y, 1, info_c.groups)
    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="CN*      ", ascii=True):

        # Determine dimensions of external and boundary sources
        qqa = 0 if external_u.shape[0] == 1 else step # Previous time step
        qq = 0 if external_u.shape[0] == 1 else step + 1
        bcx = 0 if boundary_xu.shape[0] == 1 else step
        bcy = 0 if boundary_yu.shape[0] == 1 else step

        # Update q_star
        tools._time_source_star_cn(flux_last_x, flux_last_y, flux_u, \
                        xs_total_u, xs_scatter_u, velocity_u, q_star, \
                        external_u[qqa], external_u[qq], medium_map, \
                        delta_x, delta_y, angle_xu, angle_yu, 2.0, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bcx], boundary_yu[bcy], boundary_c, \
                      medium_map, delta_x, delta_y, angle_xu, angle_xc, \
                      angle_yu, angle_yc, angle_wu, angle_wc, fine_idx, \
                      coarse_idx, factor, info_u, info_c)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_last_x, flux_last_y, xs_total_vu, \
                q_star, boundary_xu[bcx], boundary_yu[bcy], medium_map, \
                delta_x, delta_y, angle_xu, angle_yu, angle_wu, info_edge)

        # Step 5: Update and repeat
        tools._angular_edge_to_scalar(flux_last_x, flux_last_y, \
                                      flux_time[step], angle_wu, info_u)
        flux_u[:,:,:] = flux_time[step,:,:,:]

    return flux_time[:,:,:,:]


cdef double[:,:,:,:] bdf2(double[:,:,:,:]& flux_last_1, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, \
        double[:,:,:,:,:]& external_u, double[:,:,:,:,:]& boundary_xu, \
        double[:,:,:,:,:]& boundary_yu, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_yu, double[:]& angle_yc, \
        double[:]& angle_wu, double[:]& angle_wc, int[:]& fine_idx, \
        int[:]& coarse_idx, double[:]& factor, params info_u, params info_c):

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_2 = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                                 info_u.angles * info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    tools._angular_to_scalar(flux_last_1, flux_u, angle_wu, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_4d(info_c.cells_x, info_c.cells_y, 1, info_c.groups)
    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF2*    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bcx = 0 if boundary_xu.shape[0] == 1 else step
        bcy = 0 if boundary_yu.shape[0] == 1 else step

        # Update q_star
        if step == 0:
            # Run BDF1 on first step
            tools._time_source_star_bdf1(flux_last_1, q_star, external_u[qq], \
                                         velocity_u, info_u)
        else:
            # Run BDF2 on all other steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                         external_u[qq], velocity_u, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bcx], boundary_yu[bcy], boundary_c, \
                      medium_map, delta_x, delta_y, angle_xu, angle_xc, \
                      angle_yu, angle_yc, angle_wu, angle_wc, fine_idx, \
                      coarse_idx, factor, info_u, info_c)

        # Step 5: Update steps
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                        boundary_xu[bcx], boundary_yu[bcy], \
                                        medium_map, delta_x, delta_y, angle_xu, \
                                        angle_yu, angle_wu, info_u)

        tools._angular_to_scalar(flux_last_1, flux_time[step], angle_wu, info_u)

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_vu[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)
            xs_total_vc[:,:] = xs_total_c[:,:]
            tools._total_velocity(xs_total_vc, velocity_c, 1.5, info_c)

    return flux_time[:,:,:,:]


def restart_bdf2(double[:,:,:,:] flux_1, double[:,:,:,:] flux_2, \
        materials_u, materials_c, sources, geometry, quadrature_u, \
        quadrature_c, solver, time_data, hybrid_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total_u = materials_u.total
    cdef double[:,:,:] xs_scatter_u = materials_u.scatter
    cdef double[:,:,:] xs_fission_u = materials_u.fission
    cdef double[:] velocity_u = materials_u.velocity
    cdef double[:,:] xs_total_c = materials_c.total
    cdef double[:,:,:] xs_scatter_c = materials_c.scatter
    cdef double[:,:,:] xs_fission_c = materials_c.fission
    cdef double[:] velocity_c = materials_c.velocity
    cdef double[:,:,:,:,:] external_u = sources.external
    cdef double[:,:,:,:,:] boundary_xu = sources.boundary_x
    cdef double[:,:,:,:,:] boundary_yu = sources.boundary_y
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_xu = quadrature_u.angle_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_yu = quadrature_u.angle_y
    cdef double[:] angle_wu = quadrature_u.angle_w
    cdef double[:] angle_xc = quadrature_c.angle_x
    cdef double[:] angle_yc = quadrature_c.angle_y
    cdef double[:] angle_wc = quadrature_c.angle_w
    cdef int[:] fine_idx = hybrid_data.fine_idx
    cdef int[:] coarse_idx = hybrid_data.coarse_idx
    cdef double[:] factor = hybrid_data.factor

    # Convert uncollided dictionary to type params
    params_u = create_params(materials_u, quadrature_u, geometry, solver, time_data)
    info_u = parameters._to_params(params_u)
    parameters._check_bdf_timed2d(info_u, flux_1.shape[0], external_u.shape[0], \
                boundary_xu.shape[0], boundary_yu.shape[0], xs_total_u.shape[0])

    # Convert collided dictionary to type params
    params_c = create_params(materials_c, quadrature_c, geometry, solver, time_data)
    info_c = parameters._to_params(params_c)
    parameters._check_timed2d(info_c, 0, 0, xs_total_c.shape[0])

    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)

    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)

    # Run BDF2 with 2 known fluxes
    flux = multi_group_bdf2_restart(flux_1.copy(), flux_2.copy(), xs_total_u, \
                xs_total_c, xs_matrix_u, xs_matrix_c, velocity_u, velocity_c, \
                external_u, boundary_xu.copy(), boundary_yu.copy(), medium_map, \
                delta_x, delta_y, angle_xu, angle_xc, angle_yu, angle_yc, \
                angle_wu, angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)

    return np.asarray(flux)


cdef double[:,:,:,:] multi_group_bdf2_restart(double[:,:,:,:]& flux_last_1, \
        double[:,:,:,:]& flux_last_2, double[:,:]& xs_total_u, \
        double[:,:]& xs_total_c, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:,:,:,:,:]& external_u, \
        double[:,:,:,:,:]& boundary_xu, double[:,:,:,:,:]& boundary_yu, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_yu, \
        double[:]& angle_yc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        params info_u, params info_c):

    # Initialize time step, external and boundary indices
    cdef int step, qq, bcx, bcy

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.5, info_c)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    tools._angular_to_scalar(flux_last_1, flux_u, angle_wu, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_4d(info_c.cells_x, info_c.cells_y, 1, info_c.groups)
    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF2*    ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bcx = 0 if boundary_xu.shape[0] == 1 else step
        bcy = 0 if boundary_yu.shape[0] == 1 else step

        # Run BDF2 on all other steps
        tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                     external_u[qq], velocity_u, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bcx], boundary_yu[bcy], boundary_c, \
                      medium_map, delta_x, delta_y, angle_xu, angle_xc, \
                      angle_yu, angle_yc, angle_wu, angle_wc, fine_idx, \
                      coarse_idx, factor, info_u, info_c)

        # Step 5: Update steps
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                        boundary_xu[bcx], boundary_yu[bcy], \
                                        medium_map, delta_x, delta_y, angle_xu, \
                                        angle_yu, angle_wu, info_u)
        tools._angular_to_scalar(flux_last_1, flux_time[step], angle_wu, info_u)

    return flux_time[:,:,:,:]


cdef double[:,:,:,:] tr_bdf2(double[:,:,:,:]& flux_ell_x, \
        double[:,:,:,:]& flux_ell_y, double[:,:]& xs_total_u, \
        double[:,:]& xs_total_c, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:,:,:,:,:]& external_u, \
        double[:,:,:,:,:]& boundary_xu, double[:,:,:,:,:]& boundary_yu, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_yu, \
        double[:]& angle_yc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        params info_u, params info_c, params info_edge):

    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, qqb, bcx, bcxa, bcy, bcya

    # Initialize gamma
    cdef double gamma = 0.5

    # Create sigma_t + 2 / (gamma * v * dt) - CN Step
    xs_total_vu_cn = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_cn[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_cn, velocity_u, 2.0 / gamma, info_u)

    xs_total_vc_cn = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc_cn[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc_cn, velocity_c, 2.0 / gamma, info_c)

    # Create sigma_t + (2 - gamma) / ((1 - gamma) * v * dt) - BDF2 Step
    xs_total_vu_bdf2 = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_bdf2[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_bdf2, velocity_u, \
                            (2.0 - gamma) / (1.0 - gamma), info_u)

    xs_total_vc_bdf2 = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc_bdf2[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc_bdf2, velocity_c, \
                            (2.0 - gamma) / (1.0 - gamma), info_c)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                               info_u.angles * info_u.angles, info_u.groups)

    # Initialize scalar fluxes - Uncollided
    flux_ell_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    tools._angular_edge_to_scalar(flux_ell_x, flux_ell_y, flux_ell_u, angle_wu, info_u)
    flux_gamma_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize scalar fluxes - Collided
    flux_ell_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)
    flux_gamma_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_4d(info_c.cells_x, info_c.cells_y, 1, info_c.groups)
    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="TR-BDF2* ", ascii=True):

        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step * 2 # Ell Step
        qqa = 0 if external_u.shape[0] == 1 else step * 2 + 1 # Gamma Step
        qqb = 0 if external_u.shape[0] == 1 else step * 2 + 2 # Ell + 1 Step

        bcx = 0 if boundary_xu.shape[0] == 1 else step * 2 # Ell Step
        bcxa = 0 if boundary_xu.shape[0] == 1 else step * 2 + 1 # Gamma Step

        bcy = 0 if boundary_yu.shape[0] == 1 else step * 2 # Ell Step
        bcya = 0 if boundary_yu.shape[0] == 1 else step * 2 + 1 # Gamma Step

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn(flux_ell_x, flux_ell_y, flux_ell_u, \
                xs_total_u, xs_scatter_u, velocity_u, q_star, external_u[qq], \
                external_u[qqa], medium_map, delta_x, delta_y, angle_xu, \
                angle_yu, 2.0 / gamma, info_u)

        # Run hybrid method
        hybrid_method(flux_gamma_u, flux_gamma_c, xs_total_vu_cn, xs_total_vc_cn, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bcx], boundary_yu[bcy], boundary_c, \
                      medium_map, delta_x, delta_y, angle_xu, angle_xc, \
                      angle_yu, angle_yc, angle_wu, angle_wc, fine_idx, \
                      coarse_idx, factor, info_u, info_c)

        # Solve for angular flux of time step \ell + gamma
        flux_last_gamma = mg._known_source_angular(xs_total_vu_cn, q_star, \
                        boundary_xu[bcx], boundary_yu[bcy], medium_map, \
                        delta_x, delta_y, angle_xu, angle_yu, angle_wu, info_u)

        ################################################################
        # BDF2
        ################################################################
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_ell_x, flux_ell_y, flux_last_gamma, \
                        q_star, external_u[qqb], velocity_u, gamma, info_u)

        # Run hybrid method
        hybrid_method(flux_ell_u, flux_ell_c, xs_total_vu_bdf2, \
                xs_total_vc_bdf2, xs_scatter_u, xs_scatter_c, q_star, \
                source_c, boundary_xu[bcxa], boundary_yu[bcya], boundary_c, \
                medium_map, delta_x, delta_y, angle_xu, angle_xc, angle_yu, \
                angle_yc, angle_wu, angle_wc, fine_idx, coarse_idx, factor, \
                info_u, info_c)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_ell_x, flux_ell_y, xs_total_vu_bdf2, q_star, \
                boundary_xu[bcxa], boundary_yu[bcya], medium_map, \
                delta_x, delta_y, angle_xu, angle_yu, angle_wu, info_edge)

        # Step 5: Update flux_time and repeat
        tools._angular_edge_to_scalar(flux_ell_x, flux_ell_y, \
                                      flux_time[step], angle_wu, info_u)
        flux_ell_u[:,:,:] = flux_time[step,:,:,:]

    return flux_time[:,:,:,:]


cdef void hybrid_method(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        double[:,:]& xs_total_vu, double[:,:]& xs_total_vc, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:,:,:,:]& q_star, double[:,:,:,:]& source_c, \
        double[:,:,:,:]& boundary_xu, double[:,:,:,:]& boundary_yu, \
        double[:,:,:,:]& boundary_c, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_yu, double[:]& angle_yc, \
        double[:]& angle_wu, double[:]& angle_wc, int[:]& fine_idx, \
        int[:]& coarse_idx, double[:]& factor, params info_u, params info_c):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
    flux_u[:,:,:] = mg._known_source_scalar(xs_total_vu, q_star, boundary_xu, \
                        boundary_yu, medium_map, delta_x, delta_y, \
                        angle_xu, angle_yu, angle_wu, info_u)

    # Step 2: Compute collided source (I x G')
    tools._hybrid_source_collided(flux_u, xs_scatter_u, source_c, \
                                  medium_map, coarse_idx, info_u)

    # Step 3: Solve Collided Equation (I x G')
    flux_c[:,:,:] = mg.multi_group(flux_c, xs_total_vc, xs_scatter_c, \
                    source_c, boundary_c, boundary_c, medium_map, delta_x, \
                    delta_y, angle_xc, angle_yc, angle_wc, info_c)

    # Step 4: Create a new source and solve for angular flux
    tools._hybrid_source_total(flux_u, flux_c, xs_scatter_u, q_star, \
                        medium_map, coarse_idx, factor, info_u, info_c)
