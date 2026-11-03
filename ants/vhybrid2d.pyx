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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np
from tqdm.auto import tqdm

from libc.math cimport sqrt

import ants

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters
from ants.cytools_1d cimport _vhybrid_velocity, int_array_1d
from ants.parameters cimport params

from ants.datatypes import TemporalDiscretization, create_params
from ants.quadrature import _harmonic_data, _harmonic_order, _scatter_harmonics
from ants.timed2d import _harmonic_arrays
from ants.utils.sources import TimeSource

# Uncollided is fine grid (N^2 x G)
# Collided is coarse grid (N'^2 x G')
# Scattering is carried as spherical harmonic moments, see timed2d


def time_dependent(materials_u, groups_c, sources, geometry, quadrature_u, \
        angles_c, solver, time_data, edges_g):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total_u = materials_u.total
    cdef double[:,:,:,:] xs_scatter_u = _scatter_harmonics(materials_u.scatter)
    cdef double[:,:,:] xs_fission_u = tools._fission_matrix(materials_u.fission, materials_u.chi)
    cdef double[:] velocity_u = materials_u.velocity
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_xu = quadrature_u.angle_x
    cdef double[:] angle_yu = quadrature_u.angle_y
    cdef double[:] angle_wu = quadrature_u.angle_w

    # Convert uncollided dictionary to type params
    params_u = create_params(materials_u, quadrature_u, geometry, solver, time_data)
    info_u = parameters._to_params(params_u)

    # Sources of each time step, the boundaries are expanded to all angles
    external_u = TimeSource(sources.external)
    boundary_xu = TimeSource(sources.boundary_x, quadrature_u.angle_x, \
            (2, info_u.cells_y, info_u.angles**2, info_u.groups), "boundary_x")
    boundary_yu = TimeSource(sources.boundary_y, quadrature_u.angle_y, \
            (2, info_u.cells_x, info_u.angles**2, info_u.groups), "boundary_y")

    cdef double[:,:,:,:] initial_flux_x
    cdef double[:,:,:,:] initial_flux_y
    cdef double[:,:,:,:] initial_flux
    if time_data.time_disc == TemporalDiscretization.CN or time_data.time_disc == TemporalDiscretization.TR_BDF2:
        initial_flux_x = sources.initial_flux_x
        initial_flux_y = sources.initial_flux_y
    else:
        initial_flux = sources.initial_flux

    # Convert collided dictionary to type params
    params_c = create_params(materials_u, quadrature_u, geometry, solver, time_data)
    params_c.groups = groups_c[0]
    params_c.angles = angles_c[0]
    info_c = parameters._to_params(params_c)

    # Spherical harmonics for the scattering source and flux moments
    P_source_u_np, P_weights_u_np = _harmonic_arrays(materials_u, quadrature_u)
    cdef double[:,:] P_source_u = P_source_u_np
    cdef double[:,:] P_weights_u = P_weights_u_np

    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_4d(info_u.materials, info_u.groups, info_u.groups, \
                                 info_u.moments)
    tools._xs_matrix_aniso(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)

    # Optionally create a memmap file to stream per-step output
    flux_file = None
    if time_data.save_to_file is not None:
        flux_file = np.lib.format.open_memmap(
            time_data.save_to_file, mode='w+', dtype=np.float64,
            shape=(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)
        )

    if params_u.time_disc == TemporalDiscretization.BDF1:
        # Run Backward Euler
        parameters._check_bdf_timed2d(info_u, initial_flux.shape[0], external_u.steps, \
                    boundary_xu.steps, boundary_yu.steps, xs_total_u.shape[0])
        flux_final = backward_euler(groups_c, angles_c, initial_flux.copy(), xs_total_u,
                        xs_matrix_u, velocity_u, external_u, boundary_xu, \
                        boundary_yu, medium_map, delta_x, delta_y, angle_xu, \
                        angle_yu, angle_wu, edges_g, P_source_u, P_weights_u, \
                        flux_file, info_u, info_c)
    elif params_u.time_disc == TemporalDiscretization.CN:
        # Run Crank Nicolson method
        parameters._check_cn_timed2d(info_u, initial_flux_x.shape[0], \
                    initial_flux_y.shape[1], external_u.steps, boundary_xu.steps, \
                    boundary_yu.steps, xs_total_u.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        flux_final = crank_nicolson(groups_c, angles_c, initial_flux_x.copy(), \
                        initial_flux_y.copy(), xs_total_u, xs_matrix_u, velocity_u, \
                        external_u, boundary_xu, boundary_yu, \
                        medium_map, delta_x, delta_y, angle_xu, angle_yu, angle_wu, \
                        edges_g, P_source_u, P_weights_u, flux_file, info_u, \
                        info_c, info_edge)
    elif params_u.time_disc == TemporalDiscretization.BDF2:
        # Run BDF2 method
        parameters._check_bdf_timed2d(info_u, initial_flux.shape[0], external_u.steps, \
                    boundary_xu.steps, boundary_yu.steps, xs_total_u.shape[0])
        flux_final = bdf2(groups_c, angles_c, initial_flux.copy(), xs_total_u, \
                    xs_matrix_u, velocity_u, external_u, boundary_xu, \
                    boundary_yu, medium_map, delta_x, delta_y, angle_xu, \
                    angle_yu, angle_wu, edges_g, P_source_u, P_weights_u, \
                    flux_file, info_u, info_c)

    elif params_u.time_disc == TemporalDiscretization.TR_BDF2:
        # Run TR-BDF2 method
        parameters._check_tr_bdf_timed2d(info_u, initial_flux_x.shape[0], \
                    initial_flux_y.shape[1], external_u.steps, boundary_xu.steps, \
                    boundary_yu.steps, xs_total_u.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        flux_final = tr_bdf2(groups_c, angles_c, initial_flux_x.copy(), \
                        initial_flux_y.copy(), xs_total_u, xs_matrix_u, velocity_u, \
                        external_u, boundary_xu, boundary_yu, \
                        medium_map, delta_x, delta_y, angle_xu, angle_yu, angle_wu, \
                        edges_g, P_source_u, P_weights_u, flux_file, info_u, \
                        info_c, info_edge)

    return np.asarray(flux_final)


cdef double[:,:,:] backward_euler(int[:] groups_c, int[:] angles_c, \
        double[:,:,:,:]& flux_last, double[:,:]& xs_total_u, double[:,:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, object external_u, object boundary_xu, \
        object boundary_yu, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_xu, double[:]& angle_yu, double[:]& angle_wu,
        double[:]& edges_g, double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        object flux_file, params info_u, params info_c):

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize flux moments
    flux_u = tools.array_4d(info_u.cells_x, info_u.cells_y, info_u.groups, \
                            info_u.moments)
    tools._angular_to_moments(flux_last, flux_u, P_weights_u, info_u)
    flux_out = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)

    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vBDF1*   ", ascii=True):

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_yc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_yc, angle_wc = ants._angular_xy(info_c.angles, info_c.bc_x, info_c.bc_y)
            P_source_c, P_weights_c = _harmonic_data(angle_xc, angle_yc, \
                                    angle_wc, _harmonic_order(info_c.moments))
            source_c = _collided_source(info_c)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_4d(info_c.cells_x, info_c.cells_y, info_c.groups, \
                                    info_c.moments)
            source_c = _collided_source(info_c)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(grid=info_u.groups, \
                                             energy_groups=info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            _vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 1.0, info_c)

        ########################################################################
        # Boundary sources of the time step for all angles
        bcx_full = boundary_xu(step)
        bcy_full = boundary_yu(step)

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external_u(step), \
                                     velocity_u, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                q_star, source_c, bcx_full, bcy_full, boundary_c, medium_map, \
                delta_x, delta_y, angle_xu, angle_xc, angle_yu, angle_yc, \
                angle_wu, angle_wc, star_coef_c, edges_g, edges_gidx_c, \
                P_source_u, P_weights_u, P_source_c, P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    bcx_full, bcy_full, medium_map, \
                                    delta_x, delta_y, angle_xu, angle_yu, angle_wu, \
                                    info_u)

        # Step 5: Update and repeat
        tools._angular_to_scalar(flux_last, flux_out, angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = flux_out

    return flux_out


cdef double[:,:,:] crank_nicolson(int[:] groups_c, int[:] angles_c, \
        double[:,:,:,:]& flux_last_x, double[:,:,:,:]& flux_last_y, double[:,:]& xs_total_u, \
        double[:,:,:,:]& xs_scatter_u, double[:]& velocity_u, object external_u, \
        object boundary_xu, object boundary_yu, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, double[:]& angle_xu, \
        double[:]& angle_yu, double[:]& angle_wu, double[:]& edges_g, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, object flux_file, \
        params info_u, params info_c, params info_edge):
    # flux_last_x = (cells_x + 1, cells_y, angles**2, groups) - x edges
    # flux_last_y = (cells_x, cells_y + 1, angles**2, groups) - y edges

    # Initialize time step
    cdef int step

    # Create sigma_t + 2 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 2.0, info_u)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize flux moments
    flux_u = tools.array_4d(info_u.cells_x, info_u.cells_y, info_u.groups, \
                            info_u.moments)
    tools._angular_edge_to_moments(flux_last_x, flux_last_y, flux_u, \
                                   P_weights_u, info_u)
    flux_out = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)

    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vCN*     ", ascii=True):

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_yc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_yc, angle_wc = ants._angular_xy(info_c.angles, info_c.bc_x, info_c.bc_y)
            P_source_c, P_weights_c = _harmonic_data(angle_xc, angle_yc, \
                                    angle_wc, _harmonic_order(info_c.moments))
            source_c = _collided_source(info_c)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_4d(info_c.cells_x, info_c.cells_y, info_c.groups, \
                                    info_c.moments)
            source_c = _collided_source(info_c)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(grid=info_u.groups, \
                                             energy_groups=info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            _vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 2.0, info_c)

        ########################################################################
        # Boundary sources of the time step for all angles
        bcx_full = boundary_xu(step)
        bcy_full = boundary_yu(step)

        # Update q_star
        tools._time_source_star_cn_aniso(flux_last_x, flux_last_y, flux_u, \
                        xs_total_u, xs_scatter_u, velocity_u, q_star, \
                        external_u(step), external_u(step + 1), medium_map, \
                        delta_x, delta_y, angle_xu, angle_yu, P_source_u, \
                        2.0, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                q_star, source_c, bcx_full, bcy_full, boundary_c, medium_map, \
                delta_x, delta_y, angle_xu, angle_xc, angle_yu, angle_yc, \
                angle_wu, angle_wc, star_coef_c, edges_g, edges_gidx_c, \
                P_source_u, P_weights_u, P_source_c, P_weights_c, info_u, info_c)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_last_x, flux_last_y, xs_total_vu, \
                q_star, bcx_full, bcy_full, medium_map, \
                delta_x, delta_y, angle_xu, angle_yu, angle_wu, info_edge)

        # Step 5: Update and repeat
        tools._angular_edge_to_scalar(flux_last_x, flux_last_y, \
                                      flux_out, angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = flux_out
        tools._angular_edge_to_moments(flux_last_x, flux_last_y, flux_u, \
                                       P_weights_u, info_u)

    return flux_out


cdef double[:,:,:] bdf2(int[:]& groups_c, int[:]& angles_c, double[:,:,:,:]& flux_last_1, \
        double[:,:]& xs_total_u, double[:,:,:,:]& xs_scatter_u, double[:]& velocity_u, \
        object external_u, object boundary_xu, \
        object boundary_yu, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_xu, double[:]& angle_yu, double[:]& angle_wu, \
        double[:]& edges_g, double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        object flux_file, params info_u, params info_c):

    # Initialize time step
    cdef int step
    cdef double coef = 1.0

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Combine last time step and source term
    q_star = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_2 = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                                 info_u.angles * info_u.angles, info_u.groups)

    # Initialize flux moments
    flux_u = tools.array_4d(info_u.cells_x, info_u.cells_y, info_u.groups, \
                            info_u.moments)
    tools._angular_to_moments(flux_last_1, flux_u, P_weights_u, info_u)
    flux_out = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)

    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vBDF2*   ", ascii=True):

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_yc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_yc, angle_wc = ants._angular_xy(info_c.angles, info_c.bc_x, info_c.bc_y)
            P_source_c, P_weights_c = _harmonic_data(angle_xc, angle_yc, \
                                    angle_wc, _harmonic_order(info_c.moments))
            source_c = _collided_source(info_c)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_4d(info_c.cells_x, info_c.cells_y, info_c.groups, \
                                    info_c.moments)
            source_c = _collided_source(info_c)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(grid=info_u.groups, \
                                             energy_groups=info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            coef = 1.0 if step == 0 else 1.5
            _vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, coef, info_c)

        if (step == 1):
            star_coef_c = tools.array_1d(info_c.groups)
            _vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 1.5, info_c)

        ########################################################################
        # Boundary sources of the time step for all angles
        bcx_full = boundary_xu(step)
        bcy_full = boundary_yu(step)

        # Update q_star
        if step == 0:
            # Run BDF1 on first step
            tools._time_source_star_bdf1(flux_last_1, q_star, external_u(step), \
                                         velocity_u, info_u)
        else:
            # Run BDF2 on all other steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                         external_u(step), velocity_u, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                q_star, source_c, bcx_full, bcy_full, boundary_c, medium_map, \
                delta_x, delta_y, angle_xu, angle_xc, angle_yu, angle_yc, \
                angle_wu, angle_wc, star_coef_c, edges_g, edges_gidx_c, \
                P_source_u, P_weights_u, P_source_c, P_weights_c, info_u, info_c)

        # Step 5: Update steps
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1[:,:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                        bcx_full, bcy_full, \
                                        medium_map, delta_x, delta_y, angle_xu, \
                                        angle_yu, angle_wu, info_u)

        tools._angular_to_scalar(flux_last_1, flux_out, angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = flux_out

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_vu[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)

    return flux_out


cdef double[:,:,:] tr_bdf2(int[:] groups_c, int[:] angles_c, double[:,:,:,:]& flux_ell_x, \
        double[:,:,:,:]& flux_ell_y, double[:,:]& xs_total_u, double[:,:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, object external_u, object boundary_xu, \
        object boundary_yu, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_xu, double[:]& angle_yu, double[:]& angle_wu, \
        double[:]& edges_g, double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        object flux_file, params info_u, params info_c, params info_edge):

    # Initialize time step
    cdef int step

    # Initialize gamma
    cdef double gamma = 0.5
    cdef double coef = (2.0 - gamma) / (1.0 - gamma)

    # Create sigma_t + 2 / (gamma * v * dt) - CN Step
    xs_total_vu_cn = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_cn[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_cn, velocity_u, 2.0 / gamma, info_u)

    # Create sigma_t + (2 - gamma) / ((1 - gamma) * v * dt) - BDF2 Step
    xs_total_vu_bdf2 = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_bdf2[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_bdf2, velocity_u, coef, info_u)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                               info_u.angles * info_u.angles, info_u.groups)

    # Initialize flux moments - Uncollided
    flux_u = tools.array_4d(info_u.cells_x, info_u.cells_y, info_u.groups, \
                            info_u.moments)
    tools._angular_edge_to_moments(flux_ell_x, flux_ell_y, flux_u, P_weights_u, info_u)
    flux_out = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)

    boundary_c = tools.array_4d(2, 1, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vTR-BDF2*", ascii=True):



        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_yc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_yc, angle_wc = ants._angular_xy(info_c.angles, info_c.bc_x, info_c.bc_y)
            P_source_c, P_weights_c = _harmonic_data(angle_xc, angle_yc, \
                                    angle_wc, _harmonic_order(info_c.moments))
            source_c = _collided_source(info_c)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_4d(info_c.cells_x, info_c.cells_y, info_c.groups, \
                                    info_c.moments)
            source_c = _collided_source(info_c)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(grid=info_u.groups, \
                                             energy_groups=info_c.groups)

            star_coef_cn_c = tools.array_1d(info_c.groups)
            _vhybrid_velocity(star_coef_cn_c, velocity_u, edges_gidx_c, 2.0 / gamma, info_c)

            star_coef_bdf2_c = tools.array_1d(info_c.groups)
            _vhybrid_velocity(star_coef_bdf2_c, velocity_u, edges_gidx_c, coef, info_c)

        ########################################################################
        # Boundary sources of the time step for all angles (Ell and Gamma steps)
        bcx_full = boundary_xu(step * 2)
        bcy_full = boundary_yu(step * 2)
        bcxa_full = boundary_xu(step * 2 + 1)
        bcya_full = boundary_yu(step * 2 + 1)

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn_aniso(flux_ell_x, flux_ell_y, flux_u, xs_total_u, \
                xs_scatter_u, velocity_u, flux_last_gamma, external_u(step * 2), external_u(step * 2 + 1), \
                medium_map, delta_x, delta_y, angle_xu, angle_yu, P_source_u, \
                2.0 / gamma, info_u)

        # Run hybrid method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu_cn, xs_scatter_u, \
                flux_last_gamma, source_c, bcx_full, bcy_full, boundary_c, \
                medium_map, delta_x, delta_y, angle_xu, angle_xc, angle_yu, \
                angle_yc, angle_wu, angle_wc, star_coef_cn_c, edges_g, \
                edges_gidx_c, P_source_u, P_weights_u, P_source_c, P_weights_c, \
                info_u, info_c)

        # Solve for angular flux of time step \ell + gamma
        flux_last_gamma = mg._known_source_angular(xs_total_vu_cn, flux_last_gamma, \
                        bcx_full, bcy_full, medium_map, \
                        delta_x, delta_y, angle_xu, angle_yu, angle_wu, info_u)

        ################################################################
        # BDF2
        ################################################################
        tools._time_source_star_tr_bdf2_mem(flux_ell_x, flux_ell_y, flux_last_gamma, \
                            external_u(step * 2 + 2), velocity_u, gamma, info_u)

        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu_bdf2, xs_scatter_u, \
                flux_last_gamma, source_c, bcxa_full, bcya_full, boundary_c, \
                medium_map, delta_x, delta_y, angle_xu, angle_xc, angle_yu, \
                angle_yc, angle_wu, angle_wc, star_coef_bdf2_c, edges_g, \
                edges_gidx_c, P_source_u, P_weights_u, P_source_c, P_weights_c, \
                info_u, info_c)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_ell_x, flux_ell_y, xs_total_vu_bdf2, flux_last_gamma, \
                bcxa_full, bcya_full, medium_map, delta_x, delta_y, \
                angle_xu, angle_yu, angle_wu, info_edge)

        # Step 5: Update and repeat
        tools._angular_edge_to_scalar(flux_ell_x, flux_ell_y, flux_out, \
                                      angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = flux_out
        tools._angular_edge_to_moments(flux_ell_x, flux_ell_y, flux_u, \
                                       P_weights_u, info_u)

    return flux_out


cdef double[:,:,:,:] _collided_source(params info_c):
    # Isotropic collided source is the same for every collided ordinate
    cdef int angles = 1 if info_c.moments == 1 else info_c.angles * info_c.angles
    return tools.array_4d(info_c.cells_x, info_c.cells_y, angles, info_c.groups)


cdef void hybrid_method(double[:,:,:,:]& flux_u, double[:,:,:,:]& flux_c, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_vu, \
        double[:,:,:,:]& xs_scatter_u, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& source_c, double[:,:,:,:]& boundary_xu, \
        double[:,:,:,:]& boundary_yu, double[:,:,:,:]& boundary_c, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_yu, \
        double[:]& angle_yc, double[:]& angle_wu, double[:]& angle_wc, \
        double[:]& star_coef_c, double[:]& edges_g, int[:]& edges_gidx_c, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        params info_u, params info_c):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G x K)
    flux_u[:,:,:,:] = mg._known_source_moments(xs_total_vu, q_star, boundary_xu, \
                            boundary_yu, medium_map, delta_x, delta_y, angle_xu, \
                            angle_yu, angle_wu, P_weights_u, info_u)

    # Step 2: Compute collided source (I x J x N'^2 x G')
    tools._vhybrid_source_c_aniso(flux_u, xs_scatter_u, source_c, medium_map, \
                            edges_gidx_c, P_source_c, info_u, info_c)

    # Step 3: Solve Collided Equation (I x J x G' x K)
    tools._coarsen_moments(flux_u, flux_c, edges_gidx_c, info_c)
    flux_c[:,:,:,:] = mg.variable_source_iteration_moments(flux_c, xs_total_u, \
                            star_coef_c, xs_scatter_u, source_c, boundary_c, \
                            boundary_c, medium_map, delta_x, delta_y, angle_xc, \
                            angle_yc, angle_wc, edges_g, edges_gidx_c, P_source_c, \
                            P_weights_c, info_c)

    # Step 4: Create a new source and solve for angular flux
    tools._vhybrid_source_total_aniso(flux_u, flux_c, xs_scatter_u, q_star, \
                    medium_map, edges_g, edges_gidx_c, P_source_u, info_u, info_c)
