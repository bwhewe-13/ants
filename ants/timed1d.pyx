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
from ants.fixed1d import known_flux_moments as steady_state_moments
from ants.quadrature import legendre_data
from ants.utils.sources import TimeSource

# Scattering is carried as Legendre moments, (materials, groups, groups,
# moments) for xs_scatter and (cells_x, groups, moments) for the flux.
# Isotropic scattering has one moment and runs the scalar flux solvers.


def time_dependent(materials, sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_moments(materials.scatter)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:] velocity = materials.velocity
    cdef double[:,:,:] initial_flux = sources.initial_flux
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    # Legendre polynomials for the scattering source and flux moments
    P_source_np, P_weights_np = legendre_data(quadrature, info.moments)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Add fission matrix to scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Sources of each time step, the boundary is expanded to all angles
    external = TimeSource(sources.external)
    boundary_x = TimeSource(sources.boundary_x, quadrature.angle_x, \
                    (2, info.angles, info.groups), "boundary_x")

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
        parameters._check_bdf_timed1d(info, initial_flux.shape[0], external.steps, \
                                boundary_x.steps, xs_total.shape[0])
        flux_final = backward_euler(initial_flux.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x, medium_map, \
                        delta_x, angle_x, angle_w, P_source, P_weights, \
                        flux_file, info)
    elif params.time_disc == TemporalDiscretization.CN:
        # Run Crank Nicolson method
        parameters._check_cn_timed1d(info, initial_flux.shape[0], \
                    external.steps, boundary_x.steps, xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1
        flux_final = crank_nicolson(initial_flux.copy(), xs_total, xs_matrix, \
                         velocity, external, boundary_x, medium_map, \
                         delta_x, angle_x, angle_w, P_source, P_weights, \
                         flux_file, info, info_edge)
    elif params.time_disc == TemporalDiscretization.BDF2:
        # Run BDF2 method
        parameters._check_bdf_timed1d(info, initial_flux.shape[0], \
                    external.steps, boundary_x.steps, xs_total.shape[0])
        flux_final = bdf2(initial_flux.copy(), xs_total, xs_matrix, \
                           velocity, external, boundary_x, \
                           medium_map, delta_x, angle_x, angle_w, P_source, \
                           P_weights, flux_file, info)
    elif params.time_disc == TemporalDiscretization.TR_BDF2:
        # Run TR-BDF2 method
        parameters._check_tr_bdf_timed1d(info, initial_flux.shape[0], \
                external.steps, boundary_x.steps, xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux_final = tr_bdf2(initial_flux.copy(), xs_total, xs_matrix, \
                        velocity, external, boundary_x, medium_map, \
                        delta_x, angle_x, angle_w, P_source, P_weights, \
                        flux_file, info, info_edge)

    return np.asarray(flux_final)


def _scatter_moments(scatter):
    # Isotropic scattering as a single Legendre moment (no copy)
    scatter = np.asarray(scatter)
    return scatter if scatter.ndim == 4 else scatter[..., None]


cdef double[:,:] backward_euler(double[:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        object flux_file, params info):

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.groups, info.moments)
    tools._angular_to_moments(flux_last, scalar_flux, P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF1    ", ascii=True):
        # Boundary source of the time step for all angles
        bc_full = boundary_x(step)

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external(step), \
                                     velocity, info)

        # Solve for the current time step
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, xs_scatter, \
                                   q_star, bc_full, medium_map, delta_x, \
                                   angle_x, angle_w, P_source, P_weights, info)
        scalar_flux[:,:,:] = mg_result[:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        mg._known_source_angular(flux_last, xs_total_v, q_star, \
                                    bc_full, medium_map, \
                                    delta_x, angle_x, angle_w, info)

    return scalar_flux[:,:,0]


cdef double[:,:] crank_nicolson(double[:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        object flux_file, params info, params info_edge):

    # Initialize time step
    cdef int step

    # Create sigma_t + 2 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 2.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.groups, info.moments)
    tools._angular_edge_to_moments(flux_last, scalar_flux, P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="CN      ", ascii=True):

        # Boundary source of the time step for all angles
        bc_full = boundary_x(step)

        # Update q_star with the previous and current external sources
        tools._time_source_star_cn_aniso(flux_last, scalar_flux, xs_total, \
                        xs_scatter, velocity, q_star, external(step), \
                        external(step + 1), medium_map, delta_x, angle_x, \
                        P_source, 2.0, info)
        # Solve for the current time step
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_full, medium_map, \
                                delta_x, angle_x, angle_w, P_source, \
                                P_weights, info)
        scalar_flux[:,:,:] = mg_result[:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,0])
        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)
        # Solve for angular flux of previous time step
        mg._known_source_angular(flux_last, xs_total_v, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info_edge)

    return scalar_flux[:,:,0]


cdef double[:,:] bdf2(double[:,:,:]& flux_last_1, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        object flux_file, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step
    cdef int step
    cdef double[:,:,:] flux_swap

    # Combine total cross section and time coefficient (BDF1 time step)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.groups, info.moments)
    tools._angular_to_moments(flux_last_1, scalar_flux, P_weights, info)

    # Create angular flux of previous time steps - Initialize with initial
    flux_last_2 = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):

        # Boundary source of the time step for all angles
        bc_full = boundary_x(step)

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, \
                                    external(step), velocity, info)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                        external(step), velocity, info)

        # Solve for the current time step
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_full, medium_map, \
                                delta_x, angle_x, angle_w, P_source, \
                                P_weights, info)
        scalar_flux[:,:,:] = mg_result[:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        # flux_last_2 is not needed after q_star, so it is overwritten
        # with the new step and swapped with flux_last_1 (no copy)
        mg._known_source_angular(flux_last_2, xs_total_v, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info)
        flux_swap = flux_last_1
        flux_last_1 = flux_last_2
        flux_last_2 = flux_swap

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_v[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v, velocity, 1.5, info)

    return scalar_flux[:,:,0]


cdef double[:,:] tr_bdf2(double[:,:,:]& flux_last_ell, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        object flux_file, params info, params info_edge):

    # Initialize time step
    cdef int step

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

    # Initialize flux moments for previous time step
    scalar_flux_ell = tools.array_3d(info.cells_x, info.groups, info.moments)
    tools._angular_edge_to_moments(flux_last_ell, scalar_flux_ell, P_weights, info)
    scalar_flux_gamma = tools.array_3d(info.cells_x, info.groups, info.moments)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="TR-BDF2 ", ascii=True):

        # Boundary sources of the ell and gamma steps for all angles
        bc_full = boundary_x(step * 2)
        bca_full = boundary_x(step * 2 + 1)

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn_aniso(flux_last_ell, scalar_flux_ell, xs_total, \
                        xs_scatter, velocity, q_star, external(step * 2), \
                        external(step * 2 + 1), medium_map, delta_x, angle_x, \
                        P_source, 2.0 / gamma, info)

        # Solve for the \ell + gamma time step
        scalar_flux_gamma[:,:,:] = mg.multi_group_moments(
            scalar_flux_ell,
            xs_total_v_cn,
            xs_scatter,
            q_star,
            bc_full,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            P_source,
            P_weights,
            info,
        )

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux_gamma, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        mg._known_source_angular(flux_last_gamma, xs_total_v_cn, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_x, angle_w, info)
        ################################################################
        # BDF2
        ################################################################
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_last_ell, flux_last_gamma, \
                    q_star, external(step * 2 + 2), velocity, gamma, info)

        # Solve for the \ell + 1 time step
        mg_result = mg.multi_group_moments(
            scalar_flux_gamma,
            xs_total_v_bdf2,
            xs_scatter,
            q_star,
            bca_full,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            P_source,
            P_weights,
            info,
        )
        scalar_flux_ell[:,:,:] = mg_result[:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux_ell[:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux_ell, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        mg._known_source_angular(flux_last_ell, xs_total_v_bdf2, \
                                    q_star, bca_full, medium_map, \
                                    delta_x, angle_x, angle_w, info_edge)

    return scalar_flux_ell[:,:,0]


def known_source_calculation(flux, materials, sources, geometry, \
        quadrature, solver, time):
    # flux is the scalar flux (steps, cells_x, groups), or the flux moments
    # (steps, cells_x, groups, moments) for anisotropic scattering
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)

    # Initialize iterables
    cdef int step

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver, time)
    info = parameters._to_params(params)

    # Anisotropic scattering needs every flux moment
    anisotropic = (info.moments > 1)
    flux = np.asarray(flux)
    if anisotropic and (flux.ndim != 4):
        raise ValueError("Anisotropic scattering needs the flux moments, shape " \
                         "(steps, cells_x, groups, moments)")

    # Add fission matrix to scattering
    if anisotropic:
        xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, \
                                   info.moments)
        tools._xs_matrix_aniso(xs_matrix, materials.scatter, xs_fission, info)
        # Legendre polynomials for the scattering source
        P_source, _ = legendre_data(quadrature, info.moments)
    else:
        xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
        tools._xs_matrix(xs_matrix, materials.scatter, xs_fission, info)

    # Initialize angular flux (numpy array, each step is filled by an array)
    angular_flux = np.zeros((info.steps, info.cells_x, info.angles, info.groups))

    # Sources of each time step, the boundary is expanded to all angles
    external = TimeSource(sources.external)
    boundary_x = TimeSource(sources.boundary_x, quadrature.angle_x, \
                    (2, info.angles, info.groups), "boundary_x")

    for step in range(info.steps):
        # Boundary source of the time step for all angles
        bc_full = boundary_x(step)

        if anisotropic:
            angular_flux[step] = steady_state_moments(flux[step], xs_total, \
                                        xs_matrix, external(step), bc_full, \
                                        P_source, geometry, quadrature, params)
        else:
            angular_flux[step] = steady_state(flux[step], xs_total, xs_matrix, \
                                        external(step), bc_full, geometry, \
                                        quadrature, params)

    return angular_flux
