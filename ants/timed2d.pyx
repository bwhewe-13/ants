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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np
from tqdm.auto import tqdm

from libc.math cimport sqrt

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import TemporalDiscretization, create_params
from ants.quadrature import _scatter_harmonics, harmonic_data
from ants.utils.pytools import (
    artificial_scatter_setup,
    artificial_scatter_source,
    edge_to_center_angular,
)
from ants.utils.sources import TimeSource

# Scattering is carried as spherical harmonic moments, (materials, groups,
# groups, moments) for xs_scatter and (cells_x, cells_y, groups, moments)
# for the flux. Isotropic scattering has one moment and runs the scalar
# flux solvers.


def _harmonic_arrays(materials, quadrature):
    # Spherical harmonics for the scattering source and flux moments
    scatter = np.asarray(materials.scatter)
    n_legendre = scatter.shape[3] if scatter.ndim == 4 else 1
    return harmonic_data(quadrature, n_legendre)


def time_dependent(materials, sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_harmonics(materials.scatter)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:] velocity = materials.velocity
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    # Sources of each time step, the boundaries are expanded to all angles
    external = TimeSource(sources.external)
    boundary_x = TimeSource(sources.boundary_x, quadrature.angle_x, \
            (2, info.cells_y, info.angles**2, info.groups), "boundary_x")
    boundary_y = TimeSource(sources.boundary_y, quadrature.angle_y, \
            (2, info.cells_x, info.angles**2, info.groups), "boundary_y")

    # Spherical harmonics for the scattering source and flux moments
    P_source_np, P_weights_np = _harmonic_arrays(materials, quadrature)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    cdef double[:,:,:,:] initial_flux_x
    cdef double[:,:,:,:] initial_flux_y
    cdef double[:,:,:,:] initial_flux
    if time_data.time_disc == TemporalDiscretization.CN or time_data.time_disc == TemporalDiscretization.TR_BDF2:
        initial_flux_x = sources.initial_flux_x
        initial_flux_y = sources.initial_flux_y
    else:
        initial_flux = sources.initial_flux

    # Combine fission and scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Optionally create a memmap file to stream per-step output
    cdef int steps = info.steps
    cdef int cells_x = info.cells_x
    cdef int cells_y = info.cells_y
    cdef int groups = info.groups
    flux_file = None
    if time_data.save_to_file is not None:
        flux_file = np.lib.format.open_memmap(
            time_data.save_to_file, mode='w+', dtype=np.float64,
            shape=(steps, cells_x, cells_y, groups)
        )

    if params.time_disc == TemporalDiscretization.BDF1:
        parameters._check_bdf_timed2d(info, initial_flux.shape[0], external.steps, \
                        boundary_x.steps, boundary_y.steps, xs_total.shape[0])
        # Run Backward Euler
        flux_final = backward_euler(initial_flux.copy(), xs_total, xs_matrix, velocity, \
                    external, boundary_x, boundary_y, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, P_source, \
                    P_weights, flux_file, info)

    elif params.time_disc == TemporalDiscretization.CN:
        parameters._check_cn_timed2d(info, initial_flux_x.shape[0], \
                    initial_flux_y.shape[1], external.steps, boundary_x.steps, \
                    boundary_y.steps, xs_total.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux_final = crank_nicolson(initial_flux_x.copy(), initial_flux_y.copy(), \
                    xs_total, xs_matrix, velocity, external, boundary_x, \
                    boundary_y, medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, P_source, P_weights, flux_file, info, info_edge)

    elif params.time_disc == TemporalDiscretization.BDF2:
        parameters._check_bdf_timed2d(info, initial_flux.shape[0], \
                                    external.steps, boundary_x.steps, \
                                    boundary_y.steps, xs_total.shape[0])
        # Run BDF2
        flux_final = bdf2(initial_flux.copy(), xs_total, xs_matrix, velocity, external, \
                    boundary_x, boundary_y, medium_map, delta_x, \
                    delta_y, angle_x, angle_y, angle_w, P_source, P_weights, \
                    flux_file, info)

    elif params.time_disc == TemporalDiscretization.TR_BDF2:
        parameters._check_tr_bdf_timed2d(info, initial_flux_x.shape[0], \
                initial_flux_y.shape[1], external.steps, boundary_x.steps, \
                boundary_y.steps, xs_total.shape[0])

        # Create params with edges for TR/BDF2 method
        info_edge = parameters._to_params(params)
        info_edge.flux_at_edges = 1

        flux_final = tr_bdf2(initial_flux_x.copy(), initial_flux_y.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, angle_x, \
                        angle_y, angle_w, P_source, P_weights, flux_file, info, \
                        info_edge)

    return np.asarray(flux_final)


cdef double[:,:,:] backward_euler(double[:,:,:,:]& flux_last, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:]& velocity, object external, \
        object boundary_x, object boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        double[:,:]& P_source, double[:,:]& P_weights, object flux_file, \
        params info):

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Artificial scattering (as-SN): sigma_as out-scattering on the LHS,
    # lagged in-scatter source M_as . psi^{ell} added to q_star below
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v)[:] += info.sigma_as

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, \
                                 info.moments)
    tools._angular_to_moments(flux_last, scalar_flux, P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF1    ", ascii=True):
        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step)
        bc_y_full = boundary_y(step)

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external(step), \
                                     velocity, info)

        # Add lagged artificial in-scatter source M_as . psi^{ell}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, flux_last)

        # Run source iteration
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_x_full, \
                                bc_y_full, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, \
                                P_source, P_weights, info)
        scalar_flux[:,:,:,:] = mg_result[:,:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        mg._known_source_angular(flux_last, xs_total_v, q_star,
                                        bc_x_full, bc_y_full, \
                                        medium_map, delta_x, delta_y, \
                                        angle_x, angle_y, angle_w, info)

    return scalar_flux[:,:,:,0]


cdef double[:,:,:] crank_nicolson(double[:,:,:,:]& flux_last_x, \
        double[:,:,:,:]& flux_last_y, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        object boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, object flux_file, params info, \
        params info_edge):
    # flux_last_x = (cells_x + 1, cells_y, angles**2, groups) - x edges
    # flux_last_y = (cells_x, cells_y + 1, angles**2, groups) - y edges

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 2.0, info)

    # Artificial scattering (as-SN): implicit sigma_as out-scattering,
    # lagged in-scatter from the previous step's edge angular flux.
    # Factor 2 matches the doubled CN operator convention.
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v)[:] += 2.0 * info.sigma_as

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y,\
                            info.angles * info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, \
                                 info.moments)
    tools._angular_edge_to_moments(flux_last_x, flux_last_y, scalar_flux, \
                                   P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="CN      ", ascii=True):
        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step)
        bc_y_full = boundary_y(step)

        # Update q_star
        tools._time_source_star_cn_aniso(flux_last_x, flux_last_y, scalar_flux, \
                xs_total, xs_scatter, velocity, q_star, external(step), \
                external(step + 1), medium_map, delta_x, delta_y, angle_x, \
                angle_y, P_source, 2.0, info)

        # Add lagged artificial in-scatter source 2 * M_as . psi^{ell}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, \
                    edge_to_center_angular(flux_last_x, flux_last_y), 2.0)

        # Run source iteration
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                    xs_scatter, q_star, bc_x_full, \
                                    bc_y_full, medium_map, delta_x, \
                                    delta_y, angle_x, angle_y, angle_w, \
                                    P_source, P_weights, info)
        scalar_flux[:,:,:,:] = mg_result[:,:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,:,0])
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)
        # Solve for angular flux of previous time step
        mg._interface_angular(flux_last_x, flux_last_y, xs_total_v, q_star, \
                bc_x_full, bc_y_full, medium_map, \
                delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)

    return scalar_flux[:,:,:,0]


cdef double[:,:,:] bdf2(double[:,:,:,:]& flux_last_1, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        object boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, object flux_file, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step
    cdef int step
    cdef double[:,:,:,:] flux_swap

    # Create sigma_t + 1 / (v * dt) (BDF1 time step)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Artificial scattering (as-SN): sigma_as out-scattering on the LHS,
    # lagged in-scatter source M_as . psi^{ell-1} added to q_star below
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v)[:] += info.sigma_as

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, \
                                 info.moments)
    tools._angular_to_moments(flux_last_1, scalar_flux, P_weights, info)

    # Create angular flux of previous time steps
    flux_last_2 = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):
        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step)
        bc_y_full = boundary_y(step)

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, external(step), \
                                         velocity, info)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                         external(step), velocity, info)

        # Add lagged artificial in-scatter source M_as . psi^{ell-1}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, flux_last_1)

        # Run source iteration
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_x_full, \
                                bc_y_full, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, \
                                P_source, P_weights, info)
        scalar_flux[:,:,:,:] = mg_result[:,:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        # flux_last_2 is not needed after q_star, so it is overwritten
        # with the new step and swapped with flux_last_1 (no copy)
        mg._known_source_angular(flux_last_2, xs_total_v, q_star, \
                                    bc_x_full, bc_y_full, \
                                    medium_map, delta_x, delta_y, angle_x, \
                                    angle_y, angle_w, info)
        flux_swap = flux_last_1
        flux_last_1 = flux_last_2
        flux_last_2 = flux_swap

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_v[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v, velocity, 1.5, info)
            if M_as is not None:
                np.asarray(xs_total_v)[:] += info.sigma_as

    return scalar_flux[:,:,:,0]


def restart_bdf2(double[:,:,:,:] flux_1, double[:,:,:,:] flux_2, materials, \
        sources, geometry, quadrature, solver, time_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_harmonics(materials.scatter)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:] velocity = materials.velocity
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    # Sources of each time step, the boundaries are expanded to all angles
    external = TimeSource(sources.external)
    boundary_x = TimeSource(sources.boundary_x, quadrature.angle_x, \
            (2, info.cells_y, info.angles**2, info.groups), "boundary_x")
    boundary_y = TimeSource(sources.boundary_y, quadrature.angle_y, \
            (2, info.cells_x, info.angles**2, info.groups), "boundary_y")
    parameters._check_bdf_timed2d(info, flux_1.shape[0], external.steps, \
            boundary_x.steps, boundary_y.steps, xs_total.shape[0])

    # Spherical harmonics for the scattering source and flux moments
    P_source_np, P_weights_np = _harmonic_arrays(materials, quadrature)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Combine fission and scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Optionally create a memmap file to stream per-step output
    flux_file = None
    if time_data.save_to_file is not None:
        flux_file = np.lib.format.open_memmap(
            time_data.save_to_file, mode='w+', dtype=np.float64,
            shape=(info.steps, info.cells_x, info.cells_y, info.groups)
        )

    # Run BDF2 with 2 known fluxes
    flux_final = multi_group_bdf2_restart(flux_1.copy(), flux_2.copy(), xs_total, \
                        xs_matrix, velocity, external, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, P_source, P_weights, \
                        flux_file, info)

    return np.asarray(flux_final)


cdef double[:,:,:] multi_group_bdf2_restart(double[:,:,:,:]& flux_last_1, \
        double[:,:,:,:]& flux_last_2, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        object boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, object flux_file, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step
    cdef int step
    cdef double[:,:,:,:] flux_swap

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.5, info)

    # Artificial scattering (as-SN): sigma_as out-scattering on the LHS,
    # lagged in-scatter source M_as . psi^{ell-1} added to q_star below
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v)[:] += info.sigma_as

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, \
                                 info.moments)
    tools._angular_to_moments(flux_last_1, scalar_flux, P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF2    ", ascii=True):
        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step)
        bc_y_full = boundary_y(step)

        # Run BDF2 on rest of time steps
        tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                     external(step), velocity, info)

        # Add lagged artificial in-scatter source M_as . psi^{ell-1}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, flux_last_1)

        # Run source iteration
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, bc_x_full, \
                                bc_y_full, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, \
                                P_source, P_weights, info)
        scalar_flux[:,:,:,:] = mg_result[:,:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        # flux_last_2 is not needed after q_star, so it is overwritten
        # with the new step and swapped with flux_last_1 (no copy)
        mg._known_source_angular(flux_last_2, xs_total_v, q_star, \
                                    bc_x_full, bc_y_full, \
                                    medium_map, delta_x, delta_y, angle_x, \
                                    angle_y, angle_w, info)
        flux_swap = flux_last_1
        flux_last_1 = flux_last_2
        flux_last_2 = flux_swap

    return scalar_flux[:,:,:,0]


def angular_bdf2(int[:] time_steps, scalar_flux, double[:,:,:,:] initial_flux, \
        materials, sources, geometry, quadrature, solver, time_data):
    # scalar_flux is (steps, cells_x, cells_y, groups), or the flux moments
    # (steps, cells_x, cells_y, groups, moments) for anisotropic scattering
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_harmonics(materials.scatter)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:] velocity = materials.velocity
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    params = create_params(materials, quadrature, geometry, solver, time_data)
    info = parameters._to_params(params)

    # Sources of each time step, the boundaries are expanded to all angles
    external = TimeSource(sources.external)
    boundary_x = TimeSource(sources.boundary_x, quadrature.angle_x, \
            (2, info.cells_y, info.angles**2, info.groups), "boundary_x")
    boundary_y = TimeSource(sources.boundary_y, quadrature.angle_y, \
            (2, info.cells_x, info.angles**2, info.groups), "boundary_y")
    parameters._check_bdf_timed2d(info, initial_flux.shape[0], \
                                  external.steps, boundary_x.steps, \
                                  boundary_y.steps, xs_total.shape[0])

    # Anisotropic scattering needs every flux moment
    scalar_flux = np.asarray(scalar_flux)
    if info.moments > 1 and scalar_flux.ndim != 5:
        raise ValueError("Anisotropic scattering needs the flux moments, shape " \
                         "(steps, cells_x, cells_y, groups, moments)")
    if scalar_flux.ndim == 4:
        scalar_flux = scalar_flux[..., None]
    cdef double[:,:,:,:,:] flux_moments = scalar_flux.copy()

    # Spherical harmonics for the scattering source
    P_source_np, _ = _harmonic_arrays(materials, quadrature)
    cdef double[:,:] P_source = P_source_np

    # Combine fission and scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Run BDF2 with known scalar flux
    flux = multi_group_bdf2_angular(time_steps, flux_moments, initial_flux.copy(), \
                xs_total, xs_matrix, velocity, external, boundary_x, \
                boundary_y, medium_map, delta_x, delta_y, angle_x, \
                angle_y, angle_w, P_source, info)

    return np.asarray(flux)


cdef double[:,:,:,:,:] multi_group_bdf2_angular(int[:]& time_steps, \
        double[:,:,:,:,:]& scalar_flux, double[:,:,:,:]& flux_last_1, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        object boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, double[:,:]& P_source, params info):
    # flux_last_1 is \ell - 1, flux_last_2 is \ell - 2

    # Initialize time step
    cdef int step
    cdef double[:,:,:,:] flux_swap
    cdef int count= 0

    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Artificial scattering (as-SN): sigma_as out-scattering on the LHS,
    # lagged in-scatter source M_as . psi^{ell-1} added to q_star below
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v)[:] += info.sigma_as

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
        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step)
        bc_y_full = boundary_y(step)

        # Update q_star
        if info.moments == 1 and step == 0:
            # Run BDF1 on first time step
            tools._time_source_total_bdf1(scalar_flux[step,:,:,:,0], flux_last_1, \
                                xs_scatter[:,:,:,0], velocity, q_star, \
                                external(step), medium_map, info)
        elif info.moments == 1:
            # Run BDF2 on rest of time steps
            tools._time_source_total_bdf2(scalar_flux[step,:,:,:,0], flux_last_1, \
                                flux_last_2, xs_scatter[:,:,:,0], velocity, \
                                q_star, external(step), medium_map, info)
        else:
            # Anisotropic scattering of the known flux moments
            if step == 0:
                tools._time_source_star_bdf1(flux_last_1, q_star, external(step), \
                                             velocity, info)
            else:
                tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                             external(step), velocity, info)
            tools._time_right_side_aniso(q_star, scalar_flux[step], xs_scatter, \
                                         medium_map, P_source, info)

        # Add lagged artificial in-scatter source M_as . psi^{ell-1}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, flux_last_1)

        # Solve for angular flux of previous time step
        # flux_last_2 is not needed after q_star, so it is overwritten
        # with the new step and swapped with flux_last_1 (no copy)
        mg._known_source_angular(flux_last_2, xs_total_v, q_star, \
                                            bc_x_full, bc_y_full, \
                                            medium_map, delta_x, delta_y, angle_x, \
                                            angle_y, angle_w, info)
        flux_swap = flux_last_1
        flux_last_1 = flux_last_2
        flux_last_2 = flux_swap

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
            if M_as is not None:
                np.asarray(xs_total_v)[:] += info.sigma_as

    return flux_time[:,:,:,:,:]


cdef double[:,:,:] tr_bdf2(double[:,:,:,:]& flux_ell_x, double[:,:,:,:]& flux_ell_y, \
        double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        object external, object boundary_x, \
        object boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, object flux_file, params info, \
        params info_edge):

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

    # Artificial scattering (as-SN): sigma_as out-scattering on the LHS
    # of both stages, lagged in-scatter sources added to q_star below.
    # The CN stage uses the doubled operator convention (factor 2); the
    # BDF2 stage is fully implicit (factor 1).
    M_as = artificial_scatter_setup(angle_x, angle_y, angle_w, \
                                    info.sigma_as, info.beta_as)
    if M_as is not None:
        np.asarray(xs_total_v_cn)[:] += 2.0 * info.sigma_as
        np.asarray(xs_total_v_bdf2)[:] += info.sigma_as

    # Combine last time step and source term
    q_star = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)

    # Create angular flux of previous time steps
    flux_last_gamma = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Initialize flux moments for previous time step
    scalar_flux = tools.array_4d(info.cells_x, info.cells_y, info.groups, \
                                 info.moments)
    tools._angular_edge_to_moments(flux_ell_x, flux_ell_y, scalar_flux, \
                                   P_weights, info)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="TR-BDF2 ", ascii=True):



        # Boundary sources of the time step for all angles
        bc_x_full = boundary_x(step * 2)
        bc_y_full = boundary_y(step * 2)
        bc_xa_full = boundary_x(step * 2 + 1)
        bc_ya_full = boundary_y(step * 2 + 1)

        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star for CN step
        tools._time_source_star_cn_aniso(flux_ell_x, flux_ell_y, scalar_flux, \
                    xs_total, xs_scatter, velocity, q_star, external(step * 2), \
                    external(step * 2 + 1), medium_map, delta_x, delta_y, angle_x, \
                    angle_y, P_source, 2.0 / gamma, info)

        # Add lagged artificial in-scatter source 2 * M_as . psi^{ell}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, \
                    edge_to_center_angular(flux_ell_x, flux_ell_y), 2.0)

        # Solve for the \ell + gamma time step
        scalar_flux[:,:,:,:] = mg.multi_group_moments(scalar_flux, xs_total_v_cn, \
                            xs_scatter, q_star, bc_x_full, \
                            bc_y_full, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, P_source, \
                            P_weights, info)

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)
        # Solve for angular flux of \ell + gamma time step
        mg._known_source_angular(flux_last_gamma, xs_total_v_cn, q_star, \
                        bc_x_full, bc_y_full, medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info)

        ################################################################
        # BDF2
        ################################################################
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_ell_x, flux_ell_y, flux_last_gamma, \
                            q_star, external(step * 2 + 2), velocity, gamma, info)

        # Add lagged artificial in-scatter source M_as . psi^{ell+gamma}
        if M_as is not None:
            artificial_scatter_source(q_star, M_as, flux_last_gamma)

        # Solve for the \ell + 1 time step
        mg_result = mg.multi_group_moments(scalar_flux, xs_total_v_bdf2, \
                                xs_scatter, q_star, bc_xa_full, \
                                bc_ya_full, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, \
                                P_source, P_weights, info)
        scalar_flux[:,:,:,:] = mg_result[:,:,:,:]
        if flux_file is not None:
            flux_file[step] = np.asarray(scalar_flux[:,:,:,0])

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side_aniso(q_star, scalar_flux, xs_scatter, \
                                     medium_map, P_source, info)

        # Solve for angular flux of previous time step
        mg._interface_angular(flux_ell_x, flux_ell_y, xs_total_v_bdf2, \
                    q_star, bc_xa_full, bc_ya_full, medium_map, \
                    delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)

    return scalar_flux[:,:,:,0]
