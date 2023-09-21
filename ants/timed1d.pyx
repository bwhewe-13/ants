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
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

import numpy as np
from tqdm import tqdm

from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters


def bdf1(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed1d_bdf1(info, xs_total.shape[0])
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total, xs_matrix, velocity, external, \
                boundary_x.copy(), medium_map, delta_x, angle_x, angle_w, info)
    return np.asarray(flux)


def bdf2(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed1d_bdf2(info, xs_total.shape[0])
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run Backward Euler
    flux = multigroup_bdf2(xs_total, xs_matrix, velocity, external, \
                boundary_x.copy(), medium_map, delta_x, angle_x, angle_w, info)
    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf1(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):
    # Initialize time step
    cdef int step
    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)
    # Combine last time step and source term
    q_star = tools.array_1d(info.cells_x * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x, info.groups)
    # Create angular flux of previous time step
    flux_last = tools.array_3d(info.cells_x, info.angles, info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info.steps, info.cells_x, info.groups)
    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="Time Steps", ascii=True):
        # Adjust boundary condition
        tools.boundary_decay(boundary_x, step, info)
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external, velocity, info)
        # Solve for the current time step
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x, medium_map, \
                                delta_x, angle_x, angle_w, info)
        # Update previous time step
        scalar_flux[:,:] = flux_time[step,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_source_total_bdf1(q_star, scalar_flux, flux_last, \
                        xs_scatter, velocity, medium_map, external, info)
        # Solve for angular flux of previous time step
        flux_last = mg._known_source_angular(xs_total_v, q_star, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
    return flux_time[:,:,:]


cdef double[:,:,:] bdf1_one_step(double[:,:,:] flux_last, double[:,:]& scalar_flux, \
        double[:,:,:]& flux_time, double[:,:]& xs_total_v, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& q_star, double[:]& boundary_x, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        int step, params info):
    # Create sigma_t + 1 / (v * dt) (For BDF1 time step)
    tools._total_velocity(xs_total_v, velocity, 1.0, info)
    # Calculate flux_last (first time step is BDF1)
    tools.boundary_decay(boundary_x, step, info)
    # Update q_star as external + 1/(v*dt) * psi
    tools._time_source_star_bdf1(flux_last, q_star, external, velocity, info)
    # Solve for the current time step
    flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, xs_scatter, \
                                q_star, boundary_x, medium_map, delta_x, \
                                angle_x, angle_w, info)
    # Update previous time step
    scalar_flux[:,:] = flux_time[step,:,:]
    # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
    tools._time_source_total_bdf1(q_star, scalar_flux, flux_last, \
                        xs_scatter, velocity, medium_map, external, info)
    # Solve for angular flux of previous time step
    flux_last = mg._known_source_angular(xs_total_v, q_star, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
    return flux_last


cdef double[:,:,:] multigroup_bdf2(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):
    # Initialize time step
    cdef int step
    # Combine total cross section and time coefficient
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    # Combine last time step and source term
    q_star = tools.array_1d(info.cells_x * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x, info.groups)
    # Create angular flux of previous time steps
    flux_last_1 = tools.array_3d(info.cells_x, info.angles, info.groups)
    flux_last_2 = tools.array_3d(info.cells_x, info.angles, info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info.steps, info.cells_x, info.groups)
    # Calculate first time step with BDF1
    flux_last_1 = bdf1_one_step(flux_last_1, scalar_flux, flux_time, \
                xs_total_v, xs_scatter, velocity, external, q_star, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, 0, info)
    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.5, info)
    # Iterate over time steps
    for step in tqdm(range(1, info.steps), desc="Time Steps", ascii=True):
        # Adjust boundary condition
        tools.boundary_decay(boundary_x, step, info)
        # Update q_star = external + 2/(v*dt)*psi^{\ell-1} \
        #                    - 1/(2*v*dt)*psi^{\ell-2}
        tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                     external, velocity, info)
        # Solve for the current time step
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x, medium_map, \
                                delta_x, angle_x, angle_w, info)
        # Update previous time step
        scalar_flux[:,:] = flux_time[step,:,:]
        # Create q = (sigma_s + sigma_f) * phi^{\ell} + external \
        #           + 2/(v*dt) * psi^{\ell-1} - 1/(2*v*dt) * psi^{\ell-2}
        tools._time_source_total_bdf2(q_star, scalar_flux, flux_last_1, \
                                      flux_last_2, xs_scatter, velocity, \
                                      medium_map, external, info)
        # Solve for angular flux of previous time step
        flux_last_2[:,:,:] = flux_last_1[:,:,:]
        flux_last_1 = mg._known_source_angular(xs_total_v, q_star, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
    return flux_time[:,:,:]