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

from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters

import numpy as np
from tqdm import tqdm


def backward_euler(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed1d_backward_euler(info, xs_total.shape[0])
    # Combine fission and scattering
    tools._xs_matrix(xs_scatter, xs_fission, info)
    # Create sigma_t + 1 / (v * dt)
    # xs_total_v = memoryview(np.zeros((info.materials, info.groups)))
    # tools._total_velocity(xs_total_v, xs_total, velocity, info)
    # xs_total_v = memoryview(np.zeros((info.materials, info.groups)))
    tools._total_velocity(xs_total, velocity, info)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total, xs_scatter, velocity, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf1(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):
    # Initialize time step
    cdef int step
    # Combine last time step and source term
    q_star = tools.array_1d((info.cells_x + info.edges) * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_2d(info.cells_x + info.edges, info.groups)
    # Create angular flux of previous time step
    flux_last = tools.array_3d(info.cells_x + info.edges, info.angles, \
                                  info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info.steps, info.cells_x + info.edges, info.groups)
    # Iterate over time steps
    for step in tqdm(range(info.steps)):
        # Adjust boundary condition
        tools.boundary_decay(boundary_x, step, info)
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star(flux_last, q_star, external, velocity, info)
        # Solve for the current time step
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total, \
                                xs_scatter, q_star, boundary_x, medium_map, \
                                delta_x, angle_x, angle_w, info)
        # Update previous time step
        scalar_flux[:,:] = flux_time[step,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_source_total(q_star, scalar_flux, flux_last, xs_scatter, \
                                 velocity, medium_map, external, info)
        # Solve for angular flux of previous time step
        flux_last = mg._known_source(xs_total, q_star, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
    return flux_time[:,:,:]
