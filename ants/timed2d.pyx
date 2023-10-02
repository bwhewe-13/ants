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

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants cimport parameters


def backward_euler(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, double[:] boundary_y, int[:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] angle_x, \
        double[:] angle_y, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed2d(info, xs_total.shape[0])
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total, xs_matrix, velocity, external, \
                           boundary_x.copy(), boundary_y.copy(), medium_map, \
                           delta_x, delta_y, angle_x, angle_y, angle_w, info)
    return np.asarray(flux)


def crank_nicolson(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, double[:] boundary_y, int[:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] angle_x, \
        double[:] angle_y, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed2d(info, xs_total.shape[0])
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict)
    info_edge.edges = 1
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run Crank Nicolson
    flux = multigroup_cn(xs_total, xs_matrix, velocity, external, \
                         boundary_x.copy(), boundary_y.copy(), medium_map, \
                         delta_x, delta_y, angle_x, angle_y, angle_w, \
                         info, info_edge)
    return np.asarray(flux)


def bdf2(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, double[:] boundary_y, int[:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] angle_x, \
        double[:] angle_y, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed2d(info, xs_total.shape[0])
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run BDF2
    flux = multigroup_bdf2(xs_total, xs_matrix, velocity, external, \
                           boundary_x.copy(), boundary_y.copy(), medium_map, \
                           delta_x, delta_y, angle_x, angle_y, angle_w, info)
    return np.asarray(flux)


def tr_bdf2(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary_x, double[:] boundary_y, int[:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] angle_x, \
        double[:] angle_y, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_timed2d(info, xs_total.shape[0])
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict)
    info_edge.edges = 1
    # Combine fission and scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Run BDF2
    flux = multigroup_tr_bdf2(xs_total, xs_matrix, velocity, external, \
                              boundary_x.copy(), boundary_y.copy(), \
                              medium_map, delta_x, delta_y, angle_x, \
                              angle_y, angle_w, info, info_edge)
    return np.asarray(flux)


cdef double[:,:,:,:] multigroup_bdf1(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, double[:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    # Set indexing for external and boundary sources
    qq2 = 1 if info.qdim < 4 else info.steps
    bcx2 = 1 if info.bcdim_x < 4 else info.steps
    bcy2 = 1 if info.bcdim_y < 4 else info.steps

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.0, info)

    # Combine last time step and source term
    q_star = tools.array_1d(info.cells_x * info.cells_y * info.angles \
                            * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    # Create angular flux of previous time step
    flux_last = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)
    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="BDF1   ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info.qdim < 4 else step
        bcx1 = 0 if info.bcdim_x < 4 else step
        bcy1 = 0 if info.bcdim_y < 4 else step
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, \
                                     external[qq1::qq2], velocity, info)
        # Run source iteration
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx1::bcx2], \
                                boundary_y[bcy1::bcy2], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        flux_last = mg._known_source_angular(xs_total_v, q_star, \
                            boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], \
                            medium_map, delta_x, delta_y, angle_x, \
                            angle_y, angle_w, info)
    return flux_time[:,:,:,:]


cdef double[:,:,:,:] bdf1_one_step(double[:,:,:,:] flux_last, \
        double[:,:,:]& scalar_flux, double[:,:,:,:]& flux_time, \
        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:]& external, double[:]& q_star, \
        double[:]& boundary_x, double[:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, int step, params info):
    # Create sigma_t + 1 / (v * dt) (For BDF1 time step)
    tools._total_velocity(xs_total_v, velocity, 1.0, info)
    # Update q_star as external + 1/(v*dt) * psi
    tools._time_source_star_bdf1(flux_last, q_star, external, velocity, info)
    # Solve for the current time step
    flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, xs_scatter, \
                            q_star, boundary_x, boundary_y, medium_map, \
                            delta_x, delta_y, angle_x, angle_y, angle_w, info)
    # Update previous time step
    scalar_flux[:,:,:] = flux_time[step,:,:,:]
    # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
    tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
    # Solve for angular flux of previous time step
    flux_last = mg._known_source_angular(xs_total_v, q_star, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, angle_x, \
                        angle_y, angle_w, info)
    return flux_last[:,:,:,:]


cdef double[:,:,:,:] multigroup_cn(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, double[:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, params info_edge):
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, qqa, bcx1, bcx2, bcy1, bcy2
    
    # Set indexing for external and boundary sources
    qq2 = 1 if info.qdim < 4 else info.steps
    bcx2 = 1 if info.bcdim_x < 4 else info.steps
    bcy2 = 1 if info.bcdim_y < 4 else info.steps

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 2.0, info)

    # Combine last time step and source term
    q_star = tools.array_1d(info.cells_x * info.cells_y * info.angles \
                            * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    # Create angular flux of previous time step
    flux_last_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                               info.angles * info.angles, info.groups)
    flux_last_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                               info.angles * info.angles, info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)
    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="CN     ", ascii=True):
        # Determine dimensions of external and boundary sources
        qqa = 0 if info.qdim < 4 else step - 1 # Previous time step
        qq1 = 0 if info.qdim < 4 else step
        bcx1 = 0 if info.bcdim_x < 4 else step
        bcy1 = 0 if info.bcdim_y < 4 else step
        # Update q_star
        tools._time_source_star_cn(flux_last_x, flux_last_y, scalar_flux, \
                xs_total, xs_scatter, velocity, q_star, external[qqa::qq2], \
                external[qq1::qq2], medium_map, delta_x, delta_y, angle_x, \
                angle_y, 2.0, step, info)
        # Run source iteration
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx1::bcx2], \
                                boundary_y[bcy1::bcy2], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        mg._interface_angular(flux_last_x, flux_last_y, xs_total_v, q_star, \
                boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], medium_map, \
                delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)
        
    return flux_time[:,:,:,:]


cdef double[:,:,:,:] multigroup_bdf2(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, double[:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    # Set indexing for external and boundary sources
    qq2 = 1 if info.qdim < 4 else info.steps
    bcx2 = 1 if info.bcdim_x < 4 else info.steps
    bcy2 = 1 if info.bcdim_y < 4 else info.steps

    # Create sigma_t + 1 / (v * dt)
    xs_total_v = tools.array_2d(info.materials, info.groups)
    xs_total_v[:,:] = xs_total[:,:]

    # Combine last time step and source term
    q_star = tools.array_1d(info.cells_x * info.cells_y * info.angles \
                            * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    # Create angular flux of previous time steps
    flux_last_1 = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)
    flux_last_2 = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Calculate first time step with BDF1
    flux_last_1 = bdf1_one_step(flux_last_1, scalar_flux, flux_time, \
                        xs_total_v, xs_scatter, velocity, external[0::qq2], \
                        q_star, boundary_x[0::bcx2], boundary_y[0::bcy2], \
                        medium_map, delta_x, delta_y, angle_x, angle_y, \
                        angle_w, 0, info)
    # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
    xs_total_v[:,:] = xs_total[:,:]
    tools._total_velocity(xs_total_v, velocity, 1.5, info)
    # Iterate over time steps
    for step in tqdm(range(1, info.steps), desc="BDF2   ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info.qdim < 4 else step
        bcx1 = 0 if info.bcdim_x < 4 else step
        bcy1 = 0 if info.bcdim_y < 4 else step
        # Update q_star
        tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                                     external[qq1::qq2], velocity, info)
        # Run source iteration
        flux_time[step] = mg.source_iteration(scalar_flux, xs_total_v, \
                                xs_scatter, q_star, boundary_x[bcx1::bcx2], \
                                boundary_y[bcy1::bcy2], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        # Update previous time step
        scalar_flux[:,:,:] = flux_time[step,:,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + external + 1/(v*dt) * psi^{\ell-1}
        tools._time_right_side(q_star, scalar_flux, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        flux_last_2[:,:,:,:] = flux_last_1[:,:,:,:]
        flux_last_1 = mg._known_source_angular(xs_total_v, q_star, \
                            boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], \
                            medium_map, delta_x, delta_y, angle_x, \
                            angle_y, angle_w, info)
    return flux_time[:,:,:,:]


cdef double[:,:,:,:] multigroup_tr_bdf2(double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:]& external, \
        double[:]& boundary_x, double[:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info, params info_edge):
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, bcx1, bcx2, bcy1, bcy2

    # Initialize gamma
    cdef double gamma = 0.5 # 2 - sqrt(2)
    
    # Set indexing for external and boundary sources
    qq2 = 1 if info.qdim < 4 else info.steps
    bcx2 = 1 if info.bcdim_x < 4 else info.steps
    bcy2 = 1 if info.bcdim_y < 4 else info.steps

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
    q_star = tools.array_1d(info.cells_x * info.cells_y * info.angles \
                            * info.angles * info.groups)
    # Initialize scalar flux for previous time step
    scalar_ell = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    scalar_gamma = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    
    # Create angular flux of previous time steps
    flux_ell_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                               info.angles * info.angles, info.groups)
    flux_ell_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                               info.angles * info.angles, info.groups)
    flux_gamma = tools.array_4d(info.cells_x, info.cells_y, \
                               info.angles * info.angles, info.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info.steps, info.cells_x, info.cells_y, info.groups)

    # Iterate over time steps
    for step in tqdm(range(info.steps), desc="TR-BDF2", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info.qdim < 4 else step
        bcx1 = 0 if info.bcdim_x < 4 else step
        bcy1 = 0 if info.bcdim_y < 4 else step
        # Update q_star for CN step
        tools._time_source_star_cn(flux_ell_x, flux_ell_y, scalar_ell, \
                xs_total, xs_scatter, velocity, q_star, external[qq1::qq2], \
                external[qq1::qq2], medium_map, delta_x, delta_y, angle_x, \
                angle_y, 2.0 / gamma, step, info)

        # Solve for the \ell + gamma time step
        scalar_gamma = mg.source_iteration(scalar_gamma, xs_total_v_cn, \
                            xs_scatter, q_star, boundary_x[bcx1::bcx2], \
                            boundary_y[bcy1::bcy2], medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_gamma, xs_scatter, medium_map, info)
        # Solve for angular flux of \ell + gamma time step
        flux_gamma = mg._known_source_angular(xs_total_v_cn, q_star, \
                            boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], \
                            medium_map, delta_x, delta_y, angle_x, angle_y, \
                            angle_w, info)
        # Update q_star for BDF2 Step
        tools._time_source_star_tr_bdf2(flux_ell_x, flux_ell_y, flux_gamma, \
                        q_star, external[qq1::qq2], velocity, gamma, info)
        # Solve for the \ell + 1 time step
        flux_time[step] = mg.source_iteration(scalar_ell, xs_total_v_bdf2, \
                                xs_scatter, q_star, boundary_x[bcx1::bcx2], \
                                boundary_y[bcy1::bcy2], medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)
        # Update previous time step
        scalar_ell[:,:,:] = flux_time[step,:,:,:]
        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        tools._time_right_side(q_star, scalar_ell, xs_scatter, medium_map, info)
        # Solve for angular flux of previous time step
        mg._interface_angular(flux_ell_x, flux_ell_y, xs_total_v_bdf2, q_star, \
                boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], medium_map, \
                delta_x, delta_y, angle_x, angle_y, angle_w, info_edge)

        # Adjust gamma step
        if step == 0:
            gamma = 2 - sqrt(2)

            xs_total_v_cn[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v_cn, velocity, 2.0 / gamma, info)

            xs_total_v_bdf2[:,:] = xs_total[:,:]
            tools._total_velocity(xs_total_v_bdf2, velocity, \
                                  (2.0 - gamma) / (1.0 - gamma), info)
    return flux_time[:,:,:,:]