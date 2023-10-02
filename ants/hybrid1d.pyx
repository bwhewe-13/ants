########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Hybrid Multigroup Neutron Transport Problems
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

from libc.math cimport sqrt

from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')


def backward_euler(double[:,:] xs_total_u, double[:,:] xs_total_c, \
        double[:,:,:] xs_scatter_u, double[:,:,:] xs_scatter_c, \
        double[:,:,:] xs_fission_u, double[:,:,:] xs_fission_c, \
        double[:] velocity_u, double[:] velocity_c, double[:] external_u, \
        double[:] boundary_xu, int[:] medium_map, double[:] delta_x, \
        double[:] angle_xu, double[:] angle_xc, double[:] angle_wu, \
        double[:] angle_wc, int[:] fine_idx, int[:] coarse_idx, \
        double[:] factor, dict params_dict_u, dict params_dict_c):
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid1d_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid1d_collided(info_c, xs_total_u.shape[0])
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total_u, xs_total_c, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, external_u, boundary_xu.copy(), \
                medium_map, delta_x, angle_xu, angle_xc, angle_wu, \
                angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)
    return np.asarray(flux)


def crank_nicolson(double[:,:] xs_total_u, double[:,:] xs_total_c, \
        double[:,:,:] xs_scatter_u, double[:,:,:] xs_scatter_c, \
        double[:,:,:] xs_fission_u, double[:,:,:] xs_fission_c, \
        double[:] velocity_u, double[:] velocity_c, double[:] external_u, \
        double[:] boundary_xu, int[:] medium_map, double[:] delta_x, \
        double[:] angle_xu, double[:] angle_xc, double[:] angle_wu, \
        double[:] angle_wc, int[:] fine_idx, int[:] coarse_idx, \
        double[:] factor, dict params_dict_u, dict params_dict_c):
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid1d_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid1d_collided(info_c, xs_total_u.shape[0])
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict_u)
    info_edge.edges = 1
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)
    # Run Crank-Nicolson
    flux = multigroup_cn(xs_total_u, xs_total_c, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, external_u, boundary_xu.copy(), \
                medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                fine_idx, coarse_idx, factor, info_u, info_c, info_edge)
    return np.asarray(flux)


def bdf2(double[:,:] xs_total_u, double[:,:] xs_total_c, \
        double[:,:,:] xs_scatter_u, double[:,:,:] xs_scatter_c, \
        double[:,:,:] xs_fission_u, double[:,:,:] xs_fission_c, \
        double[:] velocity_u, double[:] velocity_c, double[:] external_u, \
        double[:] boundary_xu, int[:] medium_map, double[:] delta_x, \
        double[:] angle_xu, double[:] angle_xc, double[:] angle_wu, \
        double[:] angle_wc, int[:] fine_idx, int[:] coarse_idx, \
        double[:] factor, dict params_dict_u, dict params_dict_c):
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid1d_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid1d_collided(info_c, xs_total_u.shape[0])
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)
    # Run BDF2
    flux = multigroup_bdf2(xs_total_u, xs_total_c, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, external_u, boundary_xu.copy(), \
                medium_map, delta_x, angle_xu, angle_xc, angle_wu, \
                angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)
    return np.asarray(flux)


def tr_bdf2(double[:,:] xs_total_u, double[:,:] xs_total_c, \
        double[:,:,:] xs_scatter_u, double[:,:,:] xs_scatter_c, \
        double[:,:,:] xs_fission_u, double[:,:,:] xs_fission_c, \
        double[:] velocity_u, double[:] velocity_c, double[:] external_u, \
        double[:] boundary_xu, int[:] medium_map, double[:] delta_x, \
        double[:] angle_xu, double[:] angle_xc, double[:] angle_wu, \
        double[:] angle_wc, int[:] fine_idx, int[:] coarse_idx, \
        double[:] factor, dict params_dict_u, dict params_dict_c):
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid1d_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid1d_collided(info_c, xs_total_u.shape[0])
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict_u)
    info_edge.edges = 1
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)
    # Run TR-BDF2
    flux = multigroup_tr_bdf2(xs_total_u, xs_total_c, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, external_u, boundary_xu.copy(), \
                medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                fine_idx, coarse_idx, factor, info_u, info_c, info_edge)

    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf1(double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, double[:]& external_u, \
        double[:]& boundary_xu, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_wu, \
        double[:]& angle_wc, int[:]& fine_idx, int[:]& coarse_idx, \
        double[:]& factor, params info_u, params info_c):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, bc1, bc2
    
    # Set indexing for external and boundary sources
    qq2 = 1 if info_u.qdim < 4 else info_u.steps
    bc2 = 1 if info_u.bcdim_x < 4 else info_u.steps
    
    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)
    
    # Combine last time step and uncollided source term
    q_star = tools.array_1d(info_u.cells_x * info_u.angles * info_u.groups)
    # Initialize angular flux for previous time step
    flux_last = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)
    
    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
    flux_t = tools.array_2d(info_u.cells_x, info_u.groups)
    
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    # Initialize collided source and boundary
    source_c = tools.array_1d(info_c.cells_x * info_c.groups)
    cdef double[2] boundary_xc = [0.0, 0.0]

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF1   ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info_u.qdim < 4 else step
        bc1 = 0 if info_u.bcdim_x < 4 else step
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, \
                                external_u[qq1::qq2], velocity_u, info_u)
        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, flux_t, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bc1::bc2], boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc1::bc2], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)
        # Step 5: Update and repeat
        tools._angular_to_scalar(flux_last, flux_time[step], angle_wu, info_u)
    return flux_time[:,:,:]


cdef double[:,:,:] multigroup_cn(double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, double[:]& external_u, \
        double[:]& boundary_xu, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_wu, \
        double[:]& angle_wc, int[:]& fine_idx, int[:]& coarse_idx, \
        double[:]& factor, params info_u, params info_c, params info_edge):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, qqa, bc1, bc2
    
    # Set indexing for external and boundary sources
    qq2 = 1 if info_u.qdim < 4 else info_u.steps
    bc2 = 1 if info_u.bcdim_x < 4 else info_u.steps
    
    # Create sigma_t + 2 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 2.0, info_u)

    # Create sigma_t + 2 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 2.0, info_c)
    
    # Combine last time step and uncollided source term
    q_star = tools.array_1d(info_u.cells_x * info_u.angles * info_u.groups)
    # Initialize angular flux for previous time step
    flux_last = tools.array_3d(info_u.cells_x + 1, info_u.angles, info_u.groups)
    
    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
    flux_t = tools.array_2d(info_u.cells_x, info_u.groups)
    
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_1d(info_c.cells_x * info_c.groups)
    cdef double[2] boundary_xc = [0.0, 0.0]
    
    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="CN     ", ascii=True):
        # Determine dimensions of external and boundary sources
        qqa = 0 if info_u.qdim < 4 else step - 1 # Previous time step
        qq1 = 0 if info_u.qdim < 4 else step
        bc1 = 0 if info_u.bcdim_x < 4 else step
        # Update q_star
        tools._time_source_star_cn(flux_last, flux_u, xs_total_u, \
                xs_scatter_u, velocity_u, q_star, external_u[qqa::qq2], \
                external_u[qq1::qq2], medium_map, delta_x, angle_xu, \
                2.0, step, info_u)
        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, flux_t, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bc1::bc2], boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc1::bc2], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)
        # Step 5: Update and repeat
        tools._angular_edge_to_scalar(flux_last, flux_time[step], \
                                      angle_wu, info_u)
        flux_u[:,:] = flux_time[step]
    return flux_time[:,:,:]


cdef double[:,:,:] multigroup_bdf2(double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, double[:]& external_u, \
        double[:]& boundary_xu, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_wu, \
        double[:]& angle_wc, int[:]& fine_idx, int[:]& coarse_idx, \
        double[:]& factor, params info_u, params info_c):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq1, qq2, bc1, bc2
    
    # Set indexing for external and boundary sources
    qq2 = 1 if info_u.qdim < 4 else info_u.steps
    bc2 = 1 if info_u.bcdim_x < 4 else info_u.steps

    # Create sigma_t + 1 / (v * dt) - For BDF1 Step - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - For BDF1 Step - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)

    # Combine last time step and uncollided source term
    q_star = tools.array_1d(info_u.cells_x * info_u.angles * info_u.groups)
    
    # Initialize angular flux for previous time steps
    flux_last_1 = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)
    flux_last_2 = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)
    
    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
    flux_t = tools.array_2d(info_u.cells_x, info_u.groups)
    
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    # Initialize collided source and boundary
    source_c = tools.array_1d(info_c.cells_x * info_c.groups)
    cdef double[2] boundary_xc = [0.0, 0.0]

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF2   ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info_u.qdim < 4 else step
        bc1 = 0 if info_u.bcdim_x < 4 else step
        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, \
                                external_u[qq1::qq2], velocity_u, info_u)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                        external_u[qq1::qq2], velocity_u, info_u)
        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, flux_t, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bc1::bc2], boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, info_u, info_c)        
        # Solve for angular flux of time step
        flux_last_2[:,:,:] = flux_last_1[:,:,:]
        flux_last_1 = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc1::bc2], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)
        # Step 5: Update flux_time and repeat
        tools._angular_to_scalar(flux_last_1, flux_time[step], angle_wu, info_u)
        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:    
            xs_total_vu[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)
            xs_total_vc[:,:] = xs_total_c[:,:]
            tools._total_velocity(xs_total_vc, velocity_c, 1.5, info_c)
    return flux_time[:,:,:]


cdef double[:,:,:] multigroup_tr_bdf2(double[:,:]& xs_total_u, \
        double[:,:]& xs_total_c, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:]& external_u, double[:]& boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        params info_u, params info_c, params info_edge):
    # Initialize time step
    cdef int step, qq1, qq2, bc1, bc2
    # Initialize gamma
    cdef double gamma = 0.5 # 2 - sqrt(2)
    # Set indexing for external and boundary sources
    qq2 = 1 if info_u.qdim < 4 else info_u.steps
    bc2 = 1 if info_u.bcdim_x < 4 else info_u.steps
    
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

    # Combine last time step and uncollided source term
    q_star = tools.array_1d(info_u.cells_x * info_u.angles * info_u.groups)
    
    # Initialize angular flux for previous time steps
    flux_last_ell = tools.array_3d(info_u.cells_x + 1, info_u.angles, info_u.groups)
    flux_last_gamma = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)
    
    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
    flux_t = tools.array_2d(info_u.cells_x, info_u.groups)

    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)

    # Initialize collided source and boundary
    source_c = tools.array_1d(info_c.cells_x * info_c.groups)
    cdef double[2] boundary_xc = [0.0, 0.0]

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="TR-BDF2", ascii=True):
        # Determine dimensions of external and boundary sources
        qq1 = 0 if info_u.qdim < 4 else step
        bc1 = 0 if info_u.bcdim_x < 4 else step
        ################################################################
        # Crank Nicolson
        ################################################################
        # Update q_star
        tools._time_source_star_cn(flux_last_ell, flux_u, xs_total_u, \
                xs_scatter_u, velocity_u, q_star, external_u[qq1::qq2], \
                external_u[qq1::qq2], medium_map, delta_x, angle_xu, \
                2.0 / gamma, step, info_u)
        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, flux_t, xs_total_vu_cn, xs_total_vc_cn, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      boundary_xu[bc1::bc2], boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last_gamma = mg._known_source_angular(xs_total_vu_cn, q_star, \
                                        boundary_xu[bc1::bc2], medium_map, \
                                        delta_x, angle_xu, angle_wu, info_u)
        ################################################################
        # BDF2
        ################################################################
        # Update q_star
        tools._time_source_star_tr_bdf2(flux_last_ell, flux_last_gamma, \
                q_star, external_u[qq1::qq2], velocity_u, gamma, info_u)
        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, flux_t, xs_total_vu_bdf2, \
                      xs_total_vc_bdf2, xs_scatter_u, xs_scatter_c, \
                      q_star, source_c, boundary_xu[bc1::bc2], boundary_xc, \
                      medium_map, delta_x, angle_xu, angle_xc, angle_wu, \
                      angle_wc, fine_idx, coarse_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last_ell = mg._known_source_angular(xs_total_vu_bdf2, q_star, \
                                    boundary_xu[bc1::bc2], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)
        # Step 5: Update flux_time and repeat
        tools._angular_edge_to_scalar(flux_last_ell, flux_time[step], \
                                      angle_wu, info_u)
        flux_u[:,:] = flux_time[step]
        # Adjust gamma step
        if step == 0:
            gamma = 2 - sqrt(2)

            xs_total_vu_cn[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu_cn, velocity_u, 2.0 / gamma, info_u)
            
            xs_total_vc_cn[:,:] = xs_total_c[:,:]
            tools._total_velocity(xs_total_vc_cn, velocity_c, 2.0 / gamma, info_c)
            
            # Create sigma_t + (2 - gamma) / ((1 - gamma) * v * dt) - BDF2 Step
            xs_total_vu_bdf2[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu_bdf2, velocity_u, \
                                    (2.0 - gamma) / (1.0 - gamma), info_u)
            
            xs_total_vc_bdf2[:,:] = xs_total_c[:,:]
            tools._total_velocity(xs_total_vc_bdf2, velocity_c, \
                                    (2.0 - gamma) / (1.0 - gamma), info_c)

    return flux_time[:,:,:]


cdef void hybrid_method(double[:,:]& flux_u, double[:,:]& flux_c, \
        double[:,:]& flux_t, double[:,:]& xs_total_vu, double[:,:]& xs_total_vc, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
        double[:]& q_star, double[:]& source_c, double[:]& boundary_xu, \
        double[:]& boundary_xc, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_wu, \
        double[:]& angle_wc, int[:]& fine_idx, int[:]& coarse_idx, \
        double[:]& factor, params info_u, params info_c):
    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
    flux_u[:,:] = mg._known_source_scalar(xs_total_vu, q_star, boundary_xu, \
                    medium_map, delta_x, angle_xu, angle_wu, info_u)
    # Step 2: Compute collided source (I x G')
    tools._hybrid_source_collided(flux_u, xs_scatter_u, source_c, \
                            medium_map, coarse_idx, info_u, info_c)
    # Step 3: Solve Collided Equation (I x G')
    flux_c[:,:] = mg.source_iteration(flux_c, xs_total_vc, xs_scatter_c, \
                                source_c, boundary_xc, medium_map, \
                                delta_x, angle_xc, angle_wc, info_c)
    # Step 4: Create a new source and solve for angular flux
    tools._expand_hybrid_source(flux_t, flux_c, fine_idx, factor, info_u, info_c)
    tools._hybrid_source_total(flux_t, flux_u, xs_scatter_u, q_star, \
                        medium_map, fine_idx, factor, info_u, info_c)
