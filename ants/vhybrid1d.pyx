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

import ants
from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')


def backward_euler(int[:] groups_c, int[:] angles_c, double[:,:,:] initial_flux, \
        double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, double[:,:,:] xs_fission_u, \
        double[:] velocity_u, double[:,:,:,:] external_u, double[:,:,:,:] boundary_xu, \
        int[:] medium_map, double[:] delta_x, double[:] angle_xu, double[:] angle_wu, 
        double[:] edges_g, dict params_dict_u, dict params_dict_c):
    
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_bdf_timed1d(info_u, initial_flux.shape[0], \
            external_u.shape[0], boundary_xu.shape[0], xs_total_u.shape[0])
    
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    
    # Run BDF1
    flux = multigroup_bdf1(groups_c, angles_c, initial_flux.copy(), xs_total_u, \
                    xs_matrix_u, velocity_u, external_u, boundary_xu.copy(), \
                    medium_map, delta_x, angle_xu, angle_wu, edges_g, info_u, info_c)

    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf1(int[:] groups_c, int[:] angles_c, \
        double[:,:,:]& flux_last, double[:,:]& xs_total_u, double[:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, double[:,:,:,:]& external_u, double[:,:,:,:]& boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, double[:]& angle_wu, \
        double[:]& edges_g, params info_u, params info_c):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq, bc

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)
    
    # Scalar Flux Approximation
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    tools._angular_to_scalar(flux_last, flux_u, angle_wu, info_u)
    
    # Initialize time flux, collided boundary
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vBDF1*   ", ascii=True):
        
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bc = 0 if boundary_xu.shape[0] == 1 else step

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_wc = ants._angular_x(info_c.angles, info_c.bc_x)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
            source_c = tools.array_3d(info_c.cells_x, 1, info_c.groups)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = tools.int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(info_u.groups, info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            tools._vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 1.0, info_c)

        ########################################################################
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external_u[qq], \
                                     velocity_u, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                    q_star, source_c, boundary_xu[bc], boundary_xc, medium_map, \
                    delta_x, angle_xu, angle_xc, angle_wu, angle_wc, star_coef_c, \
                    edges_g, edges_gidx_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)
        
        # Step 5: Update and repeat
        tools._angular_to_scalar(flux_last, flux_time[step], angle_wu, info_u)
    
    return flux_time[:,:,:]


def crank_nicolson(int[:] groups_c, int[:] angles_c, double[:,:,:] initial_flux, \
        double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, double[:,:,:] xs_fission_u, \
        double[:] velocity_u, double[:,:,:,:] external_u, double[:,:,:,:] boundary_xu, \
        int[:] medium_map, double[:] delta_x, double[:] angle_xu, double[:] angle_wu, \
        double[:] edges_g, dict params_dict_u, dict params_dict_c):
    
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_cn_timed1d(info_u, initial_flux.shape[0], \
            external_u.shape[0], boundary_xu.shape[0], xs_total_u.shape[0])
    
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict_u)
    info_edge.edges = 1
    
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    
    # Run Crank-Nicolson
    flux = multigroup_cn(groups_c, angles_c, initial_flux.copy(), xs_total_u, \
                xs_matrix_u, velocity_u, external_u, boundary_xu.copy(), medium_map, \
                delta_x, angle_xu, angle_wu, edges_g, info_u, info_c, info_edge)
    
    return np.asarray(flux)


cdef double[:,:,:] multigroup_cn(int[:] groups_c, int[:] angles_c, \
        double[:,:,:]& flux_last, double[:,:]& xs_total_u, double[:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, double[:,:,:,:]& external_u, double[:,:,:,:]& boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, double[:]& angle_wu, \
        double[:]& edges_g, params info_u, params info_c, params info_edge):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq, qqa, bc

    # Create sigma_t + 2 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 2.0, info_u)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    tools._angular_edge_to_scalar(flux_last, flux_u, angle_wu, info_u)

    # Initialize time flux, collided boundary
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vCN*     ", ascii=True):
        
        # Determine dimensions of external and boundary sources
        qqa = 0 if external_u.shape[0] == 1 else step # Previous time step
        qq = 0 if external_u.shape[0] == 1 else step + 1
        bc = 0 if boundary_xu.shape[0] == 1 else step
        
        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_wc = ants._angular_x(info_c.angles, info_c.bc_x)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
            source_c = tools.array_3d(info_c.cells_x, 1, info_c.groups)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = tools.int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(info_u.groups, info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            tools._vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 2.0, info_c)

        ########################################################################

        # Update q_star
        tools._time_source_star_cn(flux_last, flux_u, xs_total_u, \
                    xs_scatter_u, velocity_u, q_star, external_u[qqa], \
                    external_u[qq], medium_map, delta_x, angle_xu, \
                    2.0, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                    q_star, source_c, boundary_xu[bc], boundary_xc, medium_map, \
                    delta_x, angle_xu, angle_xc, angle_wu, angle_wc, star_coef_c, \
                    edges_g, edges_gidx_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)

        # Step 5: Update and repeat
        tools._angular_edge_to_scalar(flux_last, flux_time[step], angle_wu, info_u)
        flux_u[:,:] = flux_time[step]

    return flux_time[:,:,:]


def bdf2(int[:] groups_c, int[:] angles_c, double[:,:,:] initial_flux, \
        double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, double[:,:,:] xs_fission_u, \
        double[:] velocity_u, double[:,:,:,:] external_u, double[:,:,:,:] boundary_xu, \
        int[:] medium_map, double[:] delta_x, double[:] angle_xu, double[:] angle_wu, 
        double[:] edges_g, dict params_dict_u, dict params_dict_c):
    
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_bdf_timed1d(info_u, initial_flux.shape[0], \
            external_u.shape[0], boundary_xu.shape[0], xs_total_u.shape[0])
    
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    
    # Run BDF2
    flux = multigroup_bdf2(groups_c, angles_c, initial_flux.copy(), xs_total_u, \
                    xs_matrix_u, velocity_u, external_u, boundary_xu.copy(), \
                    medium_map, delta_x, angle_xu, angle_wu, edges_g, info_u, info_c)
    
    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf2(int[:] groups_c, int[:] angles_c, \
        double[:,:,:]& flux_last_1, double[:,:]& xs_total_u, double[:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, double[:,:,:,:]& external_u, double[:,:,:,:]& boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, double[:]& angle_wu, \
        double[:]& edges_g, params info_u, params info_c):
    
    # Initialize time step, external and boundary indices
    cdef int step, qq, bc
    cdef double coef = 1.0

    # Create sigma_t + 1 / (v * dt) - For BDF1 Step - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, coef, info_u)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_2 = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize scalar fluxes
    flux_u = tools.array_2d(info_u.cells_x, info_u.groups)
    tools._angular_to_scalar(flux_last_1, flux_u, angle_wu, info_u)

    # Initialize time flux, collided boundary
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vBDF2*   ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bc = 0 if boundary_xu.shape[0] == 1 else step

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_wc = ants._angular_x(info_c.angles, info_c.bc_x)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
            source_c = tools.array_3d(info_c.cells_x, 1, info_c.groups)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = tools.int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(info_u.groups, info_c.groups)

            star_coef_c = tools.array_1d(info_c.groups)
            coef = 1.0 if step == 0 else 1.5
            tools._vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, coef, info_c)

        if (step == 1):
            star_coef_c = tools.array_1d(info_c.groups)
            tools._vhybrid_velocity(star_coef_c, velocity_u, edges_gidx_c, 1.5, info_c)

        ########################################################################

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, \
                                external_u[qq], velocity_u, info_u)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                        external_u[qq], velocity_u, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_u, xs_total_vu, xs_scatter_u, \
                    q_star, source_c, boundary_xu[bc], boundary_xc, medium_map, \
                    delta_x, angle_xu, angle_xc, angle_wu, angle_wc, star_coef_c, \
                    edges_g, edges_gidx_c, info_u, info_c)
 
        # Solve for angular flux of time step
        flux_last_2[:,:,:] = flux_last_1[:,:,:]
        flux_last_1[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    boundary_xu[bc], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)
        
        # Step 5: Update flux_time and repeat
        tools._angular_to_scalar(flux_last_1, flux_time[step], angle_wu, info_u)
        
        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_vu[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)


    return flux_time[:,:,:]


def tr_bdf2(int[:] groups_c, int[:] angles_c, double[:,:,:] initial_flux, \
        double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, double[:,:,:] xs_fission_u, \
        double[:] velocity_u, double[:,:,:,:] external_u, double[:,:,:,:] boundary_xu, \
        int[:] medium_map, double[:] delta_x, double[:] angle_xu, double[:] angle_wu, 
        double[:] edges_g, dict params_dict_u, dict params_dict_c):
    
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_tr_bdf_timed1d(info_u, initial_flux.shape[0], \
            external_u.shape[0], boundary_xu.shape[0], xs_total_u.shape[0])
    
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    
    # Create params with edges for CN method
    info_edge = parameters._to_params(params_dict_u)
    info_edge.edges = 1
    
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    
    # Run TR-BDF2
    flux = multigroup_tr_bdf2(groups_c, angles_c, initial_flux.copy(), xs_total_u, \
                xs_matrix_u, velocity_u, external_u, boundary_xu.copy(), medium_map, \
                delta_x, angle_xu, angle_wu, edges_g, info_u, info_c, info_edge)
    
    return np.asarray(flux)


cdef double[:,:,:] multigroup_tr_bdf2(int[:] groups_c, int[:] angles_c, \
        double[:,:,:]& flux_last_ell, double[:,:]& xs_total_u, double[:,:,:]& xs_scatter_u, \
        double[:]& velocity_u, double[:,:,:,:]& external_u, double[:,:,:,:]& boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, double[:]& angle_wu, \
        double[:]& edges_g, params info_u, params info_c, params info_edge):
    
    # Initialize time step
    cdef int step, qq, qqa, qqb, bc, bca

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

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_gamma = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize scalar fluxes - Uncollided
    flux_ell_u = tools.array_2d(info_u.cells_x, info_u.groups)
    tools._angular_edge_to_scalar(flux_last_ell, flux_ell_u, angle_wu, info_u)
    flux_gamma_u = tools.array_2d(info_u.cells_x, info_u.groups)

    # Initialize time flux, collided boundary
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x, info_u.groups)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="vTR-BDF2*", ascii=True):
        
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step * 2 # Ell Step
        qqa = 0 if external_u.shape[0] == 1 else step * 2 + 1 # Gamma Step
        qqb = 0 if external_u.shape[0] == 1 else step * 2 + 2 # Ell + 1 Step

        bc = 0 if boundary_xu.shape[0] == 1 else step * 2 # Ell Step
        bca = 0 if boundary_xu.shape[0] == 1 else step * 2 + 1 # Gamma Step

        ########################################################################
        # Variable Hybrid Method
        ########################################################################
        # Get New Angles, Groups for timestep
        # Set up Collided Angles
        if (step == 0) or ((step > 0) and (angles_c[step] != angles_c[step - 1])):
            info_c.angles = angles_c[step]
            angle_xc = tools.array_1d(info_c.angles)
            angle_wc = tools.array_1d(info_c.angles)
            angle_xc, angle_wc = ants._angular_x(info_c.angles, info_c.bc_x)

        if (step == 0) or ((step > 0) and (groups_c[step] != groups_c[step - 1])):
            info_c.groups = groups_c[step]

            # Initialize flux, external sources of appropriate size
            flux_c = tools.array_2d(info_c.cells_x, info_c.groups)
            source_c = tools.array_3d(info_c.cells_x, 1, info_c.groups)

            # len(edges_gidx_c) = groups_c + 1
            edges_gidx_c = tools.int_array_1d(info_c.groups + 1)
            edges_gidx_c = ants._energy_grid(info_u.groups, info_c.groups)

            star_coef_cn_c = tools.array_1d(info_c.groups)
            tools._vhybrid_velocity(star_coef_cn_c, velocity_u, edges_gidx_c, 2.0 / gamma, info_c)

            star_coef_bdf2_c = tools.array_1d(info_c.groups)
            tools._vhybrid_velocity(star_coef_bdf2_c, velocity_u, edges_gidx_c, coef, info_c)

        ########################################################################
        # Crank Nicolson
        ########################################################################
        # Update q_star
        tools._time_source_star_cn(flux_last_ell, flux_ell_u, xs_total_u, \
                    xs_scatter_u, velocity_u, q_star, external_u[qq], \
                    external_u[qqa], medium_map, delta_x, angle_xu, \
                    2.0 / gamma, info_u)
        
        # Run Hybrid Method
        hybrid_method(flux_gamma_u, flux_c, xs_total_u, xs_total_vu_cn, \
                    xs_scatter_u, q_star, source_c, boundary_xu[bc], boundary_xc, 
                    medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                    star_coef_cn_c, edges_g, edges_gidx_c, info_u, info_c)

        
        # Solve for angular flux of time step
        flux_last_gamma = mg._known_source_angular(xs_total_vu_cn, q_star, \
                                        boundary_xu[bc], medium_map, \
                                        delta_x, angle_xu, angle_wu, info_u)
        
        ########################################################################
        # BDF2
        ########################################################################
        # Update q_star
        tools._time_source_star_tr_bdf2(flux_last_ell, flux_last_gamma, \
                q_star, external_u[qqb], velocity_u, gamma, info_u)

        # Run Hybrid Method
        hybrid_method(flux_ell_u, flux_c, xs_total_u, xs_total_vu_bdf2, \
                    xs_scatter_u, q_star, source_c, boundary_xu[bca], boundary_xc, \
                    medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                    star_coef_bdf2_c, edges_g, edges_gidx_c, info_u, info_c)
        
        # Solve for angular flux of time step
        flux_last_ell[:,:,:] = mg._known_source_angular(xs_total_vu_bdf2, \
                                    q_star, boundary_xu[bca], medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)
        
        # Step 5: Update flux_time and repeat
        tools._angular_edge_to_scalar(flux_last_ell, flux_time[step], \
                                      angle_wu, info_u)
        flux_ell_u[:,:] = flux_time[step]

    return flux_time[:,:,:]


cdef void hybrid_method(double[:,:]& flux_u, double[:,:]& flux_c, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_vu, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& q_star, \
        double[:,:,:]& source_c, double[:,:,:]& boundary_xu, \
        double[:,:,:]& boundary_xc, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_xc, double[:]& angle_wu, \
        double[:]& angle_wc, double[:]& star_coef_c, double[:]& edges_g, \
        int[:]& edges_gidx_c, params info_u, params info_c):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
    flux_u[:,:] = mg._known_source_scalar(xs_total_vu, q_star, boundary_xu, \
                        medium_map, delta_x, angle_xu, angle_wu, info_u)

    # Step 2: Compute collided source (I x G')
    tools._vhybrid_source_c(flux_u, xs_scatter_u, source_c, medium_map, \
                            edges_gidx_c, info_u, info_c)
    
    # Step 3: Solve Collided Equation (I x G')
    tools._coarsen_flux(flux_u, flux_c, edges_gidx_c, info_c)
    flux_c[:,:] = mg.variable_source_iteration(flux_c, xs_total_u, star_coef_c, \
                        xs_scatter_u, source_c, boundary_xc, medium_map, delta_x, \
                        angle_xc, angle_wc, edges_g, edges_gidx_c, info_c)

    # Step 4: Create a new source and solve for angular flux
    tools._vhybrid_source_total(flux_u, flux_c, xs_scatter_u, q_star, medium_map, \
                                edges_g, edges_gidx_c, info_u, info_c)

