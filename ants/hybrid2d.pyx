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

from ants import angular_xy
from ants.utils.hybrid import coarsen_velocity, indexing

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants cimport parameters

# Uncollided is fine grid (N^2 x G)
# Collided is coarse grid (N'^2 x G')

def backward_euler(double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, \
        double[:,:,:] xs_fission_u, double[:,:] xs_total_c, \
        double[:,:,:] xs_scatter_c, double[:,:,:] xs_fission_c, \
        double[:] velocity_u, double[:] external_u, \
        double[:] boundary_xu, double[:] boundary_yu, int[:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] edges_g, \
        int[:] edges_gidx, dict params_dict_u, dict params_dict_c):
    # Create angles and weights
    angle_xu, angle_yu, angle_wu = angular_xy(params_dict_u)
    angle_xc, angle_yc, angle_wc = angular_xy(params_dict_c)
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid2d_bdf1_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid2d_bdf1_collided(info_c, xs_total_u.shape[0])
    # Do not overwrite variables
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_3d(info_u.materials, info_u.groups, info_u.groups)
    tools._xs_matrix(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)
    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_3d(info_c.materials, info_c.groups, info_c.groups)
    tools._xs_matrix(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)
    # Create collided velocity
    velocity_c = coarsen_velocity(velocity_u, edges_gidx)
    # Create sigma_t + 1 / (v * dt)
    tools._total_velocity(xs_total_vu, velocity_u, info_u)
    tools._total_velocity(xs_total_vc, velocity_c, info_c)
    # Indexing Parameters
    coarse_idx, fine_idx, factor = indexing(info_u.groups, info_c.groups, \
                                                edges_g, edges_gidx)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total_vu, xs_matrix_u, velocity_u, external_u, \
                boundary_xu.copy(), boundary_yu.copy(), medium_map, delta_x, \
                delta_y, angle_xu, angle_yu, angle_wu, xs_total_vc, \
                xs_matrix_c, velocity_c, angle_xc, angle_yc, angle_wc, \
                fine_idx, coarse_idx, factor, info_u, info_c)
    return np.asarray(flux)


cdef double[:,:,:,:] multigroup_bdf1(double[:,:]& xs_total_u, \
        double[:,:,:]& xs_scatter_u, double[:]& velocity_u, double[:]& external_u, \
        double[:]& boundary_xu, double[:]& boundary_yu, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_xu, \
        double[:]& angle_yu, double[:]& angle_wu, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_c, double[:]& angle_xc, \
        double[:]& angle_yc, double[:]& angle_wc, int[:]& fine_idx, \
        int[:]& coarse_idx, double[:]& factor, params info_u, params info_c):
    # Initialize time step
    cdef int step
    # Combine last time step and source term
    q_star = tools.array_1d(info_u.cells_x * info_u.cells_y \
                            * info_u.angles * info_u.angles * info_u.groups)
    # Initialize angular flux for previous time step
    flux_last = tools.array_4d(info_u.cells_x, info_u.cells_y, \
                            info_u.angles * info_u.angles, info_u.groups)
    # Initialize uncollided scalar flux
    flux_u = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    # Initialize collided scalar flux
    flux_c = tools.array_3d(info_c.cells_x, info_c.cells_y, info_c.groups)
    # Initialize total scalar flux
    flux_t = tools.array_3d(info_u.cells_x, info_u.cells_y, info_u.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_4d(info_u.steps, info_u.cells_x, info_u.cells_y, info_u.groups)
    # Initialize collided source
    source_c = tools.array_1d(info_c.cells_x * info_c.cells_y * info_c.groups)
    # Initialize collided boundary
    cdef double[2] boundary_c = [0.0, 0.0]
    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="Time Steps", position=1, ascii=True):
    # for step in range(info_u.steps):
        # Adjust boundary condition
        tools.boundary_decay(boundary_xu, boundary_yu, step + 1, info_u)
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star(flux_last, q_star, external_u, velocity_u, info_u)
        # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
        flux_u = mg._known_source_scalar(xs_total_u, q_star, boundary_xu, \
                            boundary_yu, medium_map, delta_x, delta_y, \
                            angle_xu, angle_yu, angle_wu, info_u)
        # Step 2: Compute collided source (I x G')
        tools._hybrid_source_collided(flux_u, xs_scatter_u, source_c, \
                                medium_map, coarse_idx, info_u, info_c)
        # Step 3: Solve Collided Equation (I x G')
        flux_c = mg.source_iteration(flux_c, xs_total_c, xs_scatter_c, \
                        source_c, boundary_c, boundary_c, medium_map, delta_x, \
                        delta_y, angle_xc, angle_yc, angle_wc, info_c)
        # Step 4: Create a new source and solve for angular flux
        tools._expand_hybrid_source(flux_t, flux_c, fine_idx, factor, info_u, info_c)
        tools._hybrid_source_total(flux_t, flux_u, xs_scatter_u, q_star, \
                            medium_map, fine_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last = mg._known_source_angular(xs_total_u, q_star, boundary_xu, \
                                boundary_yu, medium_map, delta_x, delta_y, \
                                angle_xu, angle_yu, angle_wu, info_u)
        # Step 5: Update and repeat
        tools._angular_to_scalar(flux_last, flux_time[step], angle_wu, info_u)
    print()
    return flux_time[:,:,:,:]
