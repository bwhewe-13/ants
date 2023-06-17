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

from ants import angular_x
from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants cimport parameters
from ants.utils.hybrid import hybrid_coarsen, hybrid_index

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')

def backward_euler(double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, \
        double[:,:,:] xs_fission_u, double[:] velocity_u, \
        double[:] external_u, double[:] boundary_u, int[:] medium_map, \
        double[:] delta_x, double[:] edges_g, int[:] edges_gidx, \
        dict params_dict_u, dict params_dict_c):
    # Create angles and weights
    angle_xu, angle_wu = angular_x(params_dict_u)
    angle_xc, angle_wc = angular_x(params_dict_c)
    # Convert uncollided dictionary to type params
    info_u = parameters._to_params(params_dict_u)
    parameters._check_hybrid1d_bdf1_uncollided(info_u, xs_total_u.shape[0])
    # Convert collided dictionary to type params
    info_c = parameters._to_params(params_dict_c)
    parameters._check_hybrid1d_bdf1_collided(info_c, xs_total_u.shape[0])
    # Combine fission and scattering
    tools._xs_matrix(xs_scatter_u, xs_fission_u, info_u)
    # Create collided cross sections and velocity
    xs_total_c, xs_scatter_c, velocity_c = hybrid_coarsen(xs_total_u, \
                        xs_scatter_u, velocity_u, edges_g, edges_gidx)
    # Create sigma_t + 1 / (v * dt)
    tools._total_velocity(xs_total_u, velocity_u, info_u)
    tools._total_velocity(xs_total_c, velocity_c, info_c)
    # Indexing Parameters
    coarse_idx, fine_idx, factor = hybrid_index(info_u.groups, \
                                    info_c.groups, edges_g, edges_gidx)
    # Run Backward Euler
    flux = multigroup_bdf1(xs_total_u, xs_scatter_u, velocity_u, external_u, \
                boundary_u, medium_map, delta_x, angle_xu, angle_wu, \
                xs_total_c, xs_scatter_c, velocity_c, angle_xc, angle_wc, \
                fine_idx, coarse_idx, factor, info_u, info_c)
    return np.asarray(flux)


cdef double[:,:,:] multigroup_bdf1(double[:,:]& xs_total_u, \
        double[:,:,:]& xs_scatter_u, double[:]& velocity_u, double[:]& external_u, \
        double[:]& boundary_u, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_xu, double[:]& angle_wu, double[:,:]& xs_total_c, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_c, double[:]& angle_xc, \
        double[:]& angle_wc, int[:]& fine_idx, int[:]& coarse_idx, \
        double[:]& factor, params info_u, params info_c):
    # Initialize time step
    cdef int step
    # Combine last time step and uncollided source term
    q_star = tools.array_1d((info_u.cells_x + info_u.edges) \
                            * info_u.angles * info_u.groups)
    # Initialize angular flux for previous time step
    flux_last = tools.array_3d(info_u.cells_x + info_u.edges, \
                            info_u.angles, info_u.groups)
    # Initialize uncollided scalar flux
    flux_u = tools.array_2d(info_u.cells_x + info_u.edges, info_u.groups)
    # Initialize collided scalar flux
    flux_c = tools.array_2d(info_c.cells_x + info_c.edges, info_c.groups)
    # Initialize total scalar flux
    flux_t = tools.array_2d(info_u.cells_x + info_u.edges, info_u.groups)
    # Initialize array with all scalar flux time steps
    flux_time = tools.array_3d(info_u.steps, info_u.cells_x + info_u.edges, info_u.groups)
    # Initialize collided source
    source_c = tools.array_1d((info_c.cells_x + info_c.edges) * info_c.groups)
    # Initialize collided boundary
    cdef double[2] boundary_c = [0.0, 0.0]
    # Iterate over time steps
    for step in tqdm(range(info_u.steps)):
        # Adjust boundary condition
        tools.boundary_decay(boundary_u, step, info_u)
        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star(flux_last, q_star, external_u, velocity_u, info_u)
        # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
        flux_u = tools._angular_to_scalar(mg._known_source(xs_total_u, q_star, boundary_u, \
                        medium_map, delta_x, angle_xu, angle_wu, info_u), angle_wu, info_u)
        # Step 2: Compute collided source (I x G')
        tools._hybrid_source_collided(flux_u, xs_scatter_u, source_c, \
                                medium_map, coarse_idx, info_u, info_c)
        # Step 3: Solve Collided Equation (I x G')
        flux_c = mg.source_iteration(flux_c, xs_total_c, xs_scatter_c, \
                                    source_c, boundary_c, medium_map, \
                                    delta_x, angle_xc, angle_wc, info_c)
        # Step 4: Create a new source and solve for angular flux
        tools._expand_hybrid_source(flux_t, flux_c, fine_idx, factor, info_u, info_c)
        tools._hybrid_source_total(flux_t, flux_u, xs_scatter_u, q_star, \
                            medium_map, fine_idx, factor, info_u, info_c)
        # Solve for angular flux of time step
        flux_last = mg._known_source(xs_total_u, q_star, boundary_u, \
                        medium_map, delta_x, angle_xu, angle_wu, info_u)
        # Step 5: Update and repeat
        flux_time[step] = tools._angular_to_scalar(flux_last, angle_wu, info_u)
    return flux_time[:,:,:]