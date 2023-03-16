########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

from ants cimport source_iteration_2d as si
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d

# import numpy as np
from tqdm import tqdm

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:]& external, double[:]& boundary_x, \
        double[:]& boundary_y, int[:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params2d params):
    cdef size_t step
    # Combine last time step and source term
    source_star = tools.array_1d_ijng(params)
    # Initialize fluxes
    flux_last = flux_guess.copy()
    flux_times = tools.array_4d_tijng(params)
    # for step in range(params.steps):
    for step in tqdm(range(params.steps)):
        tools.combine_source_flux(flux_last, source_star, external, \
                                  velocity, params)
        flux_times[step] = si.multigroup_angular(flux_last, xs_total_v, \
                                                 xs_scatter, source_star, \
                                                 boundary_x, boundary_y, \
                                                 medium_map, delta_x, \
                                                 delta_y, angle_x, angle_y, \
                                                 angle_w, params)
        boundary_x[:] = 0.0
        boundary_y[:] = 0.0
        flux_last[:,:,:] = flux_times[step,:,:,:]
        # print("Multigroup", step, np.sum(flux_last))
    return flux_times[:,:,:,:]


cdef double[:,:,:,:] hybrid_bdf1(double[:,:]& xs_total_vu, \
        double[:,:]& xs_total_vc, double[:,:,:]& xs_scatter_u, \
        double[:,:,:]& xs_scatter_c, double[:]& velocity_u, \
        double[:]& velocity_c, double[:]& external_u, double[:]& boundary_x, \
        double[:]& boundary_y, int[:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_xu, double[:]& angle_yu, \
        double[:]& angle_wu, double[:]& angle_xc, double[:]& angle_yc, \
        double[:]& angle_wc, int[:]& index_u, int[:]& index_c, \
        double[:]& factor_u, params2d params_u, params2d params_c):
    cdef size_t step
    # Combine last time step and source term
    source_star = tools.array_1d_ijng(params_u)
    source_c = tools.array_1d_ijg(params_c)
    source_t = tools.array_1d_ijg(params_u)
    # Initialize fluxes
    flux_last = tools.array_3d_ijng(params_u)
    flux_uncollided = tools.array_2d_ijg(params_u)
    flux_collided = tools.array_2d_ijg(params_c)

    zero_matrix = tools.array_3d_mgg(params_u)
    flux_times = tools.array_4d_tijng(params_u)

    cdef double[2] zero_boundary = [0.0, 0.0]
    for step in tqdm(range(params_u.steps)):
    # for step in range(params_u.steps):
        # print("start of loop")
        # Create source star
        # print(flux_last.shape, source_star.shape, external_u.shape, velocity_u.shape)
        tools.combine_source_flux(flux_last, source_star, external_u, \
                                  velocity_u, params_u)
        # print("after combine_source_flux")
        # Step 1: Solve Uncollided Equation (I x N x G)
        flux_uncollided = si.multigroup_scalar(flux_uncollided, xs_total_vu, \
                                               zero_matrix, source_star, \
                                               boundary_x, boundary_y, \
                                               medium_map, delta_x, delta_y, \
                                               angle_xu, angle_yu, angle_wu, \
                                               params_u)
        # print("after uncollided flux")
        # Step 2: Compute collided source
        tools.calculate_source_c(flux_uncollided, xs_scatter_u, source_c, \
                                 medium_map, index_c, params_u, params_c)
        # print("calculate source_c ")
        # Step 3: Solve Collided Equation
        flux_collided = si.multigroup_scalar(flux_collided, xs_total_vc, \
                                             xs_scatter_c, source_c, \
                                             zero_boundary, zero_boundary, \
                                             medium_map, delta_x, delta_y, \
                                             angle_xc, angle_yc, angle_wc, \
                                             params_c)
        # print("solved collided equation")
        # Step 4: Create a new source and solve for angular flux
        tools.calculate_source_t(flux_uncollided, flux_collided, \
                                 xs_scatter_u, source_t, medium_map, \
                                 index_u, factor_u, params_u, params_c)
        # print("calculated source t")
        tools.calculate_source_star(flux_last, source_star, source_t, \
                                    external_u, velocity_u, params_u)
        # print("calculated source star")
        # Solve for angular flux
        flux_times[step] = si.multigroup_angular(flux_last, xs_total_vu, \
                                                 zero_matrix, source_star, \
                                                 boundary_x, boundary_y, \
                                                 medium_map, delta_x, \
                                                 delta_y, angle_xu, angle_yu, \
                                                 angle_wu, params_u)
        # Step 5: Update and repeat
        flux_last[:,:,:] = flux_times[step,:,:,:]
        boundary_x[:] = 0.0
        boundary_y[:] = 0.0
        # print("Hybrid", step, np.sum(flux_last))
    return flux_times[:,:,:,:]
