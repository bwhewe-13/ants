########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

from ants cimport source_iteration_1d as si
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d

import numpy as np

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
                        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
                        double[:]& velocity, double[:]& source, \
                        double[:]& boundary, int[:]& medium_map, \
                        double[:]& delta_x, double[:]& angle_x, \
                        double[:]& angle_w, params1d params):
    cdef size_t step
    # Combine last time step and source term
    source_star = tools.array_1d_ing(params)
    # Initialize fluxes
    flux_last = flux_guess.copy()
    flux_times = tools.array_4d_ting(params)
    for step in range(params.steps):
        tools.combine_source_flux(flux_last, source_star, \
                                    source, velocity, params)
        flux_times[step] = si.multigroup_angular(flux_last, xs_total_v, \
                        xs_scatter, source_star, boundary, medium_map, \
                        delta_x, angle_x, angle_w, params)
        flux_last[:,:,:] = flux_times[step,:,:,:]
        print("Multigroup", step, np.sum(flux_last))
    return flux_times[:,:,:,:]


cdef double[:,:,:,:] hybrid_bdf1(double[:,:]& xs_total_vu, double[:,:]& xs_total_vc, \
                double[:,:,:]& xs_scatter_u, double[:,:,:]& xs_scatter_c, \
                double[:]& velocity_u, double[:]& velocity_c, \
                double[:]& source, double[:]& boundary, int[:]& medium_map, \
                double[:]& delta_x, double[:]& angle_xu, double[:]& angle_wu, \
                double[:]& angle_xc, double[:]& angle_wc, \
                int[:]& index_u, int[:]& index_c, double[:]& factor_u, \
                params1d params_u, params1d params_c):
    cdef size_t step
    # Combine last time step and source term
    source_star = tools.array_1d_ing(params_u)
    source_c = tools.array_1d_ig(params_c)
    # Initialize fluxes
    flux_last = tools.array_3d_ing(params_u)
    flux_uncollided = tools.array_3d_ing(params_u)
    flux_collided = tools.array_3d_ing(params_c)
    zero_matrix = tools.array_3d_mgg(params_u)

    flux_times = tools.array_4d_ting(params_u)
    for step in range(params_u.steps):
        # Step 1: Solve Uncollided Equation (I x N x G)
        tools.combine_source_flux(flux_last, source_star, \
                                    source, velocity_u, params_u)
        flux_uncollided = si.multigroup_angular(flux_last, xs_total_vu, \
                        zero_matrix, source_star, boundary, medium_map, \
                        delta_x, angle_xu, angle_wu, params_u)
        # print("uncollided flux", np.sum(flux_uncollided))
        # Step 2: Compute collided source
        tools.calculate_collided_source(flux_uncollided, xs_scatter_u, \
            source_c, medium_map, angle_wu, index_c, params_u, params_c)
        # print("collided source", np.sum(source_c))
        # Step 3: Solve Collided Equation
        # print(np.asarray(xs_total_vc))
        flux_collided = si.multigroup_angular(flux_collided, xs_total_vu, \
                        xs_scatter_u, source_c, boundary, medium_map, \
                        delta_x, angle_xu, angle_wu, params_c)
        # print("collided flux", np.sum(flux_collided))
        # Step 4: Create a new source and solve for angular flux
        tools.calculate_total_source(flux_uncollided, flux_collided, \
            xs_scatter_u, source_star, medium_map, angle_wu, angle_wu, \
            index_u, factor_u, params_u, params_c)
        tools.combine_source_flux(flux_last, source_star, \
                            source, velocity_u, params_u)
        # Solve for angular flux
        flux_times[step] = si.multigroup_angular(flux_last, xs_total_vu, \
                        zero_matrix, source_star, boundary, medium_map, \
                        delta_x, angle_xu, angle_wu, params_u)
        # Step 5: Update and repeat
        flux_last[:,:,:] = flux_times[step,:,:,:]
        print("Hybrid", step, np.sum(flux_last))
    return flux_times[:,:,:,:]