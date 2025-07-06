########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Solution for one-dimensional multigroup neutron transport problems.
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

from libc.math cimport isnan, isinf

from ants.spatial_sweep_1d cimport discrete_ordinates, _known_sweep
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants.utils.pytools import dmd_1d

import numpy as np


cdef double[:,:] multi_group(double[:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    # Source Iteration
    if info.mg == 1:
        return source_iteration(flux_guess, xs_total, xs_scatter, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    # Dynamic Mode Decomposition
    elif info.mg == 2:
        return dynamic_mode_decomp(flux_guess, xs_total, xs_scatter, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)


cdef double[:,:] source_iteration(double[:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    
    # Initialize components
    cdef int gg, qq, bc
    
    # Initialize flux
    flux = tools.array_2d(info.cells_x, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_1d(info.cells_x)
    
    # Create off-scattering term
    off_scatter = tools.array_1d(info.cells_x)
    
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    
    # Iterate until energy group convergence
    while not (converged):

        # Zero out flux
        flux[:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary_x.shape[2] == 1 else gg
            
            # Select the specific group from last iteration
            flux_1g[:] = flux_old[:,gg]
            
            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)
            
            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,qq], \
                    boundary_x[:,:,bc], medium_map, delta_x, angle_x, \
                    angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.change_gg) or (count >= info.count_gg)
        count += 1

        # Update old flux
        flux_old[:,:] = flux[:,:]

    return flux[:,:]


cdef double[:,:] variable_source_iteration(double[:,:]& flux_guess, \
        double[:,:]& xs_total_u, double[:]& star_coef_c, \
        double[:,:,:]& xs_scatter_u, double[:,:,:]& external, \
        double[:,:,:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, double[:]& edges_g, \
        int[:]& edges_gidx_c, params info):
    
    # Initialize components
    cdef int gg, qq, bc, idx1, idx2
    
    # Initialize flux
    flux = tools.array_2d(info.cells_x, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_1d(info.cells_x)
    
    # Create collided and off-scattering terms
    xs_total_c = tools.array_1d(info.materials)
    xs_scatter_c = tools.array_1d(info.materials)
    off_scatter = tools.array_1d(info.cells_x)
    
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    
    # Iterate until energy group convergence
    while not (converged):

        # Zero out flux
        flux[:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary_x.shape[2] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg+1]

            tools._variable_cross_sections(xs_total_c, xs_total_u, star_coef_c[gg], \
                        xs_scatter_c, xs_scatter_u, edges_g, idx1, idx2, info)
                        
            # Select the specific group from last iteration
            flux_1g[:] = flux_old[:,gg]

            # Calculate up and down scattering term using Gauss-Seidel
            tools._variable_off_scatter(flux, flux_old, medium_map, xs_scatter_u, \
                                        off_scatter, gg, edges_g, edges_gidx_c, \
                                        idx1, idx2, info)

            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,gg], flux_1g, xs_total_c, xs_scatter_c, \
                    off_scatter, external[:,:,qq], boundary_x[:,:,bc], \
                    medium_map, delta_x, angle_x, angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.change_gg) or (count >= info.count_gg)
        count += 1

        # Update old flux
        flux_old[:,:] = flux[:,:]

    return flux[:,:]


cdef double[:,:] dynamic_mode_decomp(double[:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    
    # Initialize components
    cdef int gg, rk, kk, qq, bc
    
    # Initialize flux
    flux = tools.array_2d(info.cells_x, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_1d(info.cells_x)

    # Initialize Y_plus and Y_minus
    y_plus = tools.array_3d(info.cells_x, info.groups, info.dmd_k - 1)
    y_minus = tools.array_3d(info.cells_x, info.groups, info.dmd_k - 1)
    
    # Create off-scattering term
    off_scatter = tools.array_1d(info.cells_x)
    
    # Set convergence limits
    cdef bint converged = False
    cdef double change = 0.0
    
    # Iterate over removed source iterations
    for rk in range(info.dmd_r + info.dmd_k):

        # Return flux if there is convergence
        if converged:
            return flux[:,:]

        # Zero out flux
        flux[:,:] = 0.0

        # Iterate over energy groups
        for gg in range(info.groups):

            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary_x.shape[2] == 1 else gg
            
            # Select the specific group from last iteration
            flux_1g[:] = flux_old[:,gg]
            
            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)
            
            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,gg], flux_1g, xs_total[:,gg], \
                    xs_scatter[:,gg,gg], off_scatter, external[:,:,qq], \
                    boundary_x[:,:,bc], medium_map, delta_x, angle_x, \
                    angle_w, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < info.change_gg)

        # Collect difference for DMD on K iterations
        if rk >= info.dmd_r:
            # Get indexing
            kk = rk - info.dmd_r
            tools._dmd_subtraction(y_minus, y_plus, flux, flux_old, kk, info)
        
        # Update old flux
        flux_old[:,:] = flux[:,:]

    # Perform DMD
    flux = dmd_1d(flux, y_minus, y_plus, info.dmd_k)

    return flux[:,:]


cdef double[:,:,:] _known_source_angular(double[:,:]& xs_total, \
        double[:,:,:]& source, double[:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source
    
    # Initialize components
    cdef int gg, qq, bc
    
    # Initialize angular flux
    angular_flux = tools.array_3d(info.cells_x + info.edges, info.angles, info.groups)
    
    # Set zero matrix placeholder for scattering
    zero = tools.array_1d(info.cells_x + info.edges)
    
    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary_x.shape[2] == 1 else gg

        # Perform angular sweep
        _known_sweep(angular_flux[:,:,gg], xs_total[:,gg], zero, \
                     source[:,:,qq], boundary_x[:,:,bc], medium_map, \
                     delta_x, angle_x, angle_w, info)
    
    return angular_flux[:,:,:]


cdef double[:,:] _known_source_scalar(double[:,:]& xs_total, \
        double[:,:,:]& source, double[:,:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source
    
    # Initialize components
    cdef int gg, qq, bc
    
    # Initialize angular flux
    scalar_flux = tools.array_3d(info.cells_x + info.edges, info.groups, 1)
    
    # Set zero matrix placeholder for scattering
    zero = tools.array_1d(info.cells_x + info.edges)
    
    # Iterate over groups
    for gg in range(info.groups):

        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary_x.shape[2] == 1 else gg

        # Perform angular sweep
        _known_sweep(scalar_flux[:,gg], xs_total[:,gg], zero, \
                     source[:,:,qq], boundary_x[:,:,bc], medium_map, \
                     delta_x, angle_x, angle_w, info)

    return scalar_flux[:,:,0]