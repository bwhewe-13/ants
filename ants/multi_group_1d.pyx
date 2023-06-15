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

from ants.spatial_sweep_1d cimport discrete_ordinates, _known_sweep
from ants cimport cytools_1d as tools
from ants.parameters cimport params
from ants.constants import *

from libc.math cimport isnan, isinf

cdef double[:,:] source_iteration(double[:,:]& flux_guess, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& external, double [:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    # Initialize components
    cdef int gg, qq1, qq2, bc1, bc2
    # Set indexing
    qq2 = 1 if info.qdim == 1 else info.groups
    bc2 = 1 if info.bcdim_x == 1 else info.groups
    # Initialize flux
    flux = tools.array_2d(info.cells_x + info.edges, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_1d(info.cells_x + info.edges)
    # Create off-scattering term
    off_scatter = tools.array_1d(info.cells_x + info.edges)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    # Iterate until energy group convergence
    while not (converged):
        flux[:,:] = 0.0
        for gg in range(info.groups):
            qq1 = 0 if info.qdim == 1 else gg
            bc1 = 0 if info.bcdim_x == 1 else gg
            # Select the specific group from last iteration
            flux_1g[:] = flux_old[:,gg]
            # Calculate up and down scattering term using Gauss-Seidel
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)
            # Use discrete ordinates for the angular dimension
            discrete_ordinates(flux[:,gg], flux_1g, xs_total[:,gg], \
                               xs_scatter[:,gg,gg], off_scatter, external[qq1::qq2], \
                               boundary_x[bc1::bc2], medium_map, delta_x, \
                               angle_x, angle_w, info)
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < EPSILON_ENERGY) or (count >= MAX_ENERGY)
        count += 1
        flux_old[:,:] = flux[:,:]
    return flux[:,:]


cdef double[:,:,:] _known_source(double[:,:]& xs_total, double[:]& source, \
        double [:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):
    # source = flux * xs_scatter + external source
    # Initialize components
    cdef int gg, q1, qq2, bc1, bc2
    # Set indexing
    qq2 = 1 if info.qdim == 1 else info.groups
    bc2 = 1 if info.bcdim_x == 1 else info.groups
    # Initialize angular flux
    angular_flux = tools.array_3d(info.cells_x + info.edges, info.angles, info.groups)
    for gg in range(info.groups):
        qq1 = 0 if info.qdim == 1 else gg
        bc1 = 0 if info.bcdim_x == 1 else gg
        _known_sweep(angular_flux[:,:,gg], xs_total[:,gg], source[qq1::qq2], \
            boundary_x[bc1::bc2], medium_map, delta_x, angle_x, angle_w, info)
    return angular_flux[:,:,:]
