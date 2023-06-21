########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Solution for two-dimensional multigroup neutron transport problems.
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

from ants.spatial_sweep_2d cimport discrete_ordinates #, _known_sweep
from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants.constants import *


cdef double[:,:,:] source_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& external, \
        double [:]& boundary_x, double [:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    # Initialize components
    cdef int gg, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    # Set indexing
    qq2 = 1 if info.qdim == 1 else info.groups
    bcx2 = 1 if info.bcdim_x <= 2 else info.groups
    bcy2 = 1 if info.bcdim_y <= 2  else info.groups
    # Initialize flux
    flux = tools.array_3d(info.cells_x + info.edges, info.cells_y \
                          + info.edges, info.groups)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_2d(info.cells_x + info.edges, info.cells_y + info.edges)
    # Create off-scattering term
    off_scatter = tools.array_2d(info.cells_x + info.edges, info.cells_y + info.edges)
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:,:] = 0.0
        for gg in range(info.groups):
            qq1 = 0 if info.qdim == 1 else gg
            bcx1 = 0 if info.bcdim_x <= 2 else gg
            bcy1 = 0 if info.bcdim_y <= 2 else gg
            flux_1g[:,:] = flux_old[:,:,gg]
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                               off_scatter, info, gg)
            discrete_ordinates(flux[:,:,gg], flux_1g, xs_total[:,gg], \
                xs_scatter[:,gg,gg], off_scatter, external[qq1::qq2], \
                boundary_x[bcx1::bcx2], boundary_y[bcy1::bcy2], medium_map, \
                delta_x, delta_y, angle_x, angle_y, angle_w, info)
        change = tools.group_convergence(flux, flux_old, info)
        if isnan(change) or isinf(change):
            change = 0.5
        converged = (change < EPSILON_ENERGY) or (count >= MAX_ENERGY)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    return flux[:,:,:]
