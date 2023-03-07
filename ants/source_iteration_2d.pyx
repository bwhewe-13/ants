########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Source iteration for two-dimensional multigroup neutron transport 
# problems.
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

from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d
from ants.constants import *

from libc.math cimport fabs #, sqrt, pow
# from libc.stdlib cimport abs as fabs 
# cimport cython
# from cython.view cimport array as cvarray
# from cython.parallel import prange

# import numpy as np

cdef double[:,:,:] multigroup_angular(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& external, \
        double [:]& boundary_x, double [:]& boundary_y, int[:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params2d params):
    # Initialize components
    cdef size_t group, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    # Set indexing
    qq2 = 1 if params.qdim == 1 else params.groups
    bcx2 = 1 if params.bcdim_x < 2 else params.groups
    bcy2 = 1 if params.bcdim_y < 2 else params.groups
    # Initialize flux
    flux = tools.array_3d_ijng(params)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_2d_ijn(params)
    # Create off-scattering term
    off_scatter = tools.array_1d_ij(params)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:,:] = 0.0
        for group in range(params.groups):
            qq1 = 0 if params.qdim == 1 else group
            bcx1 = 0 if params.bcdim_x < 2 else group
            bcy1 = 0 if params.bcdim_y < 2 else group
            flux_1g[:,:] = flux_old[:,:,group]
            tools.off_scatter_angular(flux, flux_old, medium_map, xs_scatter, \
                                      off_scatter, angle_w, params, group)
            ordinates_angular(flux[:,:,group], flux_1g, xs_total[:,group], \
                              xs_scatter[:,group,group], off_scatter, \
                              external[qq1::qq2], boundary_x[bcx1::bcx2], \
                              boundary_y[bcy1::bcy2], medium_map, delta_x, \
                              delta_y, angle_x, angle_y, angle_w, params)
        change = tools.group_convergence_angular(flux, flux_old, angle_w, params)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    return flux[:,:,:]


cdef void ordinates_angular(double[:,:] flux, double[:,:] flux_old, \
        double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
        double[:] external, double[:] boundary_x, double[:] boundary_y, \
        int[:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        params2d params):
    if params.geometry == 1: # square
        square_ordinates_angular(flux, flux_old, xs_total, xs_scatter, \
                                 off_scatter, external, boundary_x, \
                                 boundary_y, medium_map, delta_x, delta_y, \
                                 angle_x, angle_y, angle_w, params)


cdef void square_ordinates_angular(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params2d params):
    # Initialize indices etc
    cdef size_t angle, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    scalar_flux = tools.array_1d_ij(params)
    # Set indexing
    qq2 = 1 if params.qdim != 3 else params.angles
    bcx2 = 1 if params.bcdim_x != 3 else params.angles
    bcy2 = 1 if params.bcdim_y != 3 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:] = 0.0
        tools.angle_angular_to_scalar(flux_old, scalar_flux, angle_w, params)
        # for angle in prange(params.angles, nogil=True):
        for angle in range(params.angles):
            qq1 = 0 if params.qdim != 3 else angle
            bcx1 = 0 if params.bcdim_x != 3 else params.angles
            bcy1 = 0 if params.bcdim_y != 3 else params.angles
            spatial_sweep(flux[:,angle], scalar_flux, xs_total, xs_scatter, \
                          off_scatter, external[qq1::qq2], boundary_x[bcx1::bcx2], \
                          boundary_y[bcy1::bcy2], medium_map, delta_x, delta_y, \
                          angle_x[angle], angle_y[angle], 1.0, params)
        change = tools.angle_convergence_angular(flux, flux_old, angle_w, params)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]


cdef void spatial_sweep(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params2d params):
    edge_y = tools.update_y_edge(boundary_y, angle_y, params)
    if angle_y > 0.0:
        square_forward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, boundary_x, edge_y, medium_map, delta_x, \
                         delta_y, angle_x, angle_y, angle_w, params)
    elif angle_y < 0.0:
        square_backward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                          external, boundary_x, edge_y, medium_map, delta_x, \
                          delta_y, angle_x, angle_y, angle_w, params)


cdef void square_forward_y(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params2d params):
    cdef size_t cell, qq1, qq2, bcx1, bcx2
    cdef double coef_y, edge
    qq2 = 1 if params.qdim == 0 else params.cells_y
    bcx2 = 1 if params.bcdim_x == 0 else params.cells_y
    for cell in range(params.cells_y):
        qq1 = 0 if params.qdim == 0 else cell
        bcx1 = 0 if params.bcdim_x == 0 else cell
        coef_y = 2 * fabs(angle_y) / delta_y[cell]
        if angle_x > 0.0:
            edge = square_forward_x(flux[cell::params.cells_y], \
                                    flux_old[cell::params.cells_y], xs_total, \
                                    xs_scatter, off_scatter[cell::params.cells_y], \
                                    external[qq1::qq2], boundary_x[bcx1::bcx2][0], \
                                    edge_y, medium_map[cell::params.cells_y], \
                                    delta_x, angle_x, angle_w, coef_y, params)
        elif angle_x < 0.0:
            edge = square_backward_x(flux[cell::params.cells_y], \
                                     flux_old[cell::params.cells_y], xs_total, \
                                     xs_scatter, off_scatter[cell::params.cells_y], \
                                     external[qq1::qq2], boundary_x[bcx1::bcx2][1], \
                                     edge_y, medium_map[cell::params.cells_y], \
                                     delta_x, angle_x, angle_w, coef_y, params)


cdef double square_forward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x,  double angle_x, \
        double angle_w, double coef_y, params2d params):
    cdef size_t cell, mat #, qq
    cdef double center, edge1, coef_x
    edge1 = edge_x
    for cell in range(params.cells_x):
        mat = medium_map[cell]
        # qq = cell if params.qdim != 0 else 0
        coef_x = (2 * fabs(angle_x) / delta_x[cell])
        center = (coef_x * edge1 + coef_y * edge_y[cell] \
                + xs_scatter[mat] * flux_old[cell] + external[cell] \
                + off_scatter[cell]) / (xs_total[mat] + coef_x + coef_y)
        flux[cell] += angle_w * center
        edge1 = 2 * center - edge1
        edge_y[cell] = 2 * center - edge_y[cell]
    return edge1


cdef void square_backward_y(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params2d params):
    cdef size_t cell, qq1, qq2, bcx1, bcx2
    cdef double coef_y, edge
    qq2 = 1 if params.qdim == 0 else params.cells_y
    bcx2 = 1 if params.bcdim_x == 0 else params.cells_y
    for cell in range(params.cells_y-1, -1, -1):
        qq1 = 0 if params.qdim == 0 else cell
        bcx1 = 0 if params.bcdim_x == 0 else cell
        coef_y = 2 * fabs(angle_y) / delta_y[cell]
        if angle_x > 0.0:
            edge = square_forward_x(flux[cell::params.cells_y], \
                                    flux_old[cell::params.cells_y], xs_total, \
                                    xs_scatter, off_scatter[cell::params.cells_y], \
                                    external[qq1::qq2], boundary_x[bcx1::bcx2][0], \
                                    edge_y, medium_map[cell::params.cells_y], \
                                    delta_x, angle_x, angle_w, coef_y, params)
        elif angle_x < 0.0:
            edge = square_backward_x(flux[cell::params.cells_y], \
                                     flux_old[cell::params.cells_y], xs_total, \
                                     xs_scatter, off_scatter[cell::params.cells_y], \
                                     external[qq1::qq2], boundary_x[bcx1::bcx2][1], \
                                     edge_y, medium_map[cell::params.cells_y], \
                                     delta_x, angle_x, angle_w, coef_y, params)


cdef double square_backward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x,  double angle_x, \
        double angle_w, double coef_y, params2d params):
    cdef size_t cell, mat #, qq
    cdef double center, edge1, coef_x
    edge1 = edge_x
    for cell in range(params.cells_x-1, -1, -1):
        mat = medium_map[cell]
        # qq = cell if params.qdim != 0 else 0
        coef_x = (2 * fabs(angle_x) / delta_x[cell])
        center = (coef_x * edge1 + coef_y * edge_y[cell] \
                + xs_scatter[mat] * flux_old[cell] + external[cell] \
                + off_scatter[cell]) / (xs_total[mat] + coef_x + coef_y)
        flux[cell] += angle_w * center
        edge1 = 2 * center - edge1
        edge_y[cell] = 2 * center - edge_y[cell]
    return edge1

cdef double[:,:] multigroup_scalar(double[:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:]& external, \
        double [:]& boundary_x, double [:]& boundary_y, int[:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params2d params):
    # Initialize components
    cdef size_t group, qq1, qq2 #, bcx1, bcx2, bcy1, bcy2
    # Set indexing
    qq2 = 1 if params.qdim == 1 else params.groups
    # bcx2 = 1 if params.bcdim_x < 2 else params.groups
    # bcy2 = 1 if params.bcdim_y < 2 else params.groups
    # Initialize flux
    flux = tools.array_2d_ijg(params)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_1d_ij(params)
    # Create off-scattering term
    off_scatter = tools.array_1d_ij(params)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:] = 0.0
        for group in range(params.groups):
            qq1 = 0 if params.qdim == 1 else group
            # bcx1 = 0 if params.bcdim_x < 2 else group
            # bcy1 = 0 if params.bcdim_y < 2 else group
            flux_1g[:] = flux_old[:,group]
            tools.off_scatter_scalar(flux, flux_old, medium_map, xs_scatter, \
                                     off_scatter, params, group)
            ordinates_scalar(flux[:,group], flux_1g, xs_total[:,group], \
                             xs_scatter[:,group,group], off_scatter, \
                             external[qq1::qq2], boundary_x, boundary_y, \
                             medium_map, delta_x, delta_y, angle_x, angle_y, \
                             angle_w, params)
        change = tools.group_convergence_scalar(flux, flux_old, params)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    return flux[:,:]

cdef void ordinates_scalar(double[:] flux, double[:] flux_old, \
        double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
        double[:] external, double[:] boundary_x, double[:] boundary_y, \
        int[:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        params2d params):
    if params.geometry == 1: # square
        square_ordinates_scalar(flux, flux_old, xs_total, xs_scatter, \
                                off_scatter, external, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, params)

cdef void square_ordinates_scalar(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params2d params):
    # Initialize indices etc
    cdef size_t angle, qq1, qq2 #, bcx1, bcx2, bcy1, bcy2
    # Set indexing
    qq2 = 1 if params.qdim != 3 else params.angles
    # bcx2 = 1 if params.bcdim_x != 3 else params.angles
    # bcy2 = 1 if params.bcdim_y != 3 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:] = 0.0
        # for angle in prange(params.angles, nogil=True):
        for angle in range(params.angles):
            qq1 = 0 if params.qdim != 3 else angle
            # bcx1 = 0 if params.bcdim_x != 3 else params.angles
            # bcy1 = 0 if params.bcdim_y != 3 else params.angles
            spatial_sweep(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                          external[qq1::qq2], boundary_x, boundary_y, \
                          medium_map, delta_x, delta_y, angle_x[angle], \
                          angle_y[angle], angle_w[angle], params)
        change = tools.angle_convergence_scalar(flux, flux_old, params)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]

# cdef void spatial_sweep(double[:]& flux, double[:]& flux_old, \
#             double[:]& xs_total, double[:]& xs_scatter, double[:] off_scatter, \
#             double[:] external, double[:] boundary, int[:] medium_map, \
#             double[:] delta_x, double[:] delta_y, double angle_x, \
#             double angle_y, double angle_w, size_t angle, params2d params) nogil:
#     cdef double edge1, edge2, center, weight
#     cdef size_t ii, xx, jj, yy, mat
#     cdef size_t ff1, ff2, qq1, qq2, bc1, bc2
#     # Initialize Weight
#     weight = 1 if params.angular == True else 0.25*angle_w
#     # Initialize index parameters
#     ff1 = 0 if params.angular == False else angle
#     ff2 = params.angles if params.angular == True else 1
#     qq1 = 0 if params.qdim != 3 else angle
#     qq2 = 1 if params.qdim != 3 else params.angles
#     # bc1 = 0 if params.bcdim != 2 else angle
#     # bc2 = 1 if params.bcdim != 2 else params.angles
#     # Initialize boundary condition
#     cdef double[:] boundary_y
#     with cython.gil:
#         arr1d = cvarray((params.cells_x,), itemsize=sizeof(double), format="d")
#         boundary_y = arr1d
#         boundary_y[:] = 0.0
#     # Sweep over X and Y directions
#     for jj in range(params.cells_y):
#         yy = sweep_direction(angle_y, params.cells_y, jj)
#         edge1 = 0.0
#         for ii in range(params.cells_x):
#             xx = sweep_direction(angle_x, params.cells_x, ii)
#             mat = medium_map[yy::params.cells_y][xx]
#             center = ((2 * fabs(angle_x) / delta_x[xx]) * edge1 \
#                     + (2 * fabs(angle_y) / delta_y[yy]) * boundary_y[xx] \
#                     + xs_scatter[mat] * flux_old[yy::params.cells_y][xx] \
#                     + external[qq1::qq2][yy::params.cells_y][xx] \
#                     + off_scatter[yy::params.cells_y][xx]) \
#                     / (xs_total[mat] + (2 * fabs(angle_x) / delta_x[xx]) \
#                     + (2 * fabs(angle_y) / delta_y[yy]))
#             flux[ff1::ff2][yy::params.cells_y][xx] += weight * center
#             edge1 = 2 * center - edge1
#             boundary_y[xx] = 2 * center - boundary_y[xx]

# cdef int sweep_direction(double angle, int cells, int cell) nogil:
#     if angle > 0.0:
#         return cell
#     elif angle < 0.0:
#         return cells - (cell + 1)