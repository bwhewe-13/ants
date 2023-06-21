########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Spatial sweeps for two-dimensional neutron transport problems.
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

# from libc.math cimport fabs
# from cython.parallel import prange

from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants.constants import *


cdef void discrete_ordinates(double[:,:]& flux, double[:,:]& flux_old, 
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    # Rectangular spatial cells
    if info.geometry == 1:
        square_ordinates(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, boundary_x, boundary_y, medium_map, \
                         delta_x, delta_y, angle_x, angle_y, angle_w, info)


cdef void square_ordinates(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& boundary_x, double[:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, params info):
    # Initialize indices etc
    cdef int nn, qq1, qq2, bcx1, bcx2, bcy1, bcy2
    # Set indexing
    qq2 = 1 if info.qdim != 3 else info.angles * info.angles
    bcx2 = 1 if info.bcdim_x != 4 else info.angles * info.angles
    bcy2 = 1 if info.bcdim_y != 4 else info.angles * info.angles
    # Initialize unknown cell edge
    cdef double edge = 0.0
    # Add reflector array
    known_y = tools.array_1d(info.cells_x)
    known_x = tools.array_1d(info.cells_y)
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:] = 0.0
        for nn in range(info.angles * info.angles):
            qq1 = 0 if info.qdim != 3 else nn
            bcx1 = 0 if info.bcdim_x != 4 else nn
            bcy1 = 0 if info.bcdim_y != 4 else nn
            # Initialize known x and y
            tools._initialize_edge_y(known_y, boundary_y[bcy1::bcy2], \
                                     angle_y, angle_x, nn, info)
            tools._initialize_edge_x(known_x, boundary_x[bcx1::bcx2], \
                                     angle_x, angle_y, nn, info)
            # Perform spatial sweep
            square_sweep(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                    external[qq1::qq2], known_x, known_y, medium_map, delta_x, \
                    delta_y, angle_x[nn], angle_y[nn], angle_w[nn], info)
        change = tools.angle_convergence(flux, flux_old, info)
        converged = (change < EPSILON_ANGULAR) or (count >= MAX_ANGULAR)
        count += 1
        flux_old[:,:] = flux[:,:]


cdef void square_sweep(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    if angle_y > 0.0:
        square_forward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, known_x, known_y, medium_map, delta_x, \
                         delta_y, angle_x, angle_y, angle_w, info)
    elif angle_y < 0.0:
        square_backward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                          external, known_x, known_y, medium_map, delta_x, \
                          delta_y, angle_x, angle_y, angle_w, info)


cdef void square_forward_y(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    # Initialize iterables
    cdef int jj, cell
    cdef double coef_y, const, edge_x
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Determine flux edges
    if info.edges:
        flux[:,0] = known_x[:]
    # Iterate over Y spatial cells
    for jj in range(info.cells_y):
        coef_y = const * angle_y / delta_y[jj]
        cell = jj + 1 if info.edges else jj
        if angle_x > 0.0:
            known_x[jj] = square_forward_x(flux[:,cell], flux_old[:,cell], \
                                xs_total, xs_scatter, off_scatter[:,jj], \
                                external[jj::info.cells_y], known_x[jj], \
                                known_y, medium_map[:,jj], delta_x, angle_x, \
                                angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = square_backward_x(flux[:,cell], flux_old[:,cell], \
                                xs_total, xs_scatter, off_scatter[:,jj], \
                                external[jj::info.cells_y], known_x[jj], \
                                known_y, medium_map[:,jj], delta_x, angle_x, \
                                angle_w, coef_y, info)


cdef void square_backward_y(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    # Initialize iterable
    cdef int jj
    cdef double coef_y, const
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Determine flux edges
    if info.edges:
        flux[:,info.cells_y] = known_x[:]
    # Iterate over Y spatial cells
    for jj in range(info.cells_y-1, -1, -1):
        coef_y = -const * angle_y / delta_y[jj]
        if angle_x > 0.0:
            known_x[jj] = square_forward_x(flux[:,jj], flux_old[:,jj], \
                                xs_total, xs_scatter, off_scatter[:,jj], \
                                external[jj::info.cells_y], known_x[jj], \
                                known_y, medium_map[:,jj], delta_x, angle_x, \
                                angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = square_backward_x(flux[:,jj], flux_old[:,jj], \
                                xs_total, xs_scatter, off_scatter[:,jj], \
                                external[jj::info.cells_y], known_x[jj], \
                                known_y, medium_map[:,jj], delta_x, angle_x, \
                                angle_w, coef_y, info)


cdef double square_forward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double coef_y, params info):
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Determine flux edge
    if info.edges:
        flux[0] += angle_w * edge_x
    # Iterate over X spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        coef_x = (const * angle_x / delta_x[ii])
        center = (coef_x * edge_x + coef_y * edge_y[ii] + xs_scatter[mat] \
                    * flux_old[ii] + external[ii] + off_scatter[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)
        # Update flux with cell edges
        if info.edges:
            flux[ii+1] += angle_w * (2 * center - edge_x)
        # Update flux with cell centers
        else:
            flux[ii] += angle_w * center
        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]
    return edge_x


cdef double square_backward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double coef_y, params info):
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Determine flux edge
    if info.edges:
        flux[info.cells_x] = angle_w * edge_x
    # Iterate over X spatial cells
    for ii in range(info.cells_x-1, -1, -1):
        mat = medium_map[ii]
        coef_x = (-const * angle_x / delta_x[ii])
        center = (coef_x * edge_x + coef_y * edge_y[ii] + xs_scatter[mat] \
                    * flux_old[ii] + external[ii] + off_scatter[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)
        # Update flux with cell edges
        if info.edges:
            flux[ii+1] += angle_w * (2 * center - edge_x)
        # Update flux with cell centers
        else:
            flux[ii] += angle_w * center
        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]
    return edge_x
