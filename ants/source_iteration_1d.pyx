########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Source iteration for one-dimensional multigroup neutron transport 
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

from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d
from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE, INNER_TOLERANCE, PI

from libc.math cimport pow, fabs #, sqrt
# from cython.view cimport array as cvarray
# import numpy as np
# from cython.parallel import prange

cdef double[:,:,:] multigroup_angular(double[:,:,:]& flux_guess, \
                        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
                        double[:]& source, double [:]& boundary, \
                        int[:]& medium_map, double[:]& delta_x, \
                        double[:]& angle_x, double[:]& angle_w, \
                        params1d params):
    # Initialize components
    cdef size_t qq1, qq2, bc1, bc2, group
    # Set indexing
    qq2 = 1 if params.qdim == 1 else params.groups
    bc2 = 1 if params.bcdim == 0 else params.groups
    # Initialize flux
    flux = tools.array_3d_ing(params)
    flux_old = flux_guess.copy()
    flux_1g = tools.array_2d_in(params)
    # Create off-scattering term
    off_scatter = tools.array_1d_i(params)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:,:] = 0.0
        for group in range(params.groups):
            qq1 = 0 if params.qdim == 1 else group
            bc1 = 0 if params.bcdim == 0 else group
            flux_1g[:,:] = flux_old[:,:,group]
            tools.off_scatter_angular(flux, flux_old, medium_map, \
                        xs_scatter, off_scatter, angle_w, params, group)
            ordinates_angular(flux[:,:,group], flux_1g, xs_total[:,group], \
                        xs_scatter[:,group,group], off_scatter, \
                        source[qq1::qq2], boundary[bc1::bc2], \
                        medium_map, delta_x, angle_x, angle_w, params)
        change = tools.group_convergence_angular(flux, flux_old, angle_w, params)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]
    return flux[:,:,:]

cdef void ordinates_angular(double[:,:] flux, double[:,:] flux_old, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] angle_x, double[:] angle_w, \
            params1d params):
    if params.geometry == 1: # slab
        slab_ordinates_angular(flux, flux_old, xs_total, xs_scatter, off_scatter, \
            source, boundary, medium_map, delta_x, angle_x, angle_w, params)

cdef void slab_ordinates_angular(double[:,:] flux, double[:,:] flux_old, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] angle_x, double[:] angle_w, \
            params1d params):
    # Initialize indices etc
    cdef size_t angle, qq1, qq2, bc1, bc2
    scalar_flux = tools.array_1d_i(params)
    # Set indexing
    qq2 = 1 if params.qdim != 3 else params.angles
    bc2 = 1 if params.bcdim != 2 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:] = 0.0
        tools.angle_angular_to_scalar(flux_old, scalar_flux, angle_w, params)
        for angle in range(params.angles):
            qq1 = 0 if params.qdim != 3 else angle
            bc1 = 0 if params.bcdim != 2 else angle
            slab_sweep(flux[:,angle], scalar_flux, xs_total, xs_scatter, \
                off_scatter, source[qq1::qq2], boundary[bc1::bc2], medium_map, \
                delta_x, angle_x[angle], 1.0, params)
        change = tools.angle_convergence_angular(flux, flux_old, angle_w, params)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]

cdef double[:,:] multigroup_scalar(double[:,:]& flux_guess, double[:,:]& xs_total, \
                        double[:,:,:]& xs_scatter, double[:]& source, \
                        double [:]& boundary, int[:]& medium_map, \
                        double[:]& delta_x, double[:]& angle_x, \
                        double[:]& angle_w, params1d params):
    # Initialize components
    cdef size_t qq1, qq2, bc1, bc2, group
    # Set indexing
    qq2 = 1 if params.qdim == 1 else params.groups
    bc2 = 1 if params.bcdim == 0 else params.groups
    # Initialize flux
    flux = tools.array_2d_ig(params)
    flux_old = flux_guess.copy()
    # flux_old = tools.array_2d_ig(params)
    flux_1g = tools.array_1d_i(params)
    # Create off-scattering term
    off_scatter = tools.array_1d_i(params)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:,:] = 0.0
        for group in range(params.groups):
            qq1 = 0 if params.qdim == 1 else group
            bc1 = 0 if params.bcdim < 0 else group
            flux_1g[:] = flux_old[:,group]
            tools.off_scatter_scalar(flux, flux_old, medium_map, \
                        xs_scatter, off_scatter, params, group)
            ordinates_scalar(flux[:,group], flux_1g, xs_total[:,group], \
                xs_scatter[:,group,group], off_scatter, source[qq1::qq2], \
                boundary[bc1::bc2], medium_map, delta_x, angle_x, angle_w, params)
        change = tools.group_convergence_scalar(flux, flux_old, params)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:,:] = flux[:,:]
    return flux[:,:]

cdef void ordinates_scalar(double[:] flux, double[:] flux_old, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] angle_x, double[:] angle_w, \
            params1d params):
    if params.geometry == 1: # slab
        slab_ordinates_scalar(flux, flux_old, xs_total, xs_scatter, off_scatter, \
            source, boundary, medium_map, delta_x, angle_x, angle_w, params)
    elif params.geometry == 2: # sphere
        sphere_ordinates_scalar(flux, flux_old, xs_total, xs_scatter, \
            off_scatter, source, boundary, medium_map, delta_x, angle_x, \
            angle_w, params)

cdef void slab_ordinates_scalar(double[:] flux, double[:] flux_old, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] angle_x, double[:] angle_w, \
            params1d params):
    # Initialize indices etc
    cdef size_t angle, qq1, qq2, bc1, bc2
    qq2 = 1 if params.qdim != 3 else params.angles
    bc2 = 1 if params.bcdim != 2 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:] = 0.0
        for angle in range(params.angles):
            qq1 = 0 if params.qdim != 3 else angle
            bc1 = 0 if params.bcdim != 2 else angle
            slab_sweep(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                       source[qq1::qq2], boundary[bc1::bc2], medium_map, \
                       delta_x, angle_x[angle], angle_w[angle], params)
        change = tools.angle_convergence_scalar(flux, flux_old, params)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]

cdef void slab_sweep(double[:]& flux, double[:]& flux_old, \
                    double[:]& xs_total, double[:]& xs_scatter, \
                    double[:]& off_scatter, double[:]& source, \
                    double[:]& boundary, int[:]& medium_map, \
                    double[:]& delta_x, double angle_x, \
                    double angle_w, params1d params):
    cdef double edge = 0.0
    if params.bc[0] == 1: # x = 0 is reflected
        edge = slab_backward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[1], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)
        edge = slab_forward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[0], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)
    elif params.bc[1] == 1: # x = X is reflected
        edge = slab_forward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[0], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)
        edge = slab_backward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[1], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)        
    elif angle_x > 0.0:
        edge = slab_forward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[0], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)
    elif angle_x < 0.0:
        edge = slab_backward_x(flux, flux_old, xs_total, xs_scatter, \
                    off_scatter, source, boundary[1], medium_map, delta_x, \
                    angle_x, angle_w, edge, params)

cdef double slab_forward_x(double[:]& flux, double[:]& flux_old, \
                    double[:]& xs_total, double[:]& xs_scatter, \
                    double[:]& off_scatter, double[:]& source, double boundary, \
                    int[:]& medium_map, double[:]& delta_x, double angle_x, \
                    double angle_w, double edge1, params1d params):
    cdef size_t cell, mat
    cdef double edge2 = 0.0
    cdef float const1 = 0 if params.spatial == 1 else -0.5
    cdef float const2 = 1 if params.spatial == 1 else 0.5
    edge1 += boundary
    for cell in range(params.cells):
        mat = medium_map[cell]
        edge2 = (xs_scatter[mat] * flux_old[cell] + source[cell] \
                + off_scatter[cell] + edge1 * (fabs(angle_x) / delta_x[cell] \
                + const1 * xs_total[mat])) * 1 / (fabs(angle_x) / delta_x[cell] \
                + const2 * xs_total[mat])
        if params.spatial == 1:
            flux[cell] += angle_w * edge2
        elif params.spatial == 2:
            flux[cell] += 0.5 * angle_w * (edge1 + edge2) 
        edge1 = edge2
    return edge1

cdef double slab_backward_x(double[:]& flux, double[:]& flux_old, \
                    double[:]& xs_total, double[:]& xs_scatter, \
                    double[:]& off_scatter, double[:]& source, double boundary, \
                    int[:]& medium_map, double[:]& delta_x, double angle_x, \
                    double angle_w, double edge1, params1d params):
    cdef size_t cell, mat
    cdef double edge2 = 0.0
    cdef float const1 = 0 if params.spatial == 1 else -0.5
    cdef float const2 = 1 if params.spatial == 1 else 0.5
    edge1 += boundary
    for cell in range(params.cells-1, -1, -1):
        mat = medium_map[cell]
        edge2 = (xs_scatter[mat] * flux_old[cell] + source[cell] \
                + off_scatter[cell] + edge1 * (fabs(angle_x) / delta_x[cell] \
                + const1 * xs_total[mat])) * 1 / (fabs(angle_x) / delta_x[cell] \
                + const2 * xs_total[mat])
        if params.spatial == 1:
            flux[cell] += angle_w * edge2
        elif params.spatial == 2:
            flux[cell] += 0.5 * angle_w * (edge1 + edge2) 
        edge1 = edge2
    return edge1

cdef void sphere_ordinates_scalar(double[:] flux, double[:] flux_old, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] angle_x, double[:] angle_w, \
            params1d params):
    # Initialize indices etc
    cdef size_t angle, qq1, qq2, bc1, bc2
    qq2 = 1 if params.qdim != 3 else params.angles
    bc2 = 1 if params.bcdim != 2 else params.angles
    # Initialize sphere specific terms
    cdef double angle_minus, angle_plus, tau
    cdef double alpha_minus, alpha_plus
    half_angle = tools.array_1d_i(params)
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        angle_minus = -1.0
        alpha_minus = 0.0
        flux[:] = 0.0
        initialize_half_angle(flux_old, half_angle, xs_total, xs_scatter, \
                    off_scatter, source, medium_map, delta_x, 0.0, params)
        for angle in range(params.angles):
            qq1 = 0 if params.qdim != 3 else angle
            bc1 = 0 if params.bcdim != 2 else angle
            angle_plus = angle_minus + 2 * angle_w[angle]
            tau = (angle_x[angle] - angle_minus) / (angle_plus - angle_minus)
            alpha_plus = angle_coef_corrector(alpha_minus, angle_x[angle], \
                                            angle_w[angle], angle, params)
            sphere_sweep(flux, flux_old, half_angle, xs_total, xs_scatter, \
                off_scatter, source[qq1::qq2], boundary[bc1::bc2], \
                medium_map, delta_x, angle_x[angle], angle_w[angle], \
                tau, alpha_plus, alpha_minus, params)
            alpha_minus = alpha_plus
            angle_minus = angle_plus            
        change = tools.angle_convergence_scalar(flux, flux_old, params)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]

cdef double angle_coef_corrector(double alpha_minus, double angle_x, \
                        double angle_w, size_t angle, params1d params):
    # For calculating alpha_plus
    if angle != params.angles - 1:
        return alpha_minus - angle_x * angle_w
    return 0.0

cdef void initialize_half_angle(double[:]& flux, double[:]& half_angle, \
            double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
            double[:]& source, int[:]& medium_map, double[:]& delta_x, \
            double angle_plus, params1d params):
    cdef size_t cell, mat
    for cell in range(params.cells-1, -1, -1):
        mat = medium_map[cell]
        half_angle[cell] = (2 * angle_plus + delta_x[cell] * (source[cell] \
                + off_scatter[cell] + xs_scatter[mat] * flux[cell])) \
                / (2 + xs_total[mat] * delta_x[cell])
        angle_plus = 2 * half_angle[cell] - angle_plus

cdef void sphere_sweep(double[:]& flux, double[:]& flux_old, \
                double[:]& half_angle, double[:]& xs_total, \
                double[:]& xs_scatter, double[:]& off_scatter, \
                double[:]& source, double[:]& boundary, \
                int[:]& medium_map, double[:]& delta_x, double angle_x, \
                double angle_w, double tau, double alpha_plus, \
                double alpha_minus, params1d params):
    # cdef double edge = 0.0
    if angle_x > 0:
        sphere_forward_x(flux, flux_old, half_angle, xs_total, xs_scatter, \
            off_scatter, source, boundary[0], medium_map, delta_x, angle_x, \
            angle_w, tau, alpha_plus, alpha_minus, params)
    elif angle_x < 0:
        sphere_backward_x(flux, flux_old, half_angle, xs_total, xs_scatter, \
            off_scatter, source, boundary[1], medium_map, delta_x, angle_x, \
            angle_w, tau, alpha_plus, alpha_minus, params)

cdef void sphere_forward_x(double[:]& flux, double[:]& flux_old, \
                double[:]& half_angle, double[:]& xs_total, \
                double[:]& xs_scatter, double[:]& off_scatter, \
                double[:]& source, double boundary, int[:]& medium_map, \
                double[:]& delta_x, double angle_x, double angle_w, \
                double tau, double alpha_plus, double alpha_minus, \
                params1d params):
    cdef size_t cell, mat
    cdef double half_cell = half_angle[0]
    cdef double area_plus, area_minus, center, volume
    for cell in range(params.cells):
        mat = medium_map[cell]
        area_plus = edge_surface_area((cell + 1) * delta_x[cell])
        area_minus = edge_surface_area(cell * delta_x[cell])
        volume = cell_volume((cell + 1) * delta_x[cell], cell * delta_x[cell])
        center = (angle_x * (area_plus + area_minus) * half_cell \
            + 1 / angle_w * (area_plus - area_minus) * (alpha_plus \
            + alpha_minus) * (half_angle[cell]) + volume * (source[cell] \
            + off_scatter[cell] + flux_old[cell] * xs_scatter[mat])) \
            / (2 * angle_x * area_plus + 2 / angle_w * (area_plus \
            - area_minus) * alpha_plus + xs_total[mat] * volume)
        flux[cell] += angle_w * center
        if params.spatial == 1:
            half_cell = center
        elif params.spatial == 2:
            half_cell = 2 * center - half_cell
        if cell != 0:
            half_angle[cell] = 1 / tau * (center - (1 - tau) * half_angle[cell])

cdef void sphere_backward_x(double[:]& flux, double[:]& flux_old, \
                double[:]& half_angle, double[:]& xs_total, \
                double[:]& xs_scatter, double[:]& off_scatter, \
                double[:]& source, double boundary, int[:]& medium_map, \
                double[:]& delta_x, double angle_x, double angle_w, \
                double tau, double alpha_plus, double alpha_minus, \
                params1d params):
    cdef size_t cell, mat
    cdef double half_cell = boundary
    cdef double area_plus, area_minus, center, volume
    for cell in range(params.cells-1, -1, -1):
        mat = medium_map[cell]
        area_plus = edge_surface_area((cell + 1) * delta_x[cell])
        area_minus = edge_surface_area(cell * delta_x[cell])
        volume = cell_volume((cell + 1) * delta_x[cell], cell * delta_x[cell])
        center = (-angle_x * (area_plus + area_minus) * half_cell \
            + 1 / angle_w * (area_plus - area_minus) * (alpha_plus \
            + alpha_minus) * (half_angle[cell]) + volume * (source[cell] \
            + off_scatter[cell] + flux_old[cell] * xs_scatter[mat])) \
            / (-2 * angle_x * area_minus + 2 / angle_w * (area_plus \
            - area_minus) * alpha_plus + xs_total[mat] * volume)
        flux[cell] += angle_w * center
        if params.spatial == 1:
            half_cell = center
        elif params.spatial == 2:
            half_cell = 2 * center - half_cell
        if cell != 0:
            half_angle[cell] = 1 / tau * (center - (1 - tau) * half_angle[cell])

cdef double edge_surface_area(double rho):
    return 4 * PI * pow(rho, 2)

cdef double cell_volume(double rho_plus, double rho_minus):
    return 4 * PI / 3 * (pow(rho_plus, 3) - pow(rho_minus, 3))
