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

# distutils: language=c++
# cython: cdivision=True

from ants cimport cytools as tools
from ants.cytools cimport params1d
from ants cimport sweeps1d
from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE, INNER_TOLERANCE, PI

from libc.math cimport pow #, sqrt, pow
from cython.view cimport array as cvarray
import numpy as np
from cython.parallel import prange


cdef double[:] multigroup(double[:] flux_guess, double[:,:] xs_total, \
                double[:,:,:] xs_matrix, double[:] external_source, \
                double [:] boundary, int[:] medium_map, double[:] cell_width, \
                double[:] mu, double[:] angle_w, params1d params, bint angular):
    # Initialize components
    cdef size_t q_idx1, q_idx2, bc_idx1, bc_idx2, group
    # Set indexing
    q_idx2 = 1 if params.qdim == 1 else params.groups
    bc_idx2 = 1 if params.bcdim == 0 else params.groups
    # Initialize flux
    flux = tools.group_flux(params, angular)
    # flux_old = tools.group_flux(params, angular)
    flux_old = flux_guess.copy()
    # Create off-scattering term
    arr1d = cvarray((params.cells,), itemsize=sizeof(double), format="d")    
    cdef double[:] off_scatter = arr1d
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:] = 0.0
        for group in range(params.groups):
            q_idx1 = 0 if params.qdim == 1 else group
            bc_idx1 = 0 if params.bcdim < 0 else group
            tools.off_scatter_term(flux, flux_old, medium_map, xs_matrix, \
                            off_scatter, angle_w, params, group, angular)
            ordinates(flux[group::params.groups], flux_old[group::params.groups], \
                xs_total[:,group], xs_matrix[:,group,group], off_scatter, \
                external_source[q_idx1::q_idx2], boundary[bc_idx1::bc_idx2], \
                medium_map, cell_width, mu, angle_w, params, angular)
        change = tools.group_convergence(flux, flux_old, angle_w, params, angular)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]
    return flux[:]


cdef void ordinates(double[:] flux, double[:] flux_one_group, double[:] xs_total, \
            double[:] xs_scatter, double[:] off_scatter, double[:] source, \
            double[:] boundary, int[:] medium_map, double[:] cell_width, \
            double[:] mu, double[:] angle_w, params1d params, bint angular):
    if params.geometry == 1: # slab
        slab_ordinates(flux, flux_one_group, xs_total, xs_scatter, \
                off_scatter, source, boundary, medium_map, cell_width, \
                mu, angle_w, params, angular)
    elif params.geometry == 2: # sphere
        sphere_ordinates(flux, flux_one_group, xs_total, xs_scatter, \
                off_scatter, source, boundary, medium_map, cell_width, \
                mu, angle_w, params, angular)

cdef void slab_ordinates(double[:] flux, double[:] flux_one_group, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] cell_width, double[:] mu, double[:] angle_w, \
            params1d params, bint angular):
    # Initialize indices etc
    cdef size_t q_idx1, q_idx2, bc_idx1, bc_idx2, phi_idx1, phi_idx2
    cdef size_t ii, cell, angle, mat
    cdef float xs1_const = 0 if params.spatial == 1 else -0.5
    cdef float xs2_const = 1 if params.spatial == 1 else 0.5
    cdef double edge1, edge2, weight
    # Initialize fluxes
    scalar_old = tools.angle_flux(params, False)
    flux_old = flux_one_group.copy()
    # Set indexing
    phi_idx2 = params.angles if angular == True else 1
    q_idx2 = 1 if params.qdim != 3 else params.angles
    bc_idx2 = 1 if params.bcdim != 2 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        flux[:] = 0.0
        tools.angle_angular_to_scalar(scalar_old, flux_old, angle_w, params, angular)
        for angle in range(params.angles):
            q_idx1 = 0 if params.qdim != 3 else angle
            bc_idx1 = 0 if params.bcdim != 2 else angle
            phi_idx1 = 0 if angular == False else angle
            edge1 = find_boundary(edge2, mu[angle], boundary[bc_idx1::bc_idx2], params)
            weight = 1 if angular == True else angle_w[angle]
            ranger = range_finder(mu[angle], params)
            for ii in range(params.cells):
                cell = ranger[ii]
                mat = medium_map[cell]
                edge2 = (xs_scatter[mat] * scalar_old[cell] \
                        + source[q_idx1::q_idx2][cell] + off_scatter[cell] \
                        + edge1 * (abs(mu[angle] / cell_width[cell]) + xs1_const * xs_total[mat])) \
                        * 1/(abs(mu[angle] / cell_width[cell]) + xs2_const * xs_total[mat])
                if params.spatial == 1:
                    flux[phi_idx1::phi_idx2][cell] += weight * edge2
                elif params.spatial == 2:
                    flux[phi_idx1::phi_idx2][cell] += 0.5 * weight * (edge1 + edge2) 
                edge1 = edge2
        change = tools.angle_convergence(flux, flux_old, angle_w, params, angular)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]

cdef void sphere_ordinates(double[:] flux, double[:] flux_one_group, \
            double[:] xs_total, double[:] xs_scatter, double[:] off_scatter, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] cell_width, double[:] mu, double[:] angle_w, \
            params1d params, bint angular):
    # Initialize indices etc
    cdef size_t q_idx1, q_idx2, bc_idx1, bc_idx2, phi_idx1, phi_idx2
    cdef size_t ii, cell, angle, mat
    cdef float xs1_const = 0 if params.spatial == 1 else -0.5
    cdef float xs2_const = 1 if params.spatial == 1 else 0.5
    cdef double edge, weight
    cdef int offset = 0
    cdef double mu_minus, mu_plus, tau
    cdef double alpha_minus, alpha_plus
    cell_edges = calculate_cell_edges(cell_width, params)
    surface_area = calculate_surface_area(cell_edges, params)
    volume = calculate_volume(cell_edges, params)
    # Initialize fluxes
    scalar_old = tools.angle_flux(params, False)
    flux_old = flux_one_group.copy()
    # Set indexing
    phi_idx2 = params.angles if angular == True else 1
    q_idx2 = 1 if params.qdim != 3 else params.angles
    bc_idx2 = 1 if params.bcdim != 2 else params.angles
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        mu_minus = -1.0
        flux[:] = 0.0
        tools.angle_angular_to_scalar(scalar_old, flux_old, angle_w, params, angular)
        flux_half_angle = initalize_half_angle(scalar_old, xs_total, xs_scatter, off_scatter, \
                            source[q_idx1::q_idx2], boundary[bc_idx1::bc_idx2][1], \
                            medium_map, cell_width, params)
        # for angle in prange(params.angles, nogil=True):
        for angle in range(params.angles):
            # Indexing
            q_idx1 = 0 if params.qdim != 3 else angle
            bc_idx1 = 0 if params.bcdim != 2 else angle
            phi_idx1 = 0 if angular == False else angle
            # Additional sphere items
            weight = 1 if angular == True else angle_w[angle]
            mu_plus = mu_minus + 2 * angle_w[angle]
            tau = (mu[angle] - mu_minus) / (mu_plus - mu_minus)
            if angle == 0:
                alpha_minus = 0.0
            if angle == (params.angles - 1):
                alpha_plus = 0.0
            else:
                alpha_plus = alpha_minus - mu[angle] * angle_w[angle]
            offset = 1 if mu[angle] > 0 else 0
            edge = find_sphere_boundary(boundary[bc_idx1::bc_idx2][1], \
                                        flux_half_angle[0], mu[angle])
            ranger = range_finder(mu[angle], params)
            for ii in range(params.cells):
                cell = ranger[ii]
                mat = medium_map[cell]
                center = (abs(mu[angle]) * (surface_area[cell+1] + surface_area[cell]) * edge \
                        + 1 / angle_w[angle] * (surface_area[cell+1] - surface_area[cell]) \
                        * (alpha_plus + alpha_minus) * flux_half_angle[cell] + volume[cell] \
                        * (xs_scatter[mat] * scalar_old[cell] + source[q_idx1::q_idx2][cell] \
                        + off_scatter[cell])) \
                        / (2 * abs(mu[angle]) * surface_area[cell+offset] + 2 / angle_w[angle] \
                        * (surface_area[cell+1] - surface_area[cell]) * alpha_plus \
                        + xs_total[mat] * volume[cell])
                if params.spatial == 1:
                    flux[phi_idx1::phi_idx2][cell] += weight * center
                    print("Not for step method yet")
                elif params.spatial == 2:
                    flux[phi_idx1::phi_idx2][cell] += weight * center
                edge = 2 * center - edge
                if cell != 0:
                    flux_half_angle[cell] = 1 / tau * (center - (1 - tau) * flux_half_angle[cell])
            alpha_minus = alpha_plus
            mu_minus = mu_plus
        change = tools.angle_convergence(flux, flux_old, angle_w, params, angular)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]


cdef int[:] range_finder(double mu, params1d params):
    if mu > 0.0:
        arr1d = np.arange(params.cells, dtype=np.int32)
    elif mu < 0.0:
        arr1d = np.arange(params.cells-1, -1, -1, dtype=np.int32)
    cdef int[:] ranger = arr1d
    return ranger

cdef double find_boundary(double prev_edge, double mu, double[:] boundary, \
                            params1d params):
    if (mu > 0 and params.bc[0] == 1) or (mu < 0 and params.bc[1] == 1):
        return prev_edge
    elif mu > 0:
        return boundary[0]
    else:
        return boundary[1]


cdef double find_sphere_boundary(double edge, double flux_half_angle, double mu):
    if mu > 0.0:
        return flux_half_angle
    elif mu < 0.0:
        return edge

cdef double[:] initalize_half_angle(double[:] scalar_flux, double[:] xs_total, \
                                double[:] xs_scatter, double[:] off_scatter, \
                                double[:] source, double boundary, \
                                int[:] medium_map, double[:] cell_width, \
                                params1d params):
    cdef size_t cell, mat
    cdef edge = boundary
    cdef double[:] half_angle = cvarray((params.cells,), itemsize=sizeof(double), format="d")
    half_angle[:] = 0.0
    for cell in range(params.cells-1, -1, -1):
        mat = medium_map[cell]
        half_angle[cell] = (2 * edge + cell_width[cell] * (source[cell] \
                            + off_scatter[cell] + xs_scatter[mat] \
                            * scalar_flux[cell])) / \
                            (2 + xs_total[mat] * cell_width[cell])
        edge = 2 * half_angle[cell] - edge
    return half_angle

cdef double[:] calculate_cell_edges(double[:] cell_width, params1d params):
    cdef size_t cell
    cdef double[:] cell_edges = cvarray((params.cells+1,), \
                                    itemsize=sizeof(double), format="d")
    cell_edges[:] = 0.0
    for cell in range(params.cells):
        cell_edges[cell + 1] = cell_edges[cell] + cell_width[cell]
    return cell_edges

cdef double[:] calculate_surface_area(double[:] cell_edges, params1d params):
    cdef size_t cell
    cdef double[:] surface_area = cvarray((params.cells+1,), \
                                    itemsize=sizeof(double), format="d")
    surface_area[:] = 0.0
    for cell in range(params.cells + 1):
        surface_area[cell] = 4 * PI * pow((cell_edges[cell]), 2)
    return surface_area

cdef double[:] calculate_volume(double[:] cell_edges, params1d params):
    cdef size_t cell
    cdef double[:] volume = cvarray((params.cells,), itemsize=sizeof(double), format="d")
    volume[:] = 0.0
    for cell in range(params.cells):
        volume[cell] = 4 * PI / 3 * (pow(cell_edges[cell+1],3) - pow(cell_edges[cell], 3))
    return volume

# cdef void adjoint(double[:] flux, double[:] flux_one_group, double[:] xs_total, \
#             double[:] xs_scatter, double[:] off_scatter, double[:] source, \
#             double[:] boundary, int[:] medium_map, double[:] cell_width, \
#             double[:] mu, double[:] angle_w, params1d params, bint angular):
#     # Initialize indices etc
#     cdef size_t q_idx1, q_idx2, bc_idx1, bc_idx2, phi_idx1, phi_idx2
#     cdef size_t cell, angle, mat
#     cdef float xs1_const = 0 if params.spatial == 1 else -0.5
#     cdef float xs2_const = 1 if params.spatial == 1 else 0.5
#     cdef double known_edge[1]
#     # Initialize fluxes
#     # flux = tools.angle_flux(params, angular)
#     scalar_old = tools.angle_flux(params, False)
#     flux_old = flux_one_group.copy()
#     flux[:] = 0.0
#     # Set indexing
#     phi_idx2 = params.angles if angular == True else 1
#     q_idx2 = 1 if params.qdim != 3 else params.angles
#     bc_idx2 = 1 if params.bcdim != 2 else params.angles
#     # Set convergence limits
#     cdef bint converged = False
#     cdef size_t count = 1
#     cdef double change = 0.0
#     while not (converged):
#         flux[:] = 0.0
#         tools.angle_angular_to_scalar(scalar_old, flux_old, angle_w, params, angular)
#         for angle in range(params.angles):
#             q_idx1 = 0 if params.qdim != 3 else angle
#             bc_idx1 = 0 if params.bcdim != 2 else angle
#             phi_idx1 = 0 if angular == False else angle
#             known_edge[0] = sweeps1d.find_boundary(known_edge[0], mu[angle], \
#                                         boundary[bc_idx1::bc_idx2], params)
#             if mu[angle] < 0:
#                 for cell in range(params.cells):
#                     mat = medium_map[cell]
#                     flux[phi_idx1::phi_idx2][cell] = sweeps1d.sweep( \
#                         flux[phi_idx1::phi_idx2][cell], scalar_old[cell], \
#                         xs_total[mat], xs_scatter[mat], off_scatter[cell], \
#                         source[q_idx1::q_idx2][cell], mu[angle], \
#                         angle_w[angle], cell_width[cell], known_edge, \
#                         xs1_const, xs2_const, params, angular)
#             elif mu[angle] > 0:
#                 for cell in range(params.cells-1, -1, -1):
#                     mat = medium_map[cell]
#                     flux[phi_idx1::phi_idx2][cell] = sweeps1d.sweep( \
#                         flux[phi_idx1::phi_idx2][cell], scalar_old[cell], \
#                         xs_total[mat], xs_scatter[mat], off_scatter[cell], \
#                         source[q_idx1::q_idx2][cell], mu[angle], \
#                         angle_w[angle], cell_width[cell], known_edge, \
#                         xs1_const, xs2_const, params, angular)
#         change = tools.angle_convergence(flux, flux_old, angle_w, params, angular)
#         converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
#         count += 1
#         flux_old[:] = flux[:]