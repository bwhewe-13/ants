########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.constants import MAX_ITERATIONS, INNER_TOLERANCE
from ants cimport cyutils

from libcpp cimport float
from libc.math cimport sqrt, pow
from cython.view cimport array as cvarray
import numpy as np

cdef void bottom_to_top(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                        double[:]& cell_bottom_left, double[:]& cell_bottom_right, \
                        int[:]& medium_map, double[:]& xs_total, \
                        double[:]& xs_matrix, double[:]& off_scatter, \
                        double[:]& external_source, double boundary, \
                        double mu, double eta, double angle_weight, \
                        int[:]& params, double[:]& delta_x, \
                        double[:]& delta_y, size_t gg_idx, size_t nn_idx):
    cdef int cells_y = delta_y.shape[0]
    cdef double edge = 0.0
    cdef double spatial_y
    cdef int cell_y
    # The params will be changed
    cdef int ex1 = gg_idx + nn_idx * params[4]
    cdef int ex2 = params[5] * params[4]
    for cell_y in range(cells_y):
        spatial_y = 2 * eta / delta_y[cell_y]
        edge = left_to_right(scalar_flux[cell_y::cells_y], \
                scalar_flux_old[cell_y::cells_y], cell_bottom_left, \
                medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                off_scatter[cell_y::cells_y], external_source[ex1::ex2][cell_y::cells_y], \
                params, boundary, mu, spatial_y, angle_weight, delta_x, \
                cell_y, 0.0)
        # X BC == Vacuum
        if params[2] == 0:
            edge = 0.0
        edge = right_to_left(scalar_flux[cell_y::cells_y], \
                scalar_flux_old[cell_y::cells_y], cell_bottom_right, \
                medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                off_scatter[cell_y::cells_y], external_source[ex1::ex2][cell_y::cells_y], \
                params, boundary, mu, spatial_y, angle_weight, delta_x, \
                cell_y, edge)


cdef void top_to_bottom(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                        double[:]& cell_bottom_left, double[:]& cell_bottom_right, \
                        int[:]& medium_map, double[:]& xs_total, \
                        double[:]& xs_matrix, double[:]& off_scatter, \
                        double[:]& external_source, double boundary, \
                        double mu, double eta, double angle_weight, \
                        int[:]& params, double[:]& delta_x, \
                        double[:]& delta_y, size_t gg_idx, size_t nn_idx):
    cdef int cells_y = delta_y.shape[0]
    cdef double edge = 0.0
    cdef double spatial_y
    cdef int cell_y
    cdef int ex1 = gg_idx + nn_idx * params[4]
    cdef int ex2 = params[5] * params[4]
    for cell_y in range(cells_y-1, -1, -1):
        spatial_y = 2 * eta / delta_y[cell_y]
        edge = left_to_right(scalar_flux[cell_y::cells_y], 
                scalar_flux_old[cell_y::cells_y], cell_bottom_left, \
                medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                off_scatter[cell_y::cells_y], external_source[ex1::ex2][cell_y::cells_y], \
                params, boundary, mu, spatial_y, angle_weight, delta_x, \
                cell_y, 0.0)
        # X BC == Vacuum
        if params[2] == 0:
            edge = 0.0
        edge = right_to_left(scalar_flux[cell_y::cells_y], \
                scalar_flux_old[cell_y::cells_y], cell_bottom_right, \
                medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                off_scatter[cell_y::cells_y], external_source[ex1::ex2][cell_y::cells_y], \
                params, boundary, mu, spatial_y, angle_weight, delta_x, \
                cell_y, edge)

cdef double left_to_right(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                        double[:]& cell_bottom, int[:]& medium_map, \
                        double[:]& xs_total, double[:]& xs_matrix, \
                        double[:]& off_scatter, double[:]& external_source, \
                        int[:]& params, double boundary, double mu, \
                        double spatial_y, double angle_weight, \
                        double[:]& delta_x, int cell_y, double edge_one):
    # 0 --> I (positive mu)
    cdef double cell_center = 0
    cdef int cells_x = delta_x.shape[0]
    cdef int mat, cell
    for cell in range(cells_x):
        mat = medium_map[cell]
        cell_center = ((2 * mu / delta_x[cell]) * edge_one + spatial_y \
                * cell_bottom[cell] + xs_matrix[mat] * scalar_flux_old[cell] \
                + external_source[cell] + off_scatter[cell]) \
                / (xs_total[mat] + (2 * mu / delta_x[cell]) + spatial_y)
        scalar_flux[cell] += angle_weight * cell_center
        edge_one = 2 * cell_center - edge_one
        cell_bottom[cell] = 2 * cell_center - cell_bottom[cell]
    return edge_one

cdef double right_to_left(double[:]& scalar_flux, double[:]& scalar_flux_old, \
                        double[:]& cell_bottom, int[:]& medium_map, \
                        double[:]& xs_total, double[:]& xs_matrix, \
                        double[:]& off_scatter, double[:]& external_source, \
                        int[:]& params, double boundary, double mu, \
                        double spatial_y, double angle_weight, \
                        double[:]& delta_x, int cell_y, double edge_two):
    # I --> 0 (negative mu)
    cdef double cell_center = 0.0
    cdef int cells_x = delta_x.shape[0]
    cdef int mat, cell
    for cell in range(cells_x-1, -1, -1):
        mat = medium_map[cell]
        cell_center = ((2 * mu / delta_x[cell]) * edge_two + spatial_y \
                * cell_bottom[cell] + xs_matrix[mat] * scalar_flux_old[cell] \
                + external_source[cell] + off_scatter[cell]) \
                / (xs_total[mat] + (2 * mu / delta_x[cell]) + spatial_y)
        scalar_flux[cell] += angle_weight * cell_center
        edge_two = 2 * cell_center - edge_two
        # Updating the j-1/2 --> j+1/2
        cell_bottom[cell] = 2 * cell_center - cell_bottom[cell]
    return edge_two


cdef double[:] scalar_quad_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
                                double[:]& xs_total, double[:]& xs_matrix, \
                                double[:]& off_scatter, double[:]& external_source, \
                                double[:]& boundary, double[:]& mu, \
                                double[:]& eta, double[:]& angle_weight, \
                                int[:]& params, double[:]& delta_x, \
                                double[:]& delta_y, size_t gg_idx): 

    cdef int cells = medium_map.shape[0]
    cdef int angles = angle_weight.shape[0]
    cdef int cell, angle_x, angle_y
    cdef size_t nn_idx

    arr1d_1 = cvarray((cells,), itemsize=sizeof(double), format="d")
    cdef double[:] scalar_flux = arr1d_1
    scalar_flux[:] = 0

    cdef int cells_x = delta_x.shape[0]
    arr1d_2 = cvarray((cells_x,), itemsize=sizeof(double), format="d")
    cdef double[:] cell_bottom_left = arr1d_2
    cell_bottom_left[:] = 0.0

    arr1d_3 = cvarray((cells_x,), itemsize=sizeof(double), format="d")
    cdef double[:] cell_bottom_right = arr1d_3
    cell_bottom_right[:] = 0.0 

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        scalar_flux[:] = 0
        for angle_y in range(angles):
            for angle_x in range(angles):
                cell_bottom_left[:] = 0.0
                cell_bottom_right[:] = 0.0
                # Might remove this line
                nn_idx = 0 if params[5] == 1 else angle_x
                bottom_to_top(scalar_flux, scalar_flux_old, cell_bottom_left, \
                        cell_bottom_right, medium_map, xs_total, xs_matrix, \
                        off_scatter, external_source, boundary[angle_x], \
                        mu[angle_x], eta[angle_y], 0.25*angle_weight[angle_x], \
                        params, delta_x, delta_y, gg_idx, nn_idx)
                # Y BC == Vacuum
                if params[3] == 0:
                    cell_bottom_left[:] = 0.0
                    cell_bottom_right[:] = 0.0
                top_to_bottom(scalar_flux, scalar_flux_old, cell_bottom_left, \
                        cell_bottom_right, medium_map, xs_total, xs_matrix, \
                        off_scatter, external_source, boundary[angle_x], \
                        mu[angle_x], eta[angle_y], 0.25*angle_weight[angle_x], \
                        params, delta_x, delta_y, gg_idx, nn_idx)
        change = cyutils.group_scalar_convergence(scalar_flux, scalar_flux_old)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        scalar_flux_old[:] = scalar_flux[:]
    return scalar_flux[:]


cdef double[:] scalar_single_sweep(double[:] scalar_flux_old, int[:]& medium_map, \
                                double[:]& xs_total, double[:]& xs_matrix, \
                                double[:]& off_scatter, double[:]& external_source, \
                                double[:]& boundary, double[:]& mu, \
                                double[:]& eta, double[:]& angle_weight, \
                                int[:]& params, double[:]& delta_x, \
                                double[:]& delta_y, size_t gg_idx): 
    
    cdef int angles = angle_weight.shape[0]
    cdef int cells_x = delta_x.shape[0]
    cdef int cells_y = delta_y.shape[0]
    cdef int cell_x, cell_y, angle
    cdef size_t nn_idx, ex1
    cdef size_t ex2 = params[5] * params[4]
    cdef double spatial_y

    arr1d_1 = cvarray((cells_x * cells_y,), itemsize=sizeof(double), format="d")
    cdef double[:] scalar_flux = arr1d_1
    scalar_flux[:] = 0

    arr1d_2 = cvarray((cells_y,), itemsize=sizeof(double), format="d")
    cdef double[:] boundary_x = arr1d_2
    boundary_x[:] = 0.0

    arr1d_3 = cvarray((cells_x,), itemsize=sizeof(double), format="d")
    cdef double[:] boundary_y = arr1d_3
    boundary_y[:] = 0.0 

    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        scalar_flux[:] = 0.0
        for angle in range(angles):
            boundary_y[:] = 0.0
            # Might remove this line
            nn_idx = 0 if params[5] == 1 else angle
            ex1 = gg_idx + nn_idx * params[4]
            if mu[angle] > 0 and eta[angle] > 0:
                # 0 --> I, 0 --> J
                for cell_y in range(cells_y):
                    spatial_y = 2 * eta[angle] / delta_y[cell_y]
                    boundary_x[cell_y] = left_to_right(scalar_flux[cell_y::cells_y], \
                            scalar_flux_old[cell_y::cells_y], boundary_y, \
                            medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                            off_scatter[cell_y::cells_y], \
                            external_source[ex1::ex2][cell_y::cells_y], \
                            params, boundary[angle], mu[angle], spatial_y, \
                            0.25*angle_weight[angle], delta_x, cell_y, 0.0)
                    # print(cell_y, np.asarray(boundary_y))
            elif mu[angle] < 0 and eta[angle] > 0:
                # I --> 0, 0 --> J
                for cell_y in range(cells_y):
                    spatial_y = 2 * eta[angle] / delta_y[cell_y]
                    boundary_x[cell_y] = right_to_left(scalar_flux[cell_y::cells_y], \
                            scalar_flux_old[cell_y::cells_y], boundary_y, \
                            medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                            off_scatter[cell_y::cells_y], \
                            external_source[ex1::ex2][cell_y::cells_y], \
                            params, boundary[angle], abs(mu[angle]), spatial_y, \
                            0.25*angle_weight[angle], delta_x, cell_y, 0.0)
            elif mu[angle] > 0 and eta[angle] < 0:
                # 0 --> I, J --> 0
                for cell_y in range(cells_y-1, -1, -1):
                    spatial_y = abs(2 * eta[angle]) / delta_y[cell_y]
                    boundary_x[cell_y] = left_to_right(scalar_flux[cell_y::cells_y], \
                            scalar_flux_old[cell_y::cells_y], boundary_y, \
                            medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                            off_scatter[cell_y::cells_y], \
                            external_source[ex1::ex2][cell_y::cells_y], \
                            params, boundary[angle], mu[angle], spatial_y, \
                            0.25*angle_weight[angle], delta_x, cell_y, 0.0)
            elif mu[angle] < 0 and eta[angle] < 0:
                # I --> 0, J --> 0
                for cell_y in range(cells_y-1, -1, -1):
                    spatial_y = abs(2 * eta[angle]) / delta_y[cell_y]
                    boundary_x[cell_y] = right_to_left(scalar_flux[cell_y::cells_y], \
                            scalar_flux_old[cell_y::cells_y], boundary_y, \
                            medium_map[cell_y::cells_y], xs_total, xs_matrix, \
                            off_scatter[cell_y::cells_y], \
                            external_source[ex1::ex2][cell_y::cells_y], \
                            params, boundary[angle], abs(mu[angle]), spatial_y, \
                            0.25*angle_weight[angle], delta_x, cell_y, 0.0)
        change = cyutils.group_scalar_convergence(scalar_flux, scalar_flux_old)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        # print("Source Iteration {}\n{}\nChange {} Sum {}".format(count, \
        #         "="*35, change, np.sum(scalar_flux)))
        count += 1
        scalar_flux_old[:] = scalar_flux[:]
    return scalar_flux[:]


# if params[2] == 1 and params[3] == 1: # reflecting x and y
#     for cell_y in range(cells_y-1, -1, -1):
#         spatial_y = 2 * eta[angle] / delta_y[cell_y]
#         boundary_x[cell_y] = left_to_right(scalar_flux[cell_y::cells_y], \
#                 scalar_flux_old[cell_y::cells_y], boundary_y, \
#                 medium_map[cell_y::cells_y], xs_total, xs_matrix, \
#                 off_scatter[cell_y::cells_y], \
#                 external_source[ex1::ex2][cell_y::cells_y], \
#                 params, boundary[angle], mu[angle], spatial_y, \
#                 0.25*angle_weight[angle], delta_x, cell_y, boundary_x[cell_y])
# elif params[2] == 1: # reflecting x boundary, not reflecting y
#     boundary_y[:] = 0
#     for cell_y in range(cells_y-1, -1, -1):
#         spatial_y = 2 * eta[angle] / delta_y[cell_y]
#         boundary_x[cell_y] = left_to_right(scalar_flux[cell_y::cells_y], \
#                 scalar_flux_old[cell_y::cells_y], boundary_y, \
#                 medium_map[cell_y::cells_y], xs_total, xs_matrix, \
#                 off_scatter[cell_y::cells_y], \
#                 external_source[ex1::ex2][cell_y::cells_y], \
#                 params, boundary[angle], mu[angle], spatial_y, \
#                 0.25*angle_weight[angle], delta_x, cell_y, boundary_x[cell_y])
# elif params[3] == 1: # reflecting y boundary, not reflecting x
#     for cell_y in range(cells_y-1, -1, -1):
#         spatial_y = 2 * eta[angle] / delta_y[cell_y]
#         boundary_x[cell_y] = left_to_right(scalar_flux[cell_y::cells_y], \
#                 scalar_flux_old[cell_y::cells_y], boundary_y, \
#                 medium_map[cell_y::cells_y], xs_total, xs_matrix, \
#                 off_scatter[cell_y::cells_y], \
#                 external_source[ex1::ex2][cell_y::cells_y], \
#                 params, boundary[angle], mu[angle], spatial_y, \
#                 0.25*angle_weight[angle], delta_x, cell_y, 0.0)