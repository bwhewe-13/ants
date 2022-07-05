########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Calculate the residual from the angular flux (nearby problems) or the
# angular flux from the scalar flux (hybrid method).
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from cython.view cimport array as cvarray
import numpy as np

def residual(double[:,:,:] angular_flux, int[:] medium_map, \
             double[:,:] xs_total, double[:,:,:] xs_scatter, \
             double[:,:,:] xs_fission, double[:] external_source, \
             double [:] point_source, double[:] mu, double[:] angle_weight, \
             int[:] params, double cell_width):
    
    cdef size_t cells = medium_map.shape[0]
    cdef size_t groups = xs_total.shape[1]
    cdef size_t angles = mu.shape[0]
    cdef size_t materials = xs_total.shape[0]
    xs_matrix = memoryview(np.zeros((materials, groups, groups)))
    combine_self_scattering(xs_matrix, xs_scatter, xs_fission)

    cdef double dpsi, edge_one
    cdef size_t gg_idx, nn_idx

    residual_flux = memoryview(np.zeros((cells, angles, groups)))
    scalar_flux = memoryview(np.zeros((cells, groups)))
    angular_to_scalar(angular_flux, scalar_flux, angle_weight)

    for group in range(groups):
        gg_idx = 0 if params[3] == 1 else group
        for angle in range(angles):
            nn_idx = 0 if params[4] == 1 else angle
            edge_one = 0
            if mu[angle] > 0:
                for cell in range(cells):
                    material = medium_map[cell]
                    if params[1] == 1:
                        dpsi = (angular_flux[cell][angle][group] - edge_one) / cell_width
                    elif params[1] == 2:
                        dpsi = 2 * (angular_flux[cell][angle][group] - edge_one) / cell_width
                    residual_flux[cell, angle, group] = (mu[angle] * dpsi \
                        + angular_flux[cell, angle, group] * xs_total[material][group]) \
                        - (xs_matrix[material, group, group] * scalar_flux[cell, group] \
                        + external_source[gg_idx+nn_idx*params[3]::params[4]*params[3]][cell])
                    if params[1] == 1:
                        edge_one = angular_flux[cell][angle][group]
                    elif params[1] == 2:
                        edge_one = 2 * angular_flux[cell][angle][group] - edge_one
            elif mu[angle] < 0:
                for cell in range(cells-1, -1, -1):
                    material = medium_map[cell]
                    if params[1] == 1:
                        dpsi = (edge_one - angular_flux[cell][angle][group]) / cell_width
                    elif params[1] == 2:
                        dpsi = 2 * (edge_one - angular_flux[cell][angle][group]) / cell_width
                    residual_flux[cell, angle, group] = (mu[angle] * dpsi \
                        + angular_flux[cell, angle, group] * xs_total[material][group]) \
                        - (xs_matrix[material, group, group] * scalar_flux[cell, group] \
                        + external_source[gg_idx+nn_idx*params[3]::params[4]*params[3]][cell])
                    if params[1] == 1:
                        edge_one = angular_flux[cell][angle][group]
                    elif params[1] == 2:
                        edge_one = 2 * angular_flux[cell][angle][group] - edge_one
    return np.asarray(residual_flux)


cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
                double[:,:,:] xs_scatter, double[:,:,:] xs_fission):
    cdef size_t materials = xs_matrix.shape[0]
    cdef size_t groups = xs_matrix.shape[1]
    for mat in range(materials):
        for ing in range(groups):
            for outg in range(groups):
                xs_matrix[mat][ing][outg] = xs_scatter[mat][ing][outg] \
                                            + xs_fission[mat][ing][outg]


cdef void angular_to_scalar(double[:,:,:]& angular_flux, \
                    double[:,:]& scalar_flux, double[:]& angle_weight):
    cdef size_t cells = angular_flux.shape[0]
    cdef size_t angles = angular_flux.shape[1]
    cdef size_t groups = angular_flux.shape[2]
    for group in range(groups):
        for angle in range(angles):
            for cell in range(cells):
                scalar_flux[cell][group] += angle_weight[angle] \
                                    * angular_flux[cell][angle][group]