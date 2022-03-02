########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Mono-energetic spatial sweeps for different spatial discretizations.
#
########################################################################

import ants.constants as constants

import numpy as np
import numba

# @numba.jit(nopython=True, cache=True)
# def diamond(scalar_flux_old, total, scatter, source, medium_map, \
#             mu_delta_x, weight, xboundary, angular=False):
#     converged = 0
#     count = 1
#     if angular is True:
#         angular_flux = np.zeros((len(medium_map), len(mu_delta_x)))
#     while not (converged):
#         scalar_flux = np.zeros((len(medium_map)),dtype='float64')
#         psi_minus = 0.0
#         # if np.all(xboudary == [0,0]):
#         for angle in range(len(mu_delta_x)):
#             if mu_delta_x[angle] > 0: 
#                 psi_minus = 0.0
#                 for cell in range(len(medium_map)):
#                     material = medium_map[cell]
#                     psi_plus = (scatter[material] \
#                         * scalar_flux_old[cell] + source[cell] + psi_minus \
#                         * (mu_delta_x[angle] - 0.5 * total[material]))\
#                         * 1/(mu_delta_x[angle] + 0.5 * total[material])
#                     scalar_flux[cell] += (0.5 * weight[angle] \
#                                           * (psi_plus + psi_minus))
#                     if angular is True:
#                         angular_flux[cell, angle] = 0.5 * (psi_plus + psi_minus)
#                     psi_minus = psi_plus
#             elif mu_delta_x[angle] < 0:
#                 psi_plus = 0.0
#                 for cell in range(len(medium_map)-1,-1,-1):
#                     material = medium_map[cell]
#                     psi_minus = (scatter[material] \
#                         * scalar_flux_old[cell] + source[cell] + psi_plus \
#                         * (abs(mu_delta_x[angle]) - 0.5 * total[material]))\
#                         * 1/(abs(mu_delta_x[angle]) + 0.5 * total[material])
#                     scalar_flux[cell] += (0.5 * weight[angle] \
#                                           * (psi_plus + psi_minus))
#                     if angular is True:
#                         angular_flux[cell, angle] = 0.5 * (psi_plus + psi_minus)
#                     psi_plus = psi_minus
#         change = np.linalg.norm((scalar_flux - scalar_flux_old) \
#                                 /scalar_flux/(len(medium_map)))
#         converged = (change < constants.INNER_TOLERANCE) \
#                     or (count >= constants.MAX_ITERATIONS) 
#         count += 1
#         scalar_flux_old = scalar_flux.copy()
#     if angular is False:
#         angular_flux = None
#     return scalar_flux, angular_flux

# @numba.jit(nopython=True) #, cache=True)
def diamond(scalar_flux_old, total, scatter, source, medium_map, \
            mu_delta_x, weight, xboundary, point_source_locs, \
            point_sources, angular=False):
    converged = 0
    count = 1
    angular = np.zeros((len(medium_map)+1, len(mu_delta_x)*2))
    while not (converged):
        scalar_flux = np.zeros((len(medium_map)),dtype='float64')
        for angle in range(len(mu_delta_x)):
            angular_flux = _diamond_sweep(mu_delta_x[angle], medium_map, \
                scalar_flux_old, total, scatter, source, point_source_locs, \
            point_sources)
            angular[:,angle] = angular_flux.copy()
            scalar_flux += 0.5 * weight[angle] \
                            * (angular_flux[:-1] + angular_flux[1:])
            if np.sum(xboundary) != 0:
                angular_flux = _diamond_sweep(-mu_delta_x[angle], \
                    medium_map, scalar_flux_old, total, scatter, source,\
                    edge_one=angular_flux[-xboundary[1]])
                angular[:,angle+len(mu_delta_x)] = angular_flux.copy()
                scalar_flux += 0.5 * weight[angle] \
                            * (angular_flux[:-1] + angular_flux[1:])
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS) 
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular

# @numba.jit(nopython=True) #, cache=True)
def _diamond_sweep(mu_delta_x, medium_map, scalar_flux_old, \
                    total, scatter, source, point_source_locs, \
                    point_source, edge_one=0.0):
    angular_edges = np.zeros((len(medium_map)+1),dtype='float64')
    angular_edges[point_source_locs] = point_source
    if mu_delta_x > 0:
        sweep = range(len(medium_map))
        angular_edges[0] = edge_one
        offset = 1
    else:
        sweep = range(len(medium_map)-1, -1, -1)
        angular_edges[-1] = edge_one
        offset = 0
    for cell in sweep:
        material = medium_map[cell]
        angular_edges[cell+offset] = (scatter[material] * scalar_flux_old[cell] \
            + source[cell] + edge_one * (abs(mu_delta_x) \
            - 0.5 * total[material])) * 1/(abs(mu_delta_x) \
            + 0.5 * total[material])
        edge_one = angular_edges[cell+offset]
    return angular_edges
