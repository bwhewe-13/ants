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

@numba.jit(nopython=True, cache=True)
def spatial_sweep(scalar_flux_old, angular_flux_last, medium_map, \
                  xs_total, xs_scatter, external_source, ps_locs, \
                  point_source, spatial_coef, temporal_coef, \
                  first_edge=0.0, spatial="diamond", temporal="BE"):
    if spatial_coef > 0:
        sweep = range(len(medium_map))
    else:
        sweep = range(len(medium_map)-1, -1, -1)
        if len(medium_map) in ps_locs:
            first_edge += point_source[np.argwhere(ps_locs == len(medium_map))[0,0]]
    if temporal == "BE":
        temporal_dd = 0.5 * temporal_coef
    elif temporal == "BDF2":
        temporal_dd = 0.75 * temporal_coef
    angular_flux = np.zeros((len(medium_map)), dtype="float64")
    for cell in sweep:
        material = medium_map[cell]
        if cell in ps_locs:
            first_edge += point_source[np.argwhere(ps_locs == cell)[0,0]]
        second_edge = (xs_scatter[material] * scalar_flux_old[cell] \
            + external_source[cell] + angular_flux_last[cell] * temporal_coef \
            + first_edge * (abs(spatial_coef) - 0.5 * xs_total[material] \
            - temporal_dd)) * 1/(abs(spatial_coef) \
            + 0.5 * xs_total[material] + temporal_dd)
        if spatial == "diamond":
            angular_flux[cell] = 0.5 * (first_edge + second_edge)
        elif spatial == "step":
            angular_flux[cell] = second_edge
        first_edge = second_edge
    return angular_flux, first_edge

@numba.jit(nopython=True, cache=True)
def discrete_ordinates(scalar_flux_old, angular_flux_last, medium_map, \
         xs_total, xs_scatter, external_source, ps_locs, point_source, \
         spatial_coef, angle_weight, temporal_coef, spatial="diamond", \
         temporal="BE"):
    angular_flux = np.zeros((angular_flux_last.shape))
    converged = 0
    count = 1
    while not (converged):
        scalar_flux = np.zeros((len(medium_map)),dtype="float64")
        angular_flux *= 0
        for angle in range(len(spatial_coef)):
            angular_flux[:,angle], edge = spatial_sweep(scalar_flux_old, \
                        angular_flux_last[:,angle], medium_map, xs_total, \
                        xs_scatter, external_source[:,angle], ps_locs, \
                        point_source[:,angle], spatial_coef[angle], \
                        temporal_coef, spatial=spatial, temporal=temporal)
            scalar_flux += angle_weight[angle] * angular_flux[:,angle]
            if len(np.unique(np.sign(spatial_coef))) == 1:
                reflect = 2 * len(spatial_coef) - angle - 1
                angular_flux[:,reflect], _ = spatial_sweep(scalar_flux_old, \
                        angular_flux_last[:,angle], medium_map, xs_total, \
                        xs_scatter, external_source[:,angle], ps_locs, \
                        point_source[:,angle], spatial_coef[angle], \
                        temporal_coef, first_edge=edge, spatial=spatial, \
                        temporal=temporal)
                scalar_flux += angle_weight[angle] * angular_flux[:,reflect]
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS) 
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular_flux

# spatial_coef: mu / cell_width_x
# temporal_coef: 1 / (velocity * delta t)