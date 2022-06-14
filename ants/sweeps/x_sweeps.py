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
def spatial_sweep(scalar_flux_old, angular_flux_last, medium_map, \
                  xs_total, xs_scatter, external_source, ps_locs, \
                  point_source, spatial_coef, temporal_coef, \
                  first_edge=0.0, spatial="diamond", temporal="BE"):
    # See if point source is np.array([0],dtype=np.float64) --> None
    if not np.any(point_source):
        point_source = np.zeros((len(medium_map)),dtype=np.float64)
    # Determine direction of sweep
    if spatial_coef > np.float(0):
        sweep = range(len(medium_map))
    else:
        sweep = range(len(medium_map)-1, -1, -1)
        if len(medium_map) in ps_locs:
            first_edge += point_source[np.argwhere(ps_locs == len(medium_map))[0,0]]
    # Temporal discretization
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

# @numba.jit(nopython=True, cache=True)
def discrete_ordinates(scalar_flux_old, angular_flux_last, medium_map, \
         xs_total, xs_scatter, external_source, ps_locs, point_source, \
         spatial_coef, angle_weight, temporal_coef, spatial="diamond", \
         temporal="BE"):
    angular_flux = np.zeros((angular_flux_last.shape), dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        scalar_flux = np.zeros((len(medium_map)),dtype=np.float64)
        angular_flux *= 0
        for angle in range(len(spatial_coef)):
            idx_ex = (..., angle)
            if external_source.ndim == 1:
                idx_ex = ()
            idx_ps = (..., angle)
            if point_source.ndim == 1:
                idx_ps = ()
            angular_flux[:,angle], edge = spatial_sweep(scalar_flux_old, \
                        angular_flux_last[:,angle], medium_map, xs_total, \
                        xs_scatter, external_source[idx_ex], ps_locs, \
                        point_source[idx_ps], spatial_coef[angle], \
                        temporal_coef, spatial=spatial, temporal=temporal)
            scalar_flux += angle_weight[angle] * angular_flux[:,angle]
            if len(np.unique(np.sign(spatial_coef))) == 1:
                reflect = 2 * len(spatial_coef) - angle - 1
                angular_flux[:,reflect], _ = spatial_sweep(scalar_flux_old, \
                        angular_flux_last[:,angle], medium_map, xs_total, \
                        xs_scatter, external_source[idx_ex], ps_locs, \
                        point_source[idx_ps], spatial_coef[angle], \
                        temporal_coef, first_edge=edge, spatial=spatial, \
                        temporal=temporal)
                scalar_flux += angle_weight[angle] * angular_flux[:,reflect]
            # print("In angle loop", np.sum(angular_flux))
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        # print(change, np.sum(scalar_flux))
        # if change in [np.inf, np.nan]:
        # if np.isnan(change) or np.isinf(change):
        #     change = 0.
        #     print(np.sum(scalar_flux))
        # print("In Count", count, "Change", change)
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS) 
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular_flux

# spatial_coef: mu / cell_width_x
# temporal_coef: 1 / (velocity * delta t)

# @numba.jit(nopython=True, cache=True)
def scalar_sweep(scalar_flux_old, medium_map, xs_total, xs_scatter, \
            source, point_source_loc, point_source, \
            spatial_coef, angle_weight, spatial=2, \
            boundary=0, angular=False):
    dummy_angle_weight = np.ones((angle_weight.shape), dtype=np.float64)
    if angular:
        scalar_flux = np.zeros((len(medium_map), len(spatial_coef)), dtype=np.float64)
        # angle_weight = (angle_weight * 0) + 1
    else:
        scalar_flux = np.zeros((scalar_flux_old.shape), dtype=np.float64)
    # print("\n\n", scalar_flux.shape, "\n\n")
    converged = 0
    count = 95
    while not (converged):
        scalar_flux *= 0
        for angle in range(len(spatial_coef)):
            idx_ex = () if source.ndim == 1 else (..., angle)
            idx_ps = () if point_source.ndim == 1 else (..., angle)
            if angular:
                temp_flux = _to_scalar(scalar_flux_old, angle_weight)
                if boundary == 0:
                    scalar_flux[:,angle] = scalar_vacuum(temp_flux, medium_map, \
                        xs_total, xs_scatter, source[idx_ex], point_source_loc[0], \
                        point_source[idx_ps][0], spatial_coef[angle], \
                        dummy_angle_weight[angle], \
                        spatial)
                elif boundary == 1:
                    scalar_flux[:,angle] = scalar_reflected(temp_flux, medium_map, \
                        xs_total, xs_scatter, source, point_source_loc[0], \
                        point_source[idx_ps][0], spatial_coef[angle], dummy_angle_weight[angle], \
                        spatial)
            else:
                if boundary == 0:
                    scalar_flux += scalar_vacuum(scalar_flux_old, medium_map, \
                        xs_total, xs_scatter, source[idx_ex], point_source_loc[0], \
                        point_source[idx_ps][0], spatial_coef[angle], \
                        angle_weight[angle], \
                        spatial)
                elif boundary == 1:
                    scalar_flux += scalar_reflected(scalar_flux_old, medium_map, \
                        xs_total, xs_scatter, source, point_source_loc[0], \
                        point_source[idx_ps][0], spatial_coef[angle], angle_weight[angle], \
                        spatial)
        if angular:
            change = np.linalg.norm((_to_scalar(scalar_flux, angle_weight) - _to_scalar(scalar_flux_old, angle_weight)) \
                                    /_to_scalar(scalar_flux, angle_weight) /(len(medium_map)))
        else:
            change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                            / scalar_flux / (len(medium_map)))
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux

# @numba.jit(nopython=True, cache=True)
def scalar_vacuum(scalar_flux_old, medium_map, xs_total, xs_scatter, source, \
                  point_source_loc, point_source, spatial_coef, \
                  angle_weight, spatial):

    # print(scalar_flux_old.shape)
    # print(type(scalar_flux_old[0]))
    # scalar_flux = np.zeros((scalar_flux_old.shape), dtype=(np.float64, len(scalar_flux_old.shape)))
    scalar_flux = np.zeros((scalar_flux_old.shape), dtype=np.float64)
    # scalar_flux = scalar_flux_old.copy() 
    # scalar_flux *= 0

    edge_one = 0
    edge_two = 0
    # print(type(spatial_coef))
    if spatial_coef > np.float(0):
        for cell in range(len(medium_map)):
            material = medium_map[cell]
            if cell == point_source_loc:
                edge_one += point_source
            edge_two = (xs_scatter[material] * scalar_flux_old[cell] \
                        + source[cell] + edge_one * (abs(spatial_coef) \
                        - 0.5 * xs_total[material])) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
            if spatial == 2:
                scalar_flux[cell] = angle_weight * 0.5 * (edge_one + edge_two) 
            elif spatial == 1:
                scalar_flux[cell] = angle_weight * edge_two
            edge_one = edge_two
    elif spatial_coef < np.float(0):
        for cell in range(len(medium_map)-1, -1, -1):
            material = medium_map[cell]
            if (cell + 1) == point_source_loc:
                edge_two += point_source
            edge_one = (xs_scatter[material] * scalar_flux_old[cell] \
                        + source[cell] + edge_two * (abs(spatial_coef) \
                        - 0.5 * xs_total[material])) \
                        * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
            if spatial == 2:
                scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
            elif spatial == 1:
                scalar_flux[cell] += angle_weight * edge_one
            edge_two = edge_one
    return scalar_flux

# @numba.jit(nopython=True, cache=True)
# @numba.njit
def scalar_reflected(scalar_flux_old, medium_map, xs_total, xs_scatter, \
            source, point_source_loc, point_source, spatial_coef, \
            angle_weight, spatial):
    scalar_flux = np.zeros((scalar_flux_old.shape), dtype=np.float64)
    edge_one = 0
    edge_two = 0
    for cell in range(len(medium_map)):
        material = medium_map[cell]
        if cell == point_source_loc:
            edge_one += point_source
        edge_two = (xs_scatter[material] * scalar_flux_old[cell] \
                    + source[cell] + edge_one * (abs(spatial_coef) \
                    - 0.5 * xs_total[material])) \
                    * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
        if spatial == 2:
            scalar_flux[cell] = angle_weight * 0.5 * (edge_one + edge_two) 
        elif spatial == 1:
            scalar_flux[cell] = angle_weight * edge_two
        edge_one = edge_two
    for cell in range(len(medium_map)-1, -1, -1):
        material = medium_map[cell]
        if (cell + 1) == point_source_loc:
            edge_two += point_source
        edge_one = (xs_scatter[material] * scalar_flux_old[cell] \
                    + source[cell] + edge_two * (abs(spatial_coef) \
                    - 0.5 * xs_total[material])) \
                    * 1/(abs(spatial_coef) + 0.5 * xs_total[material])
        if spatial == 2:
            scalar_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two) 
        elif spatial == 1:
            scalar_flux[cell] += angle_weight * edge_one
        edge_two = edge_one
    return scalar_flux

# @numba.jit(nopython=True, cache=True)
def _to_scalar(angular_flux, angle_weight):
    return np.sum(angular_flux * angle_weight, axis=1)

def time_sweep(scalar_flux_old, angular_flux_last, medium_map, \
         xs_total, xs_scatter, external_source, ps_locs, point_source, \
         spatial_coef, angle_weight, temporal_coef, spatial=2, \
         temporal="BE"):
    ...


def time_vacuum():
    ...


