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

from ants.constants import INNER_TOLERANCE, MAX_ITERATIONS

import numpy as np
import numba

# @numba.jit(nopython=True, cache=True)
def x_scalar_sweep(neutron_flux_old, medium_map, xs_total, xs_scatter, \
                    external_source, point_source, spatial_coef, \
                    angle_weight, params, ex_group_idx):
    cells = medium_map.shape[0]
    angles = angle_weight.shape[0]
    ex_angle_idx = 0
    neutron_flux = np.zeros(neutron_flux_old.shape, dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        neutron_flux *= 0
        for angle in range(angles):
            ex_angle_idx == 0 if params[4] == 1 else angle
            if params[2] == 0:
                vacuum(neutron_flux, neutron_flux_old, medium_map, xs_total, \
                      xs_scatter, external_source, point_source[angle], \
                      spatial_coef[angle], angle_weight[angle], params, \
                      ex_group_idx, ex_angle_idx)
            if params[2] == 1:
                reflected(neutron_flux, neutron_flux_old, medium_map, xs_total, \
                      xs_scatter, external_source, point_source[angle], \
                      spatial_coef[angle], angle_weight[angle], params, \
                      ex_group_idx, ex_angle_idx)            
        change = scalar_convergence(neutron_flux, neutron_flux_old)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        neutron_flux_old = neutron_flux.copy()
    return neutron_flux

# @numba.jit(nopython=True, cache=True)
def scalar_convergence(flux, flux_old): 
    return np.linalg.norm((flux - flux_old) / flux / (flux.shape[0]))

# @numba.jit(nopython=True, cache=True)
def angular_convergence(flux, flux_old, angle_weight):
    flux = np.sum(flux * angle_weight,axis=1)
    flux_old = np.sum(flux_old * angle_weight,axis=1)
    return np.linalg.norm((flux - flux_old) / flux / (flux.shape[0]))

# @numba.jit(nopython=True, cache=True)
def vacuum(neutron_flux, neutron_flux_old, medium_map, xs_total, xs_scatter, \
           external_source, point_source, spatial_coef, angle_weight, \
           params, ex_group_idx, ex_angle_idx):
    edge_one = 0
    edge_two = 0
    xs1_const, xs2_const = (0, 1) if params[1] == 1 else (-0.5, 0.5)
    cells = medium_map.shape[0]
    if spatial_coef > 0:
        for cell in range(cells):
            material = medium_map[cell]
            if cell == params[5]:
                edge_one += point_source
            edge_two = (xs_scatter[material] * neutron_flux_old[cell] \
                + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                + edge_one * (abs(spatial_coef) + xs1_const * xs_total[material])) \
                /(abs(spatial_coef) + xs2_const * xs_total[material])
            if params[1] == 1: # Step Method
                neutron_flux[cell] += angle_weight * edge_two
            elif params[1] == 2: # Diamond Difference
                neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)
            edge_one = edge_two
    elif spatial_coef < 0:
        for cell in range(cells-1, -1, -1):
            material = medium_map[cell]
            if (cell + 1) == params[5]:
                edge_two += point_source
            edge_one = (xs_scatter[material] * neutron_flux_old[cell] \
                + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                + edge_two * (abs(spatial_coef) + xs1_const * xs_total[material])) \
                /(abs(spatial_coef) + xs2_const * xs_total[material])
            if params[1] == 1: # Step Method
                neutron_flux[cell] += angle_weight * edge_one
            elif params[1] == 2: # Diamond Difference
                neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)
            edge_two = edge_one


# @numba.jit(nopython=True, cache=True)
def reflected(neutron_flux, neutron_flux_old, medium_map, xs_total, \
              xs_scatter, external_source, point_source, spatial_coef, \
              angle_weight, params, ex_group_idx, ex_angle_idx):
    edge_one = 0
    edge_two = 0
    xs1_const, xs2_const = (0, 1) if params[1] == 1 else (-0.5, 0.5)
    cells = medium_map.shape[0]
    for cell in range(cells):
        material = medium_map[cell]
        if cell == params[5]:
            edge_one += point_source
        edge_two = (xs_scatter[material] * neutron_flux_old[cell] \
            + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
            + edge_one * (abs(spatial_coef) + xs1_const * xs_total[material])) \
            * 1/(abs(spatial_coef) + xs2_const * xs_total[material])
        if params[1] == 1: # Step Method
            neutron_flux[cell] += angle_weight * edge_two
        elif params[1] == 2: # Diamond Difference
            neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)
        edge_one = edge_two
    for cell in range(cells-1, -1, -1):
        edge_two = edge_one
        material = medium_map[cell]
        if (cell + 1) == params[5]:
            edge_two += point_source
        edge_one = (xs_scatter[material] * neutron_flux_old[cell] \
            + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
            + edge_two * (abs(spatial_coef) + xs1_const * xs_total[material])) \
            * 1/(abs(spatial_coef) + xs2_const * xs_total[material])
        if params[1] == 1: # Step Method
            neutron_flux[cell] += angle_weight * edge_one
        elif params[1] == 2: # Diamond Difference
            neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)


# @numba.jit(nopython=True, cache=True)
def x_angular_sweep(neutron_flux_old, medium_map, xs_total, xs_scatter, \
                    external_source, point_source, spatial_coef, \
                    angle_weight, params, ex_group_idx):
    cells = medium_map.shape[0]
    angles = angle_weight.shape[0]
    ex_angle_idx = 0
    neutron_flux = np.zeros(neutron_flux_old.shape, dtype=np.float64)
    dummy_angle_weight = np.ones((angles))
    converged = 0
    count = 1
    while not (converged):
        neutron_flux *= 0
        scalar_flux = angular_to_scalar(neutron_flux_old, angle_weight)
        for angle in range(angles):
            ex_angle_idx == 0 if params[4] == 1 else angle
            if params[2] == 0:
                vacuum(neutron_flux[:,angle], scalar_flux, medium_map, \
                      xs_total, xs_scatter, external_source, \
                      point_source[angle], spatial_coef[angle], \
                      dummy_angle_weight[angle], params, ex_group_idx, \
                      ex_angle_idx)
            elif params[2] == 1:
                reflected(neutron_flux[:,angle], scalar_flux, medium_map, \
                      xs_total, xs_scatter, external_source, \
                      point_source[angle], spatial_coef[angle], \
                      dummy_angle_weight[angle], params, ex_group_idx, \
                      ex_angle_idx)                
        change = angular_convergence(neutron_flux, neutron_flux_old, angle_weight)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        neutron_flux_old = neutron_flux.copy()
    return neutron_flux

# @numba.jit(nopython=True, cache=True)
def angular_to_scalar(angular_flux, angle_weight):
    return np.sum(angular_flux * angle_weight, axis=1)

# @numba.jit(nopython=True, cache=True)
def x_time_sweep(neutron_flux_old, medium_map, xs_total, xs_matrix, \
                 external_source, point_source, spatial_coef, angle_weight, \
                 params, temporal_coef, time_const, ex_group_idx):
    cells = medium_map.shape[0]
    angles = angle_weight.shape[0]
    ex_angle_idx = 0
    neutron_flux = np.zeros(neutron_flux_old.shape, dtype=np.float64)
    angular_flux_last = neutron_flux_old.copy()
    dummy_angle_weight = np.ones((angles))
    converged = 0
    count = 1
    while not (converged):
        neutron_flux *= 0
        scalar_flux = angular_to_scalar(neutron_flux_old, angle_weight)
        for angle in range(angles):
            ex_angle_idx = 0 if params[4] == 1 else angle
            time_vacuum(neutron_flux[:,angle], scalar_flux, \
                    angular_flux_last[:,angle], medium_map, xs_total, \
                    xs_matrix, external_source, point_source[angle], \
                    spatial_coef[angle], dummy_angle_weight[angle], \
                    params, temporal_coef, time_const, ex_group_idx, \
                    ex_angle_idx)
        change = angular_convergence(neutron_flux, neutron_flux_old, angle_weight)
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        # print(change, np.sum(neutron_flux))
        neutron_flux_old = neutron_flux.copy()
    return neutron_flux

# @numba.jit(nopython=True, cache=True)
def time_vacuum(neutron_flux, neutron_flux_old, angular_flux_last, \
                medium_map, xs_total, xs_matrix, external_source, \
                point_source, spatial_coef, angle_weight, params, \
                temporal_coef, time_const, ex_group_idx, ex_angle_idx):
    edge_one = 0
    edge_two = 0
    xs1_const, xs2_const = (0, 1) if params[1] == 1 else (-0.5, 0.5)
    cells = medium_map.shape[0]
    if spatial_coef > 0:
        for cell in range(cells):
            material = medium_map[cell]
            if cell == params[5]:
                edge_one += point_source
            edge_two = (xs_matrix[material] * neutron_flux_old[cell] \
                + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                + angular_flux_last[cell] * temporal_coef + edge_one \
                * (abs(spatial_coef) + xs1_const * xs_total[material] \
                - time_const * temporal_coef)) * 1/(abs(spatial_coef) \
                + xs2_const * xs_total[material] + time_const * temporal_coef)
            if params[1] == 1: # Step Method
                neutron_flux[cell] += angle_weight * edge_two
            elif params[1] == 2: # Diamond Difference
                neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)                
            edge_one = edge_two
    elif spatial_coef < 0:
        for cell in range(cells-1, -1, -1):
            material = medium_map[cell]
            if (cell + 1) == params[5]:
                edge_two += point_source
            edge_one = (xs_matrix[material] * neutron_flux_old[cell] \
                + external_source[ex_group_idx+ex_angle_idx*params[3]::params[4]*params[3]][cell] \
                + angular_flux_last[cell] * temporal_coef + edge_two \
                * (abs(spatial_coef) + xs1_const * xs_total[material] \
                - time_const * temporal_coef)) * 1/(abs(spatial_coef) \
                + xs2_const * xs_total[material] + time_const * temporal_coef)
            if params[1] == 1: # Step Method
                neutron_flux[cell] += angle_weight * edge_one
            elif params[1] == 2: # Diamond Difference
                neutron_flux[cell] += angle_weight * 0.5 * (edge_one + edge_two)
            edge_two = edge_one
