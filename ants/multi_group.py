########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE
from ants.x_sweeps import scalar_x_sweep, angular_x_sweep, time_x_sweep

import numpy as np
import numba
from tqdm import tqdm

# @numba.jit(nopython=True, cache=True)
def criticality(medium_map, xs_total, xs_scatter, xs_fission, \
                spatial_coef, angle_weight, params):
    cells = medium_map.shape[0]
    groups = xs_total.shape[1]
    scalar_flux_old = np.random.rand(cells, groups)
    scalar_flux_old /= np.linalg.norm(scalar_flux_old)
    power_source = np.zeros((cells * groups))

    point_source = np.zeros((angle_weight.shape[0]))
    converged = 0
    count = 1
    while not (converged):
        power_iteration_source(power_source, scalar_flux_old, \
                               medium_map, xs_fission)
        scalar_flux = source_iteration(medium_map, xs_total, xs_scatter, \
                        xs_fission, power_source, point_source, \
                        spatial_coef, angle_weight, params, angular=False)
        keff = np.linalg.norm(scalar_flux)
        scalar_flux /= keff
        change = convergence(scalar_flux, scalar_flux_old, \
                            angle_weight, angular=False)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux

# @numba.jit(nopython=True, cache=True)
def power_iteration_source(power_source, flux, medium_map, xs_fission):
    groups = flux.shape[1]
    power_source *= 0
    for cell, mat in enumerate(medium_map):
        for group in range(groups):
            power_source[group::groups][cell] = np.sum(flux[cell] \
                                                * xs_fission[mat][group])

# # @numba.jit(nopython=True, cache=True)
# def scalar_multi_group():
#     ...

# # @numba.jit(nopython=True, cache=True)
# def angular_multi_group():
#     ...

# # @numba.jit(nopython=True, cache=True)
# def source_iteration(medium_map, xs_total, xs_scatter, xs_fission, \
#                     external_source, point_source, spatial_coef, \
#                     angle_weight, params, angular=False):
#     cells = medium_map.shape[0]
#     groups = 
#     xs_matrix = xs_scatter + xs_fission
#     if angular == True:
#         angular_flux = 

# @numba.jit(nopython=True, cache=True)
def source_iteration(medium_map, xs_total, xs_scatter, xs_fission, \
                    external_source, point_source, spatial_coef, \
                    angle_weight, params, angular=False):
    cells = medium_map.shape[0]
    groups = xs_total.shape[1]
    if angular:
        sweep = angular_x_sweep
        angles = spatial_coef.shape[0]
        neutron_flux_old = np.zeros((cells, angles, groups), dtype=np.float64)
    else:
        sweep = scalar_x_sweep
        neutron_flux_old = np.zeros((cells, groups), dtype=np.float64)
    neutron_flux = np.zeros(neutron_flux_old.shape, dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        neutron_flux *= 0
        for group in range(groups):
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[6] == 1 else group
            neutron_flux[(...,group)] = sweep(neutron_flux_old[(...,group)], \
                    medium_map, xs_total[:,group], xs_scatter[:,group,group], \
                    external_source, point_source[ps_group_idx::params[6]], \
                    spatial_coef, angle_weight, params, ex_group_idx)
        change = convergence(neutron_flux, neutron_flux_old, \
                            angle_weight, angular=angular)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        neutron_flux_old = neutron_flux.copy()
    return neutron_flux



# @numba.jit(nopython=True, cache=True)
def convergence(flux, flux_old, angle_weight, angular=False):
    if angular:
        flux = np.sum(flux * np.atleast_3d(angle_weight),axis=1)
        flux_old = np.sum(flux_old * np.atleast_3d(angle_weight),axis=1)
    return np.linalg.norm((flux - flux_old) / flux / (flux.shape[0]))

# @numba.jit(nopython=True, cache=True)
def time_dependent(medium_map, xs_total, xs_matrix, external_source, \
                   point_source, spatial_coef, angle_weight,  velocity, \
                   params, time_step_size):
    cells = medium_map.shape[0]
    angles = spatial_coef.shape[0]
    groups = xs_total.shape[1]
    temporal_coef = _time_coef(velocity, time_step_size)
    neutron_flux = np.zeros((cells, angles, groups), dtype=np.float64)
    neutron_flux_last = np.zeros((cells, angles, groups), dtype=np.float64)
    time_step_flux = np.zeros((params[8], cells, groups), dtype=np.float64)
    time_const = 0.5 if params[7] == 1 else 0.75
    for step in tqdm(range(params[8]), desc="Time Steps"):
        neutron_flux = time_source_iteration(neutron_flux_last, medium_map, \
                        xs_total, xs_matrix, external_source, point_source, \
                        spatial_coef, angle_weight, temporal_coef, params, \
                        time_const)
        time_step_flux[step] = angular_to_scalar(neutron_flux, angle_weight)
        neutron_flux_last = neutron_flux.copy()
    return time_step_flux

# @numba.jit(nopython=True, cache=True)
def _time_coef(velocity, time_step_size):
    return 1 / (velocity * time_step_size)

# @numba.jit(nopython=True, cache=True)
def angular_to_scalar(angular_flux, angle_weight):
    return np.sum(angular_flux * np.atleast_3d(angle_weight), axis=1)

# @numba.jit(nopython=True, cache=True)
def time_source_iteration(neutron_flux_last, medium_map, xs_total, \
                          xs_matrix, external_source, point_source, \
                          spatial_coef, angle_weight, temporal_coef, \
                          params, time_const):
    cells = medium_map.shape[0]
    angles = angle_weight.shape[0]
    groups = xs_total.shape[1]
    ex_group_idx = 0
    ps_group_idx = 0
    neutron_flux_next = np.zeros((cells, angles, groups), dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        neutron_flux_next *= 0
        for group in range(groups):
            ex_group_idx = 0 if params[3] == 1 else group
            ps_group_idx = 0 if params[5] == 1 else group
            neutron_flux_next[:,:,group] = time_x_sweep(neutron_flux_last[:,:,group], \
                    medium_map, xs_total[:,group], xs_matrix[:,group,group], \
                    external_source, point_source[ps_group_idx::params[6]], \
                    spatial_coef, angle_weight, params, temporal_coef[group], \
                    time_const, ex_group_idx)
        change = convergence(neutron_flux_next, neutron_flux_last, \
                             angle_weight, True)
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        neutron_flux_last = neutron_flux_next.copy()
    return neutron_flux_next

"""
The issue is that the off_scattering function does not take old and new 
fluxes, only the new -- unlike before.
Will probably have to change this

# q_tilde = ex_source[:,group] + update_q(scatter,\
#                  scalar_flux_old, group+1, groups, group)
# if group != 0:
#     q_tilde += update_q(scatter, scalar_flux, 0, group, group)

"""
# @numba.jit(nopython=True, cache=True)
def off_scattering(medium_map, cross_section, scalar_flux, group):
    external = np.zeros((len(medium_map)))
    for cell, mat in enumerate(medium_map):
        external[cell] = sum(np.delete(cross_section[mat][group], group) \
                             * np.delete(scalar_flux[cell],group))
    return external
