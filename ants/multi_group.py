########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import ants.constants as constants

import ants.sweeps.x_sweeps as sweeps

import numpy as np
import numba

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

# @numba.jit(nopython=True, cache=True)
def source_iteration(scalar_flux_old, angular_flux_last, medium_map, \
            xs_total, xs_scatter, xs_fission, external_source, ps_locs, \
            point_source, spatial_coef, angle_weight, temporal_coef, \
            spatial="diamond", temporal="BE"):
    angular_flux = np.zeros(angular_flux_last.shape, dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        scalar_flux = np.zeros(scalar_flux_old.shape, dtype=np.float64)
        combined_source = external_source.copy()
        for group in range(scalar_flux_old.shape[1]):
            idx_ex = (..., group)
            idx_ps = (..., group)
            if point_source.ndim == 1:
                idx_ps = ()
            # combined_source[:,:,group] += np.repeat(off_scattering(medium_map, \
            #                     xs_scatter, scalar_flux, group)[:,None], \
            #                     len(angle_weight), axis=1)
            scalar_flux[:,group], angular_flux[:,:,group] = sweeps.discrete_ordinates(\
                scalar_flux_old[:,group], angular_flux_last[:,:,group], medium_map, \
                xs_total[:,group], xs_scatter[:,group,group], \
                combined_source[idx_ex], ps_locs, point_source[idx_ps], \
                spatial_coef, angle_weight, temporal_coef[group], spatial=spatial, \
                temporal=temporal)
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        # print("Out Count", count, "Change", change)
        # if np.isnan(change) or np.isinf(change):
        #     change = 0.
        #     print(np.sum(scalar_flux))
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular_flux

# @numba.jit(nopython=True, cache=True)
def source_iteration_scalar(medium_map, xs_total, xs_scatter, \
            xs_fission, external_source, point_source_loc, \
            point_source, spatial_coef, angle_weight, \
            spatial=2, boundary=0, angular=False):
    if angular:
        scalar_flux_old = np.zeros((len(medium_map), len(angle_weight), \
                                        len(xs_total[0])), dtype=np.float64)
    else:
        scalar_flux_old = np.zeros((len(medium_map), \
                                    len(xs_total[0])), dtype=np.float64)
    print("Scalar Flux Old", scalar_flux_old.shape)
    converged = 0
    count = 1
    while not (converged):
        scalar_flux = np.zeros(scalar_flux_old.shape, dtype=np.float64)
        combined_source = external_source.copy()
        for group in range(scalar_flux_old.shape[-1]):
            idx_ps = () if point_source.ndim == 1 else (..., group)
            scalar_flux[(..., group)] = sweeps.scalar_sweep(scalar_flux_old[(..., group)], \
                    medium_map, xs_total[:,group], xs_scatter[:,group,group], \
                    combined_source[(...,group)], point_source_loc, \
                    point_source[idx_ps], \
                    spatial_coef, angle_weight, spatial=spatial, \
                    boundary=boundary, angular=angular)
        if angular:
            change = np.linalg.norm((_to_scalar(scalar_flux, angle_weight) - _to_scalar(scalar_flux_old, angle_weight)) \
                                    /_to_scalar(scalar_flux, angle_weight) /(len(medium_map)))
        else:
            change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                    /scalar_flux/(len(medium_map)))
        print(count, change, np.sum(scalar_flux))
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux

# @numba.jit(nopython=True, cache=True)
def source_iteration_angular(medium_map, xs_total, xs_scatter, \
            xs_fission, external_source, point_source_loc, \
            point_source, spatial_coef, angle_weight, \
            spatial="diamond", boundary="vacuum"):
    angular_flux_old = np.zeros((len(medium_map), len(angle_weight), \
                                len(xs_total[0])), dtype=np.float64)
    scalar_flux_old = angular_to_scalar(angular_flux_old, angle_weight)
    converged = 0
    count = 1
    while not (converged):
        angular_flux = np.zeros(angular_flux_old.shape, dtype=np.float64)
        combined_source = external_source.copy()
        for group in range(scalar_flux_old.shape[1]):
            idx_ps = () if point_source.ndim == 1 else (..., group)
            angular_flux[:,:,group] = sweeps.angular_sweep(angular_flux_old[:,:,group], \
                    medium_map, xs_total[:,group], xs_scatter[:,group,group], \
                    combined_source[(...,group)], point_source_loc, \
                    point_source[idx_ps], \
                    spatial_coef, angle_weight, spatial=spatial, \
                    boundary=boundary)
        scalar_flux = angular_to_scalar(angular_flux, angle_weight)
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        # converged = count >= constants.MAX_ITERATIONS
        count += 1
        angular_flux_old = angular_flux.copy()
        scalar_flux_old = scalar_flux.copy()
        # scalar_flux_old = angular_to_scalar(angular_flux, angle_weight)
    return angular_flux

# @numba.jit(nopython=True, cache=True)
def _to_scalar(angular_flux, angle_weight):
    return np.sum(angular_flux * angle_weight[None,:,None], axis=1)


# Boundary: 0 for vacuum
#           1 for reflected
# Spatial Discretization: 1 for step
#                         2 for diamond difference