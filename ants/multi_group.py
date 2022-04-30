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

# @numba.jit(nopython=True) #, cache=True)
def off_scattering(medium_map, cross_section, scalar_flux, group):
    external = np.zeros((len(medium_map)))
    for cell, mat in enumerate(medium_map):
        external[cell] = sum(np.delete(cross_section[mat][group], group) \
                             * np.delete(scalar_flux[cell],group))
    return external

# @numba.jit(nopython=True) #, cache=True)
def source_iteration(scalar_flux_old, angular_flux_last, medium_map, \
            xs_total, xs_scatter, xs_fission, external_source, ps_locs, \
            point_source, spatial_coef, angle_weight, temporal_coef, \
            spatial="diamond", temporal="BE"):
    angular_flux = np.zeros(angular_flux_last.shape)
    converged = 0
    count = 1
    while not (converged):
        scalar_flux = np.zeros(scalar_flux_old.shape)
        combined_source = external_source.copy()
        for group in range(scalar_flux_old.shape[1]):
            # idx_ex = tuple([slice(None)] * (combined_source.ndim - 1) + [group])
            # if len(idx_ex) == 1:
            #     idx_ex = (None)
            time_coef = 0 if temporal_coef is None else temporal_coef[group]
            # combined_source[:,:,group] += np.repeat(off_scattering(medium_map, \
            #                     xs_scatter, scalar_flux, group)[:,None], \
            #                     len(angle_weight), axis=1)
            scalar_flux[:,group], angular_flux[:,:,group] = sweeps.discrete_ordinates(\
                scalar_flux_old[:,group], angular_flux_last[:,:,group], medium_map, \
                xs_total[:,group], xs_scatter[:,group,group], \
                combined_source[:,:,group], ps_locs, point_source[:,:,group], \
                spatial_coef, angle_weight, time_coef, spatial=spatial, \
                temporal=temporal)

            # scalar_flux[:,group] = fluxes[0].copy()
            # if angular_flux_last is not None:
            #     angular_flux[:,:,group] = fluxes[1].copy()

        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))        
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular_flux
