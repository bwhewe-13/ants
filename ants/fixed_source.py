########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import ants.constants as constants
import ants.multi_group as multi_group

import numpy as np
import numba

@numba.jit(nopython=True,cache=True)
def _initialize(spatial_cells, spatial_coef, velocity, time_steps, \
                time_step_size):
    if time_steps == 0:
        temporal_coef = np.zeros(velocity.shape)
    else:
        temporal_coef = 1 / (velocity * time_step_size) 
    scalar_flux = np.zeros((spatial_cells, len(velocity)))
    if len(np.unique(np.sign(spatial_coef))) == 1:
        angular_flux = np.zeros((spatial_cells, len(spatial_coef)*2, \
                                 len(velocity)))
    else:
        angular_flux = np.zeros((spatial_cells, len(spatial_coef), \
                         len(velocity)))
    return temporal_coef, angular_flux, scalar_flux

@numba.jit(nopython=True, cache=True)
def backward_euler(medium_map, xs_total, xs_scatter, xs_fission, \
            external_source, ps_locs, point_source, spatial_coef, \
            angle_weight, velocity, time_steps, time_step_size, \
            spatial="diamond"):
    temporal_coef, angular_flux_last, scalar_flux_old = _initialize( \
                            len(medium_map), spatial_coef, velocity, \
                            time_steps, time_step_size)
    full_scalar_flux = []
    if time_steps == 0:
        scalar_flux, angular_flux = multi_group.source_iteration(scalar_flux_old, \
                    angular_flux_last, medium_map, xs_total, xs_scatter, \
                    xs_fission, external_source, ps_locs, point_source, \
                    spatial_coef, angle_weight, temporal_coef, \
                    spatial=spatial, temporal="BE")
        full_scalar_flux.append(scalar_flux)
    for time_step in range(time_steps):
        scalar_flux, angular_flux = multi_group.source_iteration(scalar_flux_old, \
                    angular_flux_last, medium_map, xs_total, xs_scatter, \
                    xs_fission, external_source, ps_locs, point_source, \
                    spatial_coef, angle_weight, temporal_coef, \
                    spatial=spatial, temporal="BE")
        angular_flux_last = angular_flux.copy()
        full_scalar_flux.append(scalar_flux)
        scalar_flux_old = scalar_flux.copy()
    return full_scalar_flux, angular_flux


