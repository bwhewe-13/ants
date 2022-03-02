########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
#
########################################################################

import ants.constants as constants
import ants.x_sweeps as sweeps


import numpy as np
import numba

# @numba.jit(nopython=True) #, cache=True)
def source_iteration(groups, mu_delta_x, weight, total, scatter, \
                     fission, ex_source, medium_map, xboundary, \
                     cell_width, point_sources, scalar_flux_old=None):
    converged = 0
    count = 1
    if scalar_flux_old is None:
        scalar_flux_old = np.zeros((len(medium_map), groups))
    print(ex_source.shape)
    while not (converged):
        scalar_flux = np.zeros(scalar_flux_old.shape)
        for group in range(groups):
            # q_tilde = ex_source[:,group] + update_q(scatter,\
            #                  scalar_flux_old, group+1, groups, group)
            # if group != 0:
            #     q_tilde += update_q(scatter, scalar_flux, 0, group, group)
            scalar_flux[:,group], angular = sweeps.diamond(scalar_flux_old[:,group], total[:,group], \
                scatter[:,group,group], ex_source[:,group], medium_map, \
                mu_delta_x, weight, xboundary, point_sources[0], point_sources[1][:,group])
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))        
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    return scalar_flux, angular

def update_q(xs, phi, start, stop, g):
    return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)
