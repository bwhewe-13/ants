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

@numba.jit(nopython=True, cache=True)
def _generate_source(medium_map, cross_section, scalar_flux):
    source = np.zeros(scalar_flux.shape, dtype=np.float64)
    for cell, mat in enumerate(medium_map):
        source[cell] = cross_section[mat] @ scalar_flux[cell]
    return source

@numba.jit(nopython=True, cache=True)
def keigenvalue(medium_map, xs_total, xs_scatter, xs_fission, \
            spatial_coef, angle_weight, spatial="diamond"):
    scalar_flux_old = np.random.rand(len(medium_map), len(xs_total[0]))
    scalar_flux_old /= np.linalg.norm(scalar_flux_old)
    angular_flux = np.zeros((len(medium_map), len(angle_weight), len(xs_total[0])), dtype=np.float64)
    # angular_flux = np.zeros(len(medium_map), dtype=np.float64)
    converged = 0
    count = 1
    while not (converged):
        fission_source = _generate_source(medium_map, xs_fission, scalar_flux_old)
        scalar_flux, angular_flux = multi_group.source_iteration(scalar_flux_old, \
            angular_flux, medium_map, xs_total, xs_scatter, \
            np.array([0], dtype=np.float64), fission_source, np.array([-1], dtype=np.float64), \
            np.array([0],dtype=np.float64), spatial_coef, angle_weight, np.array([0]), \
            spatial=spatial, temporal="BE")

        # scalar_flux, angular_flux_last = multi_group.source_iteration(scalar_flux_old, \
        #             angular_flux, medium_map, xs_total, xs_scatter, \
        #             np.array([0], dtype=np.float64), fission_source, np.array([-1], dtype=np.float64), \
        #             np.array([0],dtype=np.float64), spatial_coef, angle_weight, np.array([0]), \
        #             spatial=spatial, temporal="BE")
        keff = np.linalg.norm(scalar_flux)
        scalar_flux /= keff
        change = np.linalg.norm((scalar_flux - scalar_flux_old) \
                                /scalar_flux/(len(medium_map)))
        # print('Iteration {}\n{}\nChange {} Keff {}'.format(count, \
        #              '='*35, change, keff))
        converged = (change < constants.OUTER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS)
        count += 1
        scalar_flux_old = scalar_flux.copy()
    print(keff)
    return scalar_flux, keff

