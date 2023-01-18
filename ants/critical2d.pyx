########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-Dimensional Criticality Multigroup Neutron Transport Problems
#
########################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

from ants cimport power_iteration_2d as pi
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d

import numpy as np

def power_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
                    double[:,:,:] xs_fission, int[:] medium_map, \
                    double[:] delta_x, double[:] delta_y, double[:] angle_x, \
                    double[:] angle_y, double[:] angle_w, dict params_dict):
    cdef double keff[1]
    keff[0] = 0.95
    # Convert dictionary to type params2d
    params = tools._to_params2d(params_dict)
    # Initialize components
    flux_old = np.random.rand(params.cells_x * params.cells_y, params.groups)
    tools.normalize_flux(flux_old, params)
    power_source = memoryview(np.zeros((params.cells_x * params.cells_y * params.groups)))
    flux = pi.multigroup(flux_old, xs_total, xs_scatter, xs_fission, \
                    power_source, medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, params, keff)
    return np.asarray(flux).reshape(params.cells_x, params.cells_y, \
                                    params.groups), keff[0]
