########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Criticality Multigroup Neutron Transport Problems
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

from ants cimport power_iteration_1d as pi
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d

import numpy as np


def power_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    cdef double keff[1]
    keff[0] = 0.95
    # Convert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Initialize components
    flux_old = np.random.rand(params.cells, params.groups)
    tools.normalize_flux(flux_old, params)
    power_source = memoryview(np.zeros((params.cells * params.groups)))
    flux = pi.multigroup(flux_old, xs_total, xs_scatter, xs_fission, \
                         power_source, medium_map, delta_x, angle_x, \
                         angle_w, params, keff)
    flux = np.asarray(flux).reshape(params.cells, params.groups)
    return flux, keff[0]


def nearby_power(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] nearby_source, \
        int[:] medium_map, double[:] delta_x, double[:] angle_x, \
        double[:] angle_w, double nearby_rate, dict params_dict):
    # Convert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Initialize components
    flux_old = np.random.rand(params.cells, params.groups)
    # Initialize keffective
    cdef double keff[1]
    keff[0] = tools.nearby_keffective(flux_old, nearby_rate, params)
    power_source = memoryview(np.zeros((params.cells * params.angles * params.groups)))
    flux = pi.nearby(flux_old, xs_total, xs_scatter, xs_fission, \
                     power_source, nearby_source, medium_map, delta_x, \
                     angle_x, angle_w, params, keff)
    flux = np.asarray(flux).reshape(params.cells, params.groups)
    return flux, keff[0]
