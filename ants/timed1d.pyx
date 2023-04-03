########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

from ants cimport time_dependent_1d as td
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d

import numpy as np

def backward_euler(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] velocity, double[:] external, \
        double[:] boundary, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Combine fission and scattering
    xs_matrix = memoryview(np.zeros((params.materials, params.groups, \
                            params.groups)))
    tools.combine_self_scattering(xs_matrix, xs_scatter, xs_fission, params)
    xs_total_v = memoryview(np.zeros((params.materials, params.groups)))
    tools.combine_total_velocity(xs_total_v, xs_total, velocity, params)
    flux_last = tools.array_3d(params.cells, params.angles, params.groups)
    flux = td.multigroup_bdf1(flux_last, xs_total_v, xs_matrix, velocity, \
                              external, boundary, medium_map, delta_x, \
                              angle_x, angle_w, params)
    flux = np.asarray(flux).reshape(params.steps, params.cells, \
                                    params.angles, params.groups)
    return flux
