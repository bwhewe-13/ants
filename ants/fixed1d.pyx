########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Fixed Source Multigroup Neutron Transport Problems
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

from ants cimport source_iteration_1d as si
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d

import numpy as np

def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
            double[:,:,:] xs_fission, double[:] source, double[:] boundary, \
            int[:] medium_map, double[:] delta_x, double[:] angle_x, \
            double[:] angle_w, dict params_dict):
    # Covert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Combine fission and scattering
    xs_matrix = memoryview(np.zeros((params.materials, params.groups, \
                            params.groups)))
    tools.combine_self_scattering(xs_matrix, xs_scatter, xs_fission, params)
    if params.angular == True:
        flux_old = tools.array_3d_ing(params)
        flux = si.multigroup_angular(flux_old, xs_total, xs_matrix, source, \
                    boundary, medium_map, delta_x, angle_x, angle_w, params)
        return np.asarray(flux)
    flux_old = tools.array_2d_ig(params)
    flux = si.multigroup_scalar(flux_old, xs_total, xs_matrix, source, boundary, \
                    medium_map, delta_x, angle_x, angle_w, params)
    return np.asarray(flux)
