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

from ants cimport source_iteration_2d as si
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d

import numpy as np

def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] external, double[:] boundary_x, \
        double[:] boundary_y, int[:] medium_map, double[:] delta_x, \
        double[:] delta_y, double[:] angle_x, double[:] angle_y, \
        double[:] angle_w, dict params_dict):
    # Covert dictionary to type params2d
    params = tools._to_params2d(params_dict)
    # Combine fission and scattering
    xs_matrix = memoryview(np.zeros((params.materials, params.groups, \
                            params.groups)))
    tools.combine_self_scattering(xs_matrix, xs_scatter, xs_fission, params)
    if params.angular == True:
        flux_old = tools.array_3d(params.cells_x * params.cells_y, \
                                  params.angles, params.groups)
        flux = si.multigroup_angular(flux_old, xs_total, xs_matrix, external, \
                                     boundary_x, boundary_y, medium_map, \
                                     delta_x, delta_y, angle_x, angle_y, \
                                     angle_w, params)
        flux = np.asarray(flux).reshape(params.cells_x, params.cells_y, \
                                        params.angles, params.groups)
    else:
        flux_old = tools.array_2d(params.cells_x * params.cells_y, params.groups)
        flux = si.multigroup_scalar(flux_old, xs_total, xs_matrix, external, \
                                    boundary_x, boundary_y, medium_map, \
                                    delta_x, delta_y, angle_x, angle_y, \
                                    angle_w, params)
        flux = np.asarray(flux).reshape(params.cells_x, params.cells_y, \
                                        params.groups)
    return flux
