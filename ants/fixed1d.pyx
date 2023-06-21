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

import numpy as np

from ants cimport multi_group_1d as mg
from ants cimport cytools_1d as tools
from ants cimport parameters


def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] external, \
        double[:] boundary_x, int[:] medium_map, double[:] delta_x, \
        double[:] angle_x, double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])
    # Add fission matrix to scattering
    tools._xs_matrix(xs_scatter, xs_fission, info)
    # Initialize flux_old to zeros
    flux_old = tools.array_2d(info.cells_x + info.edges, info.groups)
    # Run source iteration
    flux = mg.source_iteration(flux_old, xs_total, xs_scatter, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    if info.angular:
        # Create (sigma_s + sigma_f) * phi + external function
        source = tools.array_1d((info.cells_x + info.edges) * info.angles * info.groups)
        tools._source_total(source, flux, xs_scatter, medium_map, external, info)
        # Solve for angular flux using scalar flux
        angular_flux = mg._known_source(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
        return np.asarray(angular_flux)
    # Convert to numpy array
    return np.asarray(flux)
