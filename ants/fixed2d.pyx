########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-Dimensional Fixed Source Multigroup Neutron Transport Problems
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

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants cimport parameters


def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:] external, double[:] boundary_x, \
        double[:] boundary_y, int[:,:] medium_map, double[:] delta_x, \
        double[:] delta_y, double[:] angle_x, double[:] angle_y, \
        double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed2d_source_iteration(info, xs_total.shape[0])
    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    # Solve for cell center first
    info.edges = 0
    # Initialize flux_old to zeros
    flux_old = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    # Run source iteration
    flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    # Return scalar flux cell centers
    if (info.angular == False) and (params_dict.get("edges", 0) == 0):
        return np.asarray(flux)
    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, params_dict)


def known_source_calculation(double[:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:] external, double[:] boundary_x, \
        double[:] boundary_y, int[:,:] medium_map, double[:] delta_x, \
        double[:] delta_y, double[:] angle_x, double[:] angle_y, \
        double[:] angle_w, dict params_dict):
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_1d(info.cells_x * info.cells_y * info.angles \
                            * info.angles * info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info)
    # Return scalar flux cell edges
    if (info.angular == False) and (info.edges == 1):
        scalar_flux = mg._known_source_scalar(xs_total, source, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, info)
        return np.asarray(scalar_flux)
    # Solve for angular flux
    angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, info)
    # Return angular flux (either edges or centers)
    return np.asarray(angular_flux)