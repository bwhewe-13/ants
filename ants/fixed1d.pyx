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
from ants.datatypes import CrossSections, QuadratureData, SpatialGrid


def source_iteration(xs, double[:,:,:] external, \
        double[:,:,:] boundary_x, int[:] medium_map, grid, \
        quad, dict params_dict):

    _xs_total = xs.total
    cdef double[:,:] xs_total = _xs_total
    _xs_scatter = xs.scatter
    cdef double[:,:,:] xs_scatter = _xs_scatter
    _xs_fission = xs.fission
    cdef double[:,:,:] xs_fission = _xs_fission
    _delta_x = grid.delta_x
    cdef double[:] delta_x = _delta_x
    _angle_x = quad.angle_x
    cdef double[:] angle_x = _angle_x
    _angle_w = quad.angle_w
    cdef double[:] angle_w = _angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.edges = 0

    # Initialize flux_old to zeros
    flux_old = tools.array_2d(info.cells_x, info.groups)

    # Run source iteration
    flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, info)

    # Return scalar flux cell centers
    if (info.angular == False) and (params_dict.get("edges", 0) == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
            boundary_x, medium_map, grid, quad, params_dict)


def dynamic_mode_decomp(xs, double[:,:,:] external, \
        double[:,:,:] boundary_x, int[:] medium_map, grid, \
        quad, dict params_dict):

    _xs_total = xs.total
    cdef double[:,:] xs_total = _xs_total
    _xs_scatter = xs.scatter
    cdef double[:,:,:] xs_scatter = _xs_scatter
    _xs_fission = xs.fission
    cdef double[:,:,:] xs_fission = _xs_fission
    _delta_x = grid.delta_x
    cdef double[:] delta_x = _delta_x
    _angle_x = quad.angle_x
    cdef double[:] angle_x = _angle_x
    _angle_w = quad.angle_w
    cdef double[:] angle_w = _angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.edges = 0

    # Initialize flux_old to zeros
    flux_old = tools.array_2d(info.cells_x, info.groups)

    # Run source iteration
    flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)

    # Return scalar flux cell centers
    if (info.angular == False) and (params_dict.get("edges", 0) == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
            boundary_x, medium_map, grid, quad, params_dict)


def known_source_calculation(double[:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:] external, \
        double[:,:,:] boundary_x, int[:] medium_map, grid, \
        quad, dict params_dict):

    _delta_x = grid.delta_x
    cdef double[:] delta_x = _delta_x
    _angle_x = quad.angle_x
    cdef double[:] angle_x = _angle_x
    _angle_w = quad.angle_w
    cdef double[:] angle_w = _angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params_dict)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_3d(info.cells_x, info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info)

    # Return scalar flux cell edges
    if (info.angular == False) and (info.edges == 1):
        scalar_flux = mg._known_source_scalar(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
        return np.asarray(scalar_flux)

    # Solve for angular flux
    angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    # Return angular flux (either edges or centers)
    return np.asarray(angular_flux)