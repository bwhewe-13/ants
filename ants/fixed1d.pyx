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
from ants.datatypes import create_params, MultigroupSolver


def fixed_source(materials, sources, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef double[:,:,:] external = sources.external
    cdef double[:,:,:] boundary_x = sources.boundary_x
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.flux_at_edges = 0

    # Initialize flux_old to zeros
    flux_old = tools.array_2d(info.cells_x, info.groups)

    # Multigroup solver
    if params.mg_solver == MultigroupSolver.SOURCE_ITERATION:
        flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)
    elif params.mg_solver == MultigroupSolver.DMD:
        flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)

    # Return scalar flux cell centers
    if (params.angular == False) and (params.flux_at_edges == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges
    return known_flux(flux, xs_total, xs_matrix, external, boundary_x, geometry, \
                    quadrature, params)


def known_flux(double[:,:] flux, double[:,:] xs_total, double[:,:,:] xs_matrix, \
        double[:,:,:] external, double[:,:,:] boundary_x, geometry, quadrature, params):
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_3d(info.cells_x, info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info)

    # Return scalar flux cell edges
    if (info.angular == False) and (info.flux_at_edges == 1):
        scalar_flux = mg._known_source_scalar(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
        return np.asarray(scalar_flux)

    # Solve for angular flux
    angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    # Return angular flux (either edges or centers)
    return np.asarray(angular_flux)