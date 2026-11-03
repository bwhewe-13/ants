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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np

from ants cimport cytools_1d as tools
from ants cimport multi_group_1d as mg
from ants cimport parameters

from ants.datatypes import MultigroupSolver, create_params
from ants.quadrature import legendre_data
from ants.utils.sources import steady_boundary, steady_external


def fixed_source(materials, sources, geometry, quadrature, solver):
    # Anisotropic scattering
    if materials.scatter.ndim == 4:
        return fixed_source_aniso(materials, sources, geometry, quadrature, solver)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:] external
    cdef double[:,:,:] boundary_x
    cdef int[:,:] q_index
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])

    # Compact external sources are read through the slab index
    external_np, q_index_np = steady_external(sources.external, \
                    (info.cells_x,), info.angles, info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x)

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
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info, \
                    q_index)
    elif params.mg_solver == MultigroupSolver.DMD:
        flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info, \
                    q_index)

    # Return scalar flux cell centers
    if (params.angular == False) and (params.flux_at_edges == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges
    return known_flux(flux, xs_total, xs_matrix, external, boundary_x, geometry, \
                    quadrature, params, q_index)


def known_flux(double[:,:] flux, double[:,:] xs_total, double[:,:,:] xs_matrix, \
        double[:,:,:] external, double[:,:,:] boundary_x, geometry, quadrature, \
        params, int[:,:] q_index=None):
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_3d(info.cells_x, info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info, \
                        q_index)

    # Return scalar flux cell edges
    if (info.angular == False) and (info.flux_at_edges == 1):
        scalar_flux = mg._known_source_scalar(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
        return np.asarray(scalar_flux)

    # Solve for angular flux
    angular_flux = tools.array_3d(info.cells_x + info.flux_at_edges, info.angles, \
                            info.groups)
    mg._known_source_angular(angular_flux, xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    # Return angular flux (either edges or centers)
    return np.asarray(angular_flux)


def fixed_source_aniso(materials, sources, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:] external
    cdef double[:,:,:] boundary_x
    cdef int[:,:] q_index
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed1d_source_iteration(info, xs_total.shape[0])

    # Compact external sources are read through the slab index
    external_np, q_index_np = steady_external(sources.external, \
                    (info.cells_x,), info.angles, info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x)

    # Legendre polynomials for the scattering source and flux moments
    P_source_np, P_weights_np = legendre_data(quadrature, info.moments)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Add fission matrix to scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.flux_at_edges = 0

    # Initialize flux moments to zeros
    flux_old = tools.array_3d(info.cells_x, info.groups, info.moments)

    # Multigroup solver
    flux = mg.multi_group_moments(flux_old, xs_total, xs_matrix, external, \
                boundary_x, medium_map, delta_x, angle_x, angle_w, P_source, \
                P_weights, info, q_index)

    # Return scalar flux cell centers
    if (params.angular == False) and (params.flux_at_edges == 0):
        return np.ascontiguousarray(np.asarray(flux)[:,:,0])

    # For angular flux or scalar flux edges
    return known_flux_moments(flux, xs_total, xs_matrix, external, boundary_x, \
                              P_source, geometry, quadrature, params, q_index)


def known_flux_moments(double[:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:,:] xs_matrix, double[:,:,:] external, \
        double[:,:,:] boundary_x, double[:,:] P_source, geometry, \
        quadrature, params, int[:,:] q_index=None):
    # P_source is the (2l + 1) P_l array from legendre_data
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create sum_l (2l + 1) P_l (sigma_s + sigma_f)_l * phi_l + external
    source = tools.array_3d(info.cells_x, info.angles, info.groups)
    tools._source_total_aniso(source, flux, xs_matrix, medium_map, external, \
                              P_source, info, q_index)

    # Return scalar flux cell edges
    if (info.angular == False) and (info.flux_at_edges == 1):
        scalar_flux = mg._known_source_scalar(xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)
        return np.asarray(scalar_flux)

    # Solve for angular flux
    angular_flux = tools.array_3d(info.cells_x + info.flux_at_edges, info.angles, \
                            info.groups)
    mg._known_source_angular(angular_flux, xs_total, source, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    # Return angular flux (either edges or centers)
    return np.asarray(angular_flux)
