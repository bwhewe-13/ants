########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Three-Dimensional Fixed Source Multigroup Neutron Transport Problems
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

from ants cimport cytools_3d as tools
from ants cimport multi_group_3d as mg
from ants cimport parameters

from ants.datatypes import MultigroupSolver, create_params
from ants.quadrature import (
    _check_reflection,
    _scatter_harmonics,
    artificial_scatter_matrix,
    harmonic_data,
)
from ants.utils.pytools import artificial_scatter_iteration
from ants.utils.sources import (
    check_steady_boundary,
    check_steady_external,
    dense_source,
    steady_boundary,
    steady_external,
)


def fixed_source(materials, sources, geometry, quadrature, solver):
    # Anisotropic scattering
    if materials.scatter.ndim == 4:
        return fixed_source_aniso(materials, sources, geometry, quadrature, solver)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:,:,:] external
    cdef double[:,:,:,:,:] boundary_x
    cdef double[:,:,:,:,:] boundary_y
    cdef double[:,:,:,:,:] boundary_z
    cdef int[:,:] q_index
    cdef int[:,:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] delta_z = geometry.delta_z
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_z = quadrature.angle_z
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed3d_source_iteration(info, xs_total.shape[0])
    _check_inputs(geometry, quadrature, params)

    # Compact external sources are read through the slab index. The
    # artificial in-scatter source is dense, so as-SN uses a dense source.
    if info.sigma_as > 0.0:
        external_np, q_index_np = dense_source(sources.external), None
    else:
        external_np, q_index_np = steady_external(sources.external, \
                (info.cells_x, info.cells_y, info.cells_z), info.angles, \
                info.groups)
    if q_index_np is None:
        check_steady_external(external_np, (info.cells_x, info.cells_y, \
                              info.cells_z), info.angles, info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x, (info.cells_y, info.cells_z), \
                    info.angles, info.groups, "boundary_x")
    boundary_y = steady_boundary(sources.boundary_y, (info.cells_x, info.cells_z), \
                    info.angles, info.groups, "boundary_y")
    boundary_z = steady_boundary(sources.boundary_z, (info.cells_x, info.cells_y), \
                    info.angles, info.groups, "boundary_z")

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.flux_at_edges = 0

    # Initialize flux_old to zeros
    flux_old = tools.array_4d(info.cells_x, info.cells_y, info.cells_z, info.groups)

    # Effective total cross section / external source returned by the as-SN
    # solver (plain arrays when artificial scattering is disabled)
    xs_total_eff = xs_total
    external_eff = external

    # Multigroup solver
    if info.sigma_as > 0.0:
        # Artificial scattering (as-SN) ray effect mitigation
        flux, xs_total_eff, external_eff = source_iteration_as(flux_old, \
                    xs_total, xs_matrix, external, boundary_x, boundary_y, \
                    boundary_z, medium_map, delta_x, delta_y, delta_z, angle_x, \
                    angle_y, angle_z, angle_w, info)
    elif params.mg_solver == MultigroupSolver.SOURCE_ITERATION:
        flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                    delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, \
                    info, q_index)
    elif params.mg_solver == MultigroupSolver.DMD:
        flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                    delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, \
                    info, q_index)

    # Return scalar flux cell centers
    if (info.angular == False) and (params.flux_at_edges == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges. For as-SN runs, the effective
    # total cross section and combined source reproduce the converged
    # transport balance in the known-source sweeps, see fixed2d.
    return known_flux(flux, xs_total_eff, xs_matrix, external_eff, boundary_x, \
                boundary_y, boundary_z, geometry, quadrature, params, q_index)


def _check_inputs(geometry, quadrature, params):
    # The sweeps do not check for missing arrays or bounds
    if (quadrature.angle_y is None) or (quadrature.angle_z is None):
        raise ValueError("Need angle_y and angle_z for a 3D quadrature, " \
                         "see ants.angular_xyz")
    if (geometry.delta_y is None) or (geometry.delta_z is None):
        raise ValueError("Need delta_y and delta_z for a 3D geometry")

    angles = (quadrature.angle_x, quadrature.angle_y, quadrature.angle_z, \
              quadrature.angle_w)
    if any(np.shape(angle) != (params.angles,) for angle in angles):
        raise ValueError("angle_x, angle_y, angle_z, and angle_w need the " \
                         "same number of ordinates")
    # The sweep direction is set by the sign of each cosine
    if any(np.any(angle == 0.0) for angle in angles[:3]):
        raise ValueError("Need nonzero angle_x, angle_y, and angle_z")

    cells = (params.cells_x, params.cells_y, params.cells_z)
    medium_map = np.asarray(geometry.medium_map)
    if medium_map.shape != cells:
        raise ValueError(f"medium_map has shape {medium_map.shape}, " \
                         f"expected {cells}")
    if (medium_map.min() < 0) or (medium_map.max() >= params.materials):
        raise ValueError("medium_map needs material indices from 0 to " \
                         f"{params.materials - 1}")

    # The angular flux, cell edges, and artificial scattering sweep each
    # ordinate once
    ordered = (params.angular == True) or (params.flux_at_edges == 1) \
                or (params.sigma_as > 0.0)
    _check_reflection(quadrature, geometry, ordered)


def source_iteration_as(double[:,:,:,:] flux_guess, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:,:] external, \
        double[:,:,:,:,:] boundary_x, double[:,:,:,:,:] boundary_y, \
        double[:,:,:,:,:] boundary_z, int[:,:,:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] delta_z, \
        double[:] angle_x, double[:] angle_y, double[:] angle_z, \
        double[:] angle_w, parameters.params info):
    # Artificial scattering (as-SN), see source_iteration_as in fixed2d.
    # The kernel width uses all of the ordinates on the sphere, as in
    # Frank et al. (2020).
    cdef double[:,:] xs_total_as
    cdef double[:,:,:,:,:] source
    cdef double[:,:,:,:,:] psi

    # Artificial scatter matrix M_as[q,p] with rows summing to sigma_as
    M_as = artificial_scatter_matrix(np.asarray(angle_x), np.asarray(angle_y), \
                np.asarray(angle_w), info.sigma_as, info.beta_as, \
                angle_z=np.asarray(angle_z))

    # Add artificial out-scattering to the total cross section
    xs_total_as = np.asarray(xs_total) + info.sigma_as

    # Buffers for the known-source sweep
    source = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            info.angles, info.groups)
    psi = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                         info.angles, info.groups)

    def solve(flux, combined):
        # Standard multigroup problem with the combined source
        return np.asarray(mg.source_iteration(flux, xs_total_as, xs_matrix, \
                    combined, boundary_x, boundary_y, boundary_z, medium_map, \
                    delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, \
                    angle_w, info))

    def angular(flux, combined):
        # Recover the angular flux from the converged scalar flux
        tools._source_total(source, flux, xs_matrix, medium_map, combined, info)
        mg._known_source_angular(psi, xs_total_as, source, boundary_x, \
                    boundary_y, boundary_z, medium_map, delta_x, delta_y, \
                    delta_z, angle_x, angle_y, angle_z, info)
        return np.asarray(psi)

    def change(flux, flux_old):
        # Difference in scalar flux
        return tools.group_convergence(flux, flux_old, info)

    # The in-scatter source M_as . psi is lagged one iteration
    flux, external_combined = artificial_scatter_iteration( \
                np.asarray(flux_guess), external, M_as, solve, angular, change, \
                info.max_iter_angular, info.tol_energy)
    return flux, np.asarray(xs_total_as), external_combined


def known_flux(double[:,:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:,:] external, \
        double[:,:,:,:,:] boundary_x, double[:,:,:,:,:] boundary_y, \
        double[:,:,:,:,:] boundary_z, geometry, quadrature, params, \
        int[:,:] q_index=None):
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:,:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] delta_z = geometry.delta_z
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_z = quadrature.angle_z
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)
    check_steady_boundary(boundary_x, (info.cells_y, info.cells_z), \
                          info.angles, info.groups, "boundary_x")
    check_steady_boundary(boundary_y, (info.cells_x, info.cells_z), \
                          info.angles, info.groups, "boundary_y")
    check_steady_boundary(boundary_z, (info.cells_x, info.cells_y), \
                          info.angles, info.groups, "boundary_z")

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info, \
                        q_index)

    # Solve for angular flux at cell centers
    if info.flux_at_edges == 0:
        angular_flux = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                                      info.angles, info.groups)
        mg._known_source_angular(angular_flux, xs_total, source, boundary_x, \
                                 boundary_y, boundary_z, medium_map, delta_x, \
                                 delta_y, delta_z, angle_x, angle_y, angle_z, info)
        return np.asarray(angular_flux)

    # Solve for angular or scalar flux at cell interfaces
    xdim = info.angles if info.angular else 1
    flux_edge_x = tools.array_5d(info.cells_x + 1, info.cells_y, info.cells_z, \
                                 xdim, info.groups)
    flux_edge_y = tools.array_5d(info.cells_x, info.cells_y + 1, info.cells_z, \
                                 xdim, info.groups)
    flux_edge_z = tools.array_5d(info.cells_x, info.cells_y, info.cells_z + 1, \
                                 xdim, info.groups)
    mg._interface_flux(flux_edge_x, flux_edge_y, flux_edge_z, xs_total, source, \
                       boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                       delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, info)
    if info.angular:
        return np.asarray(flux_edge_x), np.asarray(flux_edge_y), \
                np.asarray(flux_edge_z)
    return np.asarray(flux_edge_x[...,0,:]), np.asarray(flux_edge_y[...,0,:]), \
            np.asarray(flux_edge_z[...,0,:])


def fixed_source_aniso(materials, sources, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_harmonics(materials.scatter, \
                                                         full_sphere=True)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:,:,:] external
    cdef double[:,:,:,:,:] boundary_x
    cdef double[:,:,:,:,:] boundary_y
    cdef double[:,:,:,:,:] boundary_z
    cdef int[:,:] q_index
    cdef int[:,:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] delta_z = geometry.delta_z
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_z = quadrature.angle_z
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed3d_source_iteration(info, xs_total.shape[0])
    _check_inputs(geometry, quadrature, params)

    # Compact external sources are read through the slab index. The
    # artificial in-scatter source is dense, so as-SN uses a dense source.
    if info.sigma_as > 0.0:
        external_np, q_index_np = dense_source(sources.external), None
    else:
        external_np, q_index_np = steady_external(sources.external, \
                (info.cells_x, info.cells_y, info.cells_z), info.angles, \
                info.groups)
    if q_index_np is None:
        check_steady_external(external_np, (info.cells_x, info.cells_y, \
                              info.cells_z), info.angles, info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x, (info.cells_y, info.cells_z), \
                    info.angles, info.groups, "boundary_x")
    boundary_y = steady_boundary(sources.boundary_y, (info.cells_x, info.cells_z), \
                    info.angles, info.groups, "boundary_y")
    boundary_z = steady_boundary(sources.boundary_z, (info.cells_x, info.cells_y), \
                    info.angles, info.groups, "boundary_z")

    # Spherical harmonics for the scattering source and flux moments
    P_source_np, P_weights_np = harmonic_data(quadrature, materials.scatter.shape[3])
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Add fission matrix to scattering
    xs_matrix = tools.array_4d(info.materials, info.groups, info.groups, info.moments)
    tools._xs_matrix_aniso(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.flux_at_edges = 0

    # Initialize flux moments to zeros
    flux_old = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                              info.groups, info.moments)

    # Effective total cross section / external source returned by the as-SN
    # solver (plain arrays when artificial scattering is disabled)
    xs_total_eff = xs_total
    external_eff = external

    # Multigroup solver
    if info.sigma_as > 0.0:
        # Artificial scattering (as-SN) ray effect mitigation
        flux, xs_total_eff, external_eff = source_iteration_as_aniso(flux_old, \
                    xs_total, xs_matrix, external, boundary_x, boundary_y, \
                    boundary_z, medium_map, delta_x, delta_y, delta_z, angle_x, \
                    angle_y, angle_z, angle_w, P_source, P_weights, info)
    else:
        flux = mg.multi_group_moments(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                    delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, \
                    P_source, P_weights, info, q_index)

    # Return scalar flux cell centers
    if (info.angular == False) and (params.flux_at_edges == 0):
        return np.ascontiguousarray(np.asarray(flux)[...,0])

    # For angular flux or scalar flux edges, see fixed_source
    return known_flux_moments(flux, xs_total_eff, xs_matrix, external_eff, \
                boundary_x, boundary_y, boundary_z, P_source, geometry, \
                quadrature, params, q_index)


def source_iteration_as_aniso(double[:,:,:,:,:] flux_guess, \
        double[:,:] xs_total, double[:,:,:,:] xs_matrix, \
        double[:,:,:,:,:] external, double[:,:,:,:,:] boundary_x, \
        double[:,:,:,:,:] boundary_y, double[:,:,:,:,:] boundary_z, \
        int[:,:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] delta_z, double[:] angle_x, double[:] angle_y, \
        double[:] angle_z, double[:] angle_w, double[:,:] P_source, \
        double[:,:] P_weights, parameters.params info):
    # Artificial scattering (as-SN) with flux moments, see source_iteration_as
    cdef double[:,:] xs_total_as
    cdef double[:,:,:,:,:] source
    cdef double[:,:,:,:,:] psi

    # Artificial scatter matrix M_as[q,p] with rows summing to sigma_as
    M_as = artificial_scatter_matrix(np.asarray(angle_x), np.asarray(angle_y), \
                np.asarray(angle_w), info.sigma_as, info.beta_as, \
                angle_z=np.asarray(angle_z))

    # Add artificial out-scattering to the total cross section
    xs_total_as = np.asarray(xs_total) + info.sigma_as

    # Buffers for the known-source sweep
    source = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            info.angles, info.groups)
    psi = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                         info.angles, info.groups)

    def solve(flux, combined):
        # Standard multigroup problem with the combined source
        return np.asarray(mg.multi_group_moments(flux, xs_total_as, xs_matrix, \
                    combined, boundary_x, boundary_y, boundary_z, medium_map, \
                    delta_x, delta_y, delta_z, angle_x, angle_y, angle_z, \
                    angle_w, P_source, P_weights, info))

    def angular(flux, combined):
        # Recover the angular flux from the converged flux moments
        tools._source_total_aniso(source, flux, xs_matrix, medium_map, \
                                  combined, P_source, info)
        mg._known_source_angular(psi, xs_total_as, source, boundary_x, \
                    boundary_y, boundary_z, medium_map, delta_x, delta_y, \
                    delta_z, angle_x, angle_y, angle_z, info)
        return np.asarray(psi)

    def change(flux, flux_old):
        # Difference in scalar flux
        return tools.group_convergence(flux[...,0], flux_old[...,0], info)

    # The in-scatter source M_as . psi is lagged one iteration
    flux, external_combined = artificial_scatter_iteration( \
                np.asarray(flux_guess), external, M_as, solve, angular, change, \
                info.max_iter_angular, info.tol_energy)
    return flux, np.asarray(xs_total_as), external_combined


def known_flux_moments(double[:,:,:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:,:] xs_matrix, double[:,:,:,:,:] external, \
        double[:,:,:,:,:] boundary_x, double[:,:,:,:,:] boundary_y, \
        double[:,:,:,:,:] boundary_z, double[:,:] P_source, geometry, \
        quadrature, params, int[:,:] q_index=None):
    # P_source is the (2l + 1) Y_k array from harmonic_data
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:,:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] delta_z = geometry.delta_z
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_z = quadrature.angle_z
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)
    check_steady_boundary(boundary_x, (info.cells_y, info.cells_z), \
                          info.angles, info.groups, "boundary_x")
    check_steady_boundary(boundary_y, (info.cells_x, info.cells_z), \
                          info.angles, info.groups, "boundary_y")
    check_steady_boundary(boundary_z, (info.cells_x, info.cells_y), \
                          info.angles, info.groups, "boundary_z")

    # Create sum_k (2l + 1) Y_k (sigma_s + sigma_f)_l * phi_k + external
    source = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                            info.angles, info.groups)
    tools._source_total_aniso(source, flux, xs_matrix, medium_map, external, \
                              P_source, info, q_index)

    # Solve for angular flux at cell centers
    if info.flux_at_edges == 0:
        angular_flux = tools.array_5d(info.cells_x, info.cells_y, info.cells_z, \
                                      info.angles, info.groups)
        mg._known_source_angular(angular_flux, xs_total, source, boundary_x, \
                                 boundary_y, boundary_z, medium_map, delta_x, \
                                 delta_y, delta_z, angle_x, angle_y, angle_z, info)
        return np.asarray(angular_flux)

    # Solve for angular or scalar flux at cell interfaces
    xdim = info.angles if info.angular else 1
    flux_edge_x = tools.array_5d(info.cells_x + 1, info.cells_y, info.cells_z, \
                                 xdim, info.groups)
    flux_edge_y = tools.array_5d(info.cells_x, info.cells_y + 1, info.cells_z, \
                                 xdim, info.groups)
    flux_edge_z = tools.array_5d(info.cells_x, info.cells_y, info.cells_z + 1, \
                                 xdim, info.groups)
    mg._interface_flux(flux_edge_x, flux_edge_y, flux_edge_z, xs_total, source, \
                       boundary_x, boundary_y, boundary_z, medium_map, delta_x, \
                       delta_y, delta_z, angle_x, angle_y, angle_z, angle_w, info)
    if info.angular:
        return np.asarray(flux_edge_x), np.asarray(flux_edge_y), \
                np.asarray(flux_edge_z)
    return np.asarray(flux_edge_x[...,0,:]), np.asarray(flux_edge_y[...,0,:]), \
            np.asarray(flux_edge_z[...,0,:])
