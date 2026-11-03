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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters

from ants.datatypes import MultigroupSolver, create_params
from ants.quadrature import _scatter_harmonics, artificial_scatter_matrix, harmonic_data
from ants.utils.sources import dense_source, steady_boundary, steady_external


def fixed_source(materials, sources, geometry, quadrature, solver):
    # Anisotropic scattering
    if materials.scatter.ndim == 4:
        return fixed_source_aniso(materials, sources, geometry, quadrature, solver)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:,:] external
    cdef double[:,:,:,:] boundary_x
    cdef double[:,:,:,:] boundary_y
    cdef int[:,:] q_index
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed2d_source_iteration(info, xs_total.shape[0])

    # Compact external sources are read through the slab index. The
    # artificial in-scatter source is dense, so as-SN uses a dense source.
    if info.sigma_as > 0.0:
        external_np, q_index_np = dense_source(sources.external), None
    else:
        external_np, q_index_np = steady_external(sources.external, \
                (info.cells_x, info.cells_y), info.angles * info.angles, \
                info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x)
    boundary_y = steady_boundary(sources.boundary_y)

    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Solve for cell center first
    info.flux_at_edges = 0

    # Initialize flux_old to zeros
    flux_old = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Effective total cross section / external source returned by the as-SN
    # solver (plain arrays when artificial scattering is disabled)
    xs_total_eff = xs_total
    external_eff = external

    # Multigroup solver
    if info.sigma_as > 0.0:
        # Artificial scattering (as-SN) ray effect mitigation
        flux, xs_total_eff, external_eff = source_iteration_as(flux_old, \
                    xs_total, xs_matrix, external, boundary_x, boundary_y, \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)
    elif params.mg_solver == MultigroupSolver.SOURCE_ITERATION:
        flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, info, q_index)
    elif params.mg_solver == MultigroupSolver.DMD:
        flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, angle_x, \
                    angle_y, angle_w, info, q_index)

    # Return scalar flux cell centers
    if (info.angular == False) and (params.flux_at_edges == 0):
        return np.asarray(flux)

    # For angular flux or scalar flux edges. For as-SN runs, the effective
    # total cross section (sigma_t + sigma_as) and the combined source
    # (external + artificial in-scatter) reproduce the converged transport
    # balance in the known-source sweeps.
    return known_flux(flux, xs_total_eff, xs_matrix, external_eff, boundary_x, \
                    boundary_y, geometry, quadrature, params, q_index)


def source_iteration_as(double[:,:,:] flux_guess, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        parameters.params info):
    """Solve a 2D fixed-source problem with artificial scattering (as-SN).

    Implements Eq. (17) of Frank, Kusch, Camminady & Hauck (2020), "Ray
    Effect Mitigation for the Discrete Ordinates Method Using Artificial
    Scattering", Nucl. Sci. Eng.:

        Omega_q . grad(psi_q) + (sigma_t + sigma_as) psi_q
            = sigma_s Phi + sum_p M_as[q,p] psi_p + q_ext

    The artificial out-scattering sigma_as is folded into the total cross
    section (stabilizing the iteration, per the paper), and the
    forward-peaked in-scatter source M_as . psi is lagged one outer
    iteration (the source-iteration treatment of Eq. (25)).  Each outer
    pass reuses the standard multigroup solver for the physical scattering
    and a known-source sweep to recover the angular flux.

    Returns (scalar_flux, xs_total_as, external_combined) so callers can
    reproduce the converged transport balance in known-source sweeps.
    """
    cdef int as_iter
    cdef double as_change
    cdef double[:,:,:] flux = flux_guess.copy()
    cdef double[:,:,:] flux_old = flux_guess.copy()
    cdef double[:,:,:,:] ext_combined
    cdef double[:,:,:,:] psi
    cdef double[:,:] xs_total_as_view
    cdef int N_angles = info.angles * info.angles

    # Compute artificial scatter matrix M_as[q,p] once. Rows sum to
    # sigma_as (particle conservation), balancing the sigma_as psi_q
    # out-scattering term added to the total cross section below.
    M_as = artificial_scatter_matrix(np.asarray(angle_x), np.asarray(angle_y), \
                np.asarray(angle_w), info.sigma_as, info.beta_as)

    # Add artificial out-scattering to the total cross section (Eq. 17 LHS)
    xs_total_as = np.asarray(xs_total) + info.sigma_as
    xs_total_as_view = xs_total_as

    # Artificial in-scatter source (I x J x N^2 x G), lagged one iteration
    art_source = np.zeros((info.cells_x, info.cells_y, N_angles, info.groups))
    external_combined = np.asarray(external) + art_source

    # Buffer for the known-source sweep: (sigma_s + sigma_f) phi + external
    source = tools.array_4d(info.cells_x, info.cells_y, N_angles, info.groups)

    # Iterate on the artificial scatter source
    for as_iter in range(info.max_iter_angular):
        ext_combined = external_combined

        # Solve standard multigroup problem with combined source
        flux_old[:,:,:] = flux[:,:,:]
        flux = mg.source_iteration(flux, xs_total_as_view, xs_matrix, \
                            ext_combined, boundary_x, boundary_y, medium_map, \
                            delta_x, delta_y, angle_x, angle_y, angle_w, info)

        # Recover the angular flux from the converged scalar flux
        tools._source_total(source, flux, xs_matrix, medium_map, \
                            ext_combined, info)
        psi = mg._known_source_angular(xs_total_as_view, source, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info)

        # Update artificial in-scatter source:
        # art_source[i,j,q,g] = sum_p M_as[q,p] psi[i,j,p,g]
        art_source = np.einsum("qp,ijpg->ijqg", M_as, np.asarray(psi))
        external_combined = np.asarray(external) + art_source

        # Check convergence: difference in scalar flux
        as_change = tools.group_convergence(flux, flux_old, info)
        if as_change < info.tol_angular:
            break

    return np.asarray(flux), xs_total_as, external_combined


def known_flux(double[:,:,:] flux, double[:,:] xs_total, double[:,:,:] xs_matrix, \
        double[:,:,:,:] external, double[:,:,:,:] boundary_x, \
        double[:,:,:,:] boundary_y, geometry, quadrature, params, \
        int[:,:] q_index=None):
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info, \
                        q_index)

    # Solve for angular flux at cell centers
    if (info.angular == True) and (info.flux_at_edges == 0):
        angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, info)
        return np.asarray(angular_flux)

    # Solve for angular flux cell interfaces
    elif (info.angular == True) and (info.flux_at_edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                                     info.angles * info.angles, info.groups)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                                     info.angles * info.angles, info.groups)
        mg._interface_angular(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x), np.asarray(flux_edge_y)

    # Solve for scalar flux cell interfaces
    elif (info.angular == False) and (info.flux_at_edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, info.groups, 1)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, info.groups, 1)
        mg._interface_scalar(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x[...,0]), np.asarray(flux_edge_y[...,0])

    return -1


def known_flux_single(int group, double[:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:] external, double[:,:,:,:] boundary_x, \
        double[:,:,:,:] boundary_y, geometry, quadrature, params):
    # This is for solving for angular flux or cell interfaces
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, external.shape[2], 1)
    if external.shape[2] == 1:
        tools._source_total_single(source, flux, xs_matrix, medium_map, \
                                    external, group, info)
    else:
        tools._source_total_nsingle(source, flux, xs_matrix, medium_map, \
                                    external, group, info)

    # Solve for angular flux at cell centers
    angular_flux = mg._known_source_single(xs_total, source, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, group, info)
    return np.asarray(angular_flux)


def fixed_source_aniso(materials, sources, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = _scatter_harmonics(materials.scatter)
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:,:] external
    cdef double[:,:,:,:] boundary_x
    cdef double[:,:,:,:] boundary_y
    cdef int[:,:] q_index
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_fixed2d_source_iteration(info, xs_total.shape[0])

    # Compact external sources are read through the slab index. The
    # artificial in-scatter source is dense, so as-SN uses a dense source.
    if info.sigma_as > 0.0:
        external_np, q_index_np = dense_source(sources.external), None
    else:
        external_np, q_index_np = steady_external(sources.external, \
                (info.cells_x, info.cells_y), info.angles * info.angles, \
                info.groups)
    external = external_np
    q_index = q_index_np
    boundary_x = steady_boundary(sources.boundary_x)
    boundary_y = steady_boundary(sources.boundary_y)

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
    flux_old = tools.array_4d(info.cells_x, info.cells_y, info.groups, info.moments)

    # Effective total cross section / external source returned by the as-SN
    # solver (plain arrays when artificial scattering is disabled)
    xs_total_eff = xs_total
    external_eff = external

    # Multigroup solver
    if info.sigma_as > 0.0:
        # Artificial scattering (as-SN) ray effect mitigation
        flux, xs_total_eff, external_eff = source_iteration_as_aniso(flux_old, \
                    xs_total, xs_matrix, external, boundary_x, boundary_y, \
                    medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, \
                    P_source, P_weights, info)
    else:
        flux = mg.multi_group_moments(flux_old, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, P_source, P_weights, info, \
                    q_index)

    # Return scalar flux cell centers
    if (info.angular == False) and (params.flux_at_edges == 0):
        return np.ascontiguousarray(np.asarray(flux)[:,:,:,0])

    # For angular flux or scalar flux edges, see fixed_source
    return known_flux_moments(flux, xs_total_eff, xs_matrix, external_eff, \
                    boundary_x, boundary_y, P_source, geometry, quadrature, params, \
                    q_index)


def source_iteration_as_aniso(double[:,:,:,:] flux_guess, double[:,:] xs_total, \
        double[:,:,:,:] xs_matrix, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        double[:,:] P_source, double[:,:] P_weights, parameters.params info):
    # Artificial scattering (as-SN) with flux moments, see source_iteration_as
    cdef int as_iter
    cdef double as_change
    cdef double[:,:,:,:] flux = flux_guess.copy()
    cdef double[:,:,:,:] flux_old = flux_guess.copy()
    cdef double[:,:,:,:] ext_combined
    cdef double[:,:,:,:] psi
    cdef double[:,:] xs_total_as_view
    cdef int N_angles = info.angles * info.angles

    # Artificial scatter matrix M_as[q,p] with rows summing to sigma_as
    M_as = artificial_scatter_matrix(np.asarray(angle_x), np.asarray(angle_y), \
                np.asarray(angle_w), info.sigma_as, info.beta_as)

    # Add artificial out-scattering to the total cross section
    xs_total_as = np.asarray(xs_total) + info.sigma_as
    xs_total_as_view = xs_total_as

    # Artificial in-scatter source (I x J x N^2 x G), lagged one iteration
    art_source = np.zeros((info.cells_x, info.cells_y, N_angles, info.groups))
    external_combined = np.asarray(external) + art_source

    # Buffer for the known-source sweep
    source = tools.array_4d(info.cells_x, info.cells_y, N_angles, info.groups)

    # Iterate on the artificial scatter source
    for as_iter in range(info.max_iter_angular):
        ext_combined = external_combined

        # Solve standard multigroup problem with combined source
        flux_old[:,:,:,:] = flux[:,:,:,:]
        flux = mg.multi_group_moments(flux, xs_total_as_view, xs_matrix, \
                            ext_combined, boundary_x, boundary_y, medium_map, \
                            delta_x, delta_y, angle_x, angle_y, angle_w, \
                            P_source, P_weights, info)

        # Recover the angular flux from the converged flux moments
        tools._source_total_aniso(source, flux, xs_matrix, medium_map, \
                                  ext_combined, P_source, info)
        psi = mg._known_source_angular(xs_total_as_view, source, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info)

        # Update artificial in-scatter source
        art_source = np.einsum("qp,ijpg->ijqg", M_as, np.asarray(psi))
        external_combined = np.asarray(external) + art_source

        # Check convergence: difference in scalar flux
        as_change = tools.group_convergence(flux[:,:,:,0], flux_old[:,:,:,0], info)
        if as_change < info.tol_angular:
            break

    return np.asarray(flux), xs_total_as, external_combined


def known_flux_moments(double[:,:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:,:] xs_matrix, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        double[:,:] P_source, geometry, quadrature, params, \
        int[:,:] q_index=None):
    # P_source is the (2l + 1) Y_k array from harmonic_data
    # Unpack Python DataTypes to Cython memoryviews
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create sum_k (2l + 1) Y_k (sigma_s + sigma_f)_l * phi_k + external
    source = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)
    tools._source_total_aniso(source, flux, xs_matrix, medium_map, external, \
                              P_source, info, q_index)

    # Solve for angular flux at cell centers
    if (info.angular == True) and (info.flux_at_edges == 0):
        angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, info)
        return np.asarray(angular_flux)

    # Solve for angular flux cell interfaces
    elif (info.angular == True) and (info.flux_at_edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                                     info.angles * info.angles, info.groups)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                                     info.angles * info.angles, info.groups)
        mg._interface_angular(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x), np.asarray(flux_edge_y)

    # Solve for scalar flux cell interfaces
    elif (info.angular == False) and (info.flux_at_edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, info.groups, 1)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, info.groups, 1)
        mg._interface_scalar(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x[...,0]), np.asarray(flux_edge_y[...,0])

    return -1
