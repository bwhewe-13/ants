########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One-Dimensional Criticality Multigroup Neutron Transport Problems
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

import logging

import numpy as np

from ants cimport cytools_1d as tools
from ants cimport multi_group_1d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import create_params
from ants.quadrature import legendre_data
from ants.utils.pytools import check_keffective, warn_not_converged

logger = logging.getLogger(__name__)


def k_criticality(materials, geometry, quadrature, solver):
    # Anisotropic scattering
    if materials.scatter.ndim == 4:
        return k_criticality_aniso(materials, geometry, quadrature, solver)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_critical1d_power_iteration(info)

    # Initialize keff
    cdef double[1] keff = [0.95]

    # Initialize and normalize flux (fixed seed for reproducible iterations)
    flux_old = np.random.default_rng(42).random((info.cells_x, info.groups))
    tools._normalize_flux(flux_old, info)

    # Solve using the power iteration
    flux = power_iteration(flux_old, keff, xs_total, xs_scatter, xs_fission, \
                        medium_map, delta_x, angle_x, angle_w, info)

    # Return scalar flux at cell centers and keff
    if (info.angular == False) and (info.flux_at_edges == 0):
        return np.asarray(flux), keff[0]

    # For returning angular flux or flux at cell edges
    return known_flux(flux, keff[0], materials, geometry, quadrature, params)


cdef double[:,:] power_iteration(double[:,:]& flux_guess,double[:]& keff,\
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):

    # Initialize flux
    flux = tools.array_2d(info.cells_x, info.groups)
    flux_old = flux_guess.copy()

    # Initialize power source
    source = tools.array_3d(info.cells_x, 1, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_3d(2, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until converge
    while not (converged):
        # Update power source term
        tools._fission_source(flux_old, xs_fission, source, medium_map, info, keff[0])

        # Solve for scalar flux
        flux = mg.multi_group(flux_old, xs_total, xs_scatter, source, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)

        # Update keffective
        keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
                                           medium_map, info, keff[0])
        check_keffective(keff[0], count)

        # Normalize flux
        tools._normalize_flux(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        logger.info(f"Count: {str(count).zfill(3)}\tKeff: {keff[0]:.8f}")
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1
        flux_old[:,:] = flux[:,:]

    if change >= info.tol_keff:
        warn_not_converged("Power", "max_iter_keff")
    logger.info(f"Convergence: {change:.6e}")
    return flux[:,:]


def known_flux(double[:,:] flux, keff, materials, geometry, quadrature, params):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_3d(info.cells_x, 1, info.groups)
    tools._source_total_critical(source, flux, xs_scatter, xs_fission, \
                                 medium_map, keff, info)

    # Need zero array for boundary
    boundary_x = tools.array_3d(2, 1, 1)

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


def nearby_power_iteration(double[:,:,:] residual, double n_rate, materials, \
        geometry, quadrature, solver):
    # Anisotropic scattering
    if materials.scatter.ndim == 4:
        return nearby_power_iteration_aniso(residual, n_rate, materials, \
                                            geometry, quadrature, solver)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_critical1d_nearby_power(info)

    # Initialize flux (fixed seed for reproducible iterations)
    flux_old = np.random.default_rng(42).random((info.cells_x, info.groups))

    # Initialize keffective
    cdef double[1] keff

    # Initialize half step keffective for nearby problems
    keff[0] = tools._nearby_keffective(flux_old, n_rate, info)

    # Solve using the modified power iteration
    flux = multi_group_nearby(flux_old, keff, residual, xs_total, xs_scatter, \
                        xs_fission, medium_map, delta_x, angle_x, angle_w, info)

    return np.asarray(flux), keff[0]


cdef double[:,:] multi_group_nearby(double[:,:]& flux_guess, double[:]& keff, \
        double[:,:,:]& residual, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):

    # Initialize flux
    flux = tools.array_2d(info.cells_x, info.groups)
    flux_old = flux_guess.copy()

    # Initialize power source
    source = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_3d(2, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until converged
    while not (converged):
        # Update nearby power source term
        tools._nearby_fission_source(flux_old, xs_fission, source, \
                                     residual, medium_map, info, keff[0])

        # Solve for scalar flux
        flux = mg.multi_group(flux_old, xs_total, xs_scatter, source, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, info)

        # Update keffective
        keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
                                           medium_map, info, keff[0])
        check_keffective(keff[0], count)

        # Normalize flux
        tools._normalize_flux(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        logger.info(f"Count: {str(count).zfill(3)}\tKeff: {keff[0]:.8f}")
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1
        flux_old[:,:] = flux[:,:]

    if change >= info.tol_keff:
        warn_not_converged("Power", "max_iter_keff")
    logger.info(f"Convergence: {change:.6e}")
    return flux[:,:]


################################################################################
# Anisotropic Scattering (Legendre moments)
################################################################################

def k_criticality_aniso(materials, geometry, quadrature, solver):
    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)

    # Solve for the flux moments and keff
    flux, keff = flux_moments(materials, geometry, quadrature, params)

    # Return scalar flux at cell centers and keff
    if (params.angular == False) and (params.flux_at_edges == 0):
        return np.ascontiguousarray(flux[:,:,0]), keff

    # For returning angular flux or flux at cell edges
    return known_flux_moments(flux, keff, materials, geometry, quadrature, params)


def flux_moments(materials, geometry, quadrature, params):
    # Flux moments (cells_x, groups, moments) and keff for anisotropic scattering
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)
    parameters._check_critical1d_power_iteration(info)

    # Legendre polynomials for the scattering source and flux moments
    P_source_np, P_weights_np = legendre_data(quadrature, info.moments)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Initialize keff
    cdef double[1] keff = [0.95]

    # Initialize and normalize flux (fixed seed for reproducible iterations)
    flux_old = np.zeros((info.cells_x, info.groups, info.moments))
    flux_old[:,:,0] = np.random.default_rng(42).random((info.cells_x, info.groups))
    tools._normalize_moments(flux_old, info)

    # Solve using the power iteration
    flux = power_iteration_aniso(flux_old, keff, xs_total, xs_scatter, \
                xs_fission, medium_map, delta_x, angle_x, angle_w, P_source, \
                P_weights, info)

    return np.asarray(flux), keff[0]


cdef double[:,:,:] power_iteration_aniso(double[:,:,:]& flux_guess, \
        double[:]& keff, double[:,:]& xs_total, double[:,:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, double[:,:]& P_source, \
        double[:,:]& P_weights, params info):

    # Initialize flux moments
    flux = tools.array_3d(info.cells_x, info.groups, info.moments)
    flux_old = flux_guess.copy()

    # Initialize power source (fission is isotropic)
    source = tools.array_3d(info.cells_x, 1, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_3d(2, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until converge
    while not (converged):
        # Update power source term
        tools._fission_source(flux_old[:,:,0], xs_fission, source, medium_map, \
                              info, keff[0])

        # Solve for flux moments
        flux = mg.multi_group_moments(flux_old, xs_total, xs_scatter, source, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, \
                    P_source, P_weights, info)

        # Update keffective
        keff[0] = tools._update_keffective(flux[:,:,0], flux_old[:,:,0], \
                                           xs_fission, medium_map, info, keff[0])
        check_keffective(keff[0], count)

        # Normalize flux
        tools._normalize_moments(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux[:,:,0], flux_old[:,:,0], info)
        logger.info(f"Count: {str(count).zfill(3)}\tKeff: {keff[0]:.8f}")
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]

    if change >= info.tol_keff:
        warn_not_converged("Power", "max_iter_keff")
    logger.info(f"Convergence: {change:.6e}")
    return flux[:,:,:]


def known_flux_moments(double[:,:,:] flux, keff, materials, geometry, \
        quadrature, params):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Legendre polynomials for the scattering source
    P_source_np, _ = legendre_data(quadrature, info.moments)
    cdef double[:,:] P_source = P_source_np

    # Create sum_l (2l + 1) P_l sigma_s,l * phi_l + sigma_f * phi / keff
    source = tools.array_3d(info.cells_x, info.angles, info.groups)
    tools._source_total_critical_aniso(source, flux, xs_scatter, xs_fission, \
                                       medium_map, keff, P_source, info)

    # Need zero array for boundary
    boundary_x = tools.array_3d(2, 1, 1)

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


def nearby_power_iteration_aniso(double[:,:,:] residual, double n_rate, \
        materials, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_critical1d_nearby_power(info)

    # Legendre polynomials for the scattering source and flux moments
    P_source_np, P_weights_np = legendre_data(quadrature, info.moments)
    cdef double[:,:] P_source = P_source_np
    cdef double[:,:] P_weights = P_weights_np

    # Initialize flux (fixed seed for reproducible iterations)
    flux_old = np.zeros((info.cells_x, info.groups, info.moments))
    flux_old[:,:,0] = np.random.default_rng(42).random((info.cells_x, info.groups))

    # Initialize keffective
    cdef double[1] keff

    # Initialize half step keffective for nearby problems
    keff[0] = tools._nearby_keffective(flux_old[:,:,0], n_rate, info)

    # Solve using the modified power iteration
    flux = multi_group_nearby_aniso(flux_old, keff, residual, xs_total, \
                xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w, \
                P_source, P_weights, info)

    return np.ascontiguousarray(np.asarray(flux)[:,:,0]), keff[0]


cdef double[:,:,:] multi_group_nearby_aniso(double[:,:,:]& flux_guess, \
        double[:]& keff, double[:,:,:]& residual, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, double[:,:]& P_source, double[:,:]& P_weights, \
        params info):

    # Initialize flux moments
    flux = tools.array_3d(info.cells_x, info.groups, info.moments)
    flux_old = flux_guess.copy()

    # Initialize power source
    source = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_3d(2, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until converged
    while not (converged):
        # Update nearby power source term
        tools._nearby_fission_source(flux_old[:,:,0], xs_fission, source, \
                                     residual, medium_map, info, keff[0])

        # Solve for flux moments
        flux = mg.multi_group_moments(flux_old, xs_total, xs_scatter, source, \
                    boundary_x, medium_map, delta_x, angle_x, angle_w, \
                    P_source, P_weights, info)

        # Update keffective
        keff[0] = tools._update_keffective(flux[:,:,0], flux_old[:,:,0], \
                                           xs_fission, medium_map, info, keff[0])
        check_keffective(keff[0], count)

        # Normalize flux
        tools._normalize_moments(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux[:,:,0], flux_old[:,:,0], info)
        logger.info(f"Count: {str(count).zfill(3)}\tKeff: {keff[0]:.8f}")
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1
        flux_old[:,:,:] = flux[:,:,:]

    if change >= info.tol_keff:
        warn_not_converged("Power", "max_iter_keff")
    logger.info(f"Convergence: {change:.6e}")
    return flux[:,:,:]
