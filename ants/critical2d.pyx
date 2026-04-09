########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two-Dimensional Criticality Multigroup Neutron Transport Problems
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

import logging

import numpy as np

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import create_params

logger = logging.getLogger(__name__)


def k_criticality(materials, geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Convert dictionary to type params1d
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_critical2d_power_iteration(info)

    # Initialize keff
    cdef double[1] keff = [0.95]

    # Initialize and normalize flux
    flux_old = np.random.rand(info.cells_x, info.cells_y, info.groups)
    tools._normalize_flux(flux_old, info)

    # Solve using the power iteration
    flux = power_iteration(flux_old, xs_total, xs_scatter, xs_fission, medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info, keff)

    # Return scalar flux at cell centers and keff
    if (info.angular == False) and (info.flux_at_edges == 0):
        return np.asarray(flux), keff[0]

    # For returning angular flux or flux at cell edges
    return known_flux(flux, keff[0], materials, geometry, quadrature, params)


cdef double[:,:,:] power_iteration(double[:,:,:]& flux_guess, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params info, double[:]& keff):

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()

    # Initialize power source
    source = tools.array_4d(info.cells_x, info.cells_y, 1, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_4d(2, 1, 1, 1)
    boundary_y = tools.array_4d(2, 1, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until convergence
    while not (converged):

        # Update power source term
        tools._fission_source(flux_old, xs_fission, source, medium_map, \
                              info, keff[0])

        # Solve for scalar flux
        flux = mg.multi_group(flux_old, xs_total, xs_scatter, source, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)

        # Calculate k-effective
        keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
                                           medium_map, info, keff[0])
        tools._normalize_flux(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        logger.info("Count: %s\tKeff: %.8f", str(count).zfill(3), keff[0])
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1

        # Update old flux
        flux_old[:,:,:] = flux[:,:,:]

    logger.info("Convergence: %2.6e", change)
    return flux[:,:,:]


def known_flux(double[:,:,:] flux, keff,  materials, geometry, quadrature, params):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert dictionary to type params
    info = parameters._to_params(params)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, 1, info.groups)
    tools._source_total_critical(source, flux, xs_scatter, xs_fission, \
                                 medium_map, keff, info)

    # Vacuum boundaries
    boundary_x = tools.array_4d(2, 1, 1, 1)
    boundary_y = tools.array_4d(2, 1, 1, 1)

    # Return scalar flux cell edges
    if (info.angular == False) and (info.flux_at_edges == 1):
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


def nearby_power_iteration(double[:,:,:,:] residual, double n_rate, materials, \
        geometry, quadrature, solver):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = materials.fission
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Covert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_critical2d_nearby_power(info)

    # Initialize flux
    flux_old = np.random.rand(info.cells_x, info.cells_y, info.groups)
    tools._normalize_flux(flux_old, info)

    # Initialize keffective
    cdef double[1] keff

    # Initialize half step keffective for nearby problems
    keff[0] = tools._nearby_keffective(flux_old, n_rate, info)

    # Solve using the modified power iteration
    flux = multi_group_nearby(flux_old, keff, residual, xs_total, xs_scatter, \
                            xs_fission, medium_map, delta_x, delta_y, angle_x, \
                            angle_y, angle_w, info)

    return np.asarray(flux), keff[0]


cdef double[:,:,:] multi_group_nearby(double[:,:,:]& flux_guess, \
        double[:]& keff, double[:,:,:,:]& residual, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission,  int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):

    # Initialize flux
    flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    flux_old = flux_guess.copy()

    # Initialize power source
    fission_source = tools.array_4d(info.cells_x, info.cells_y, 1, info.groups)

    # Vacuum boundaries
    boundary_x = tools.array_4d(2, 1, 1, 1)
    boundary_y = tools.array_4d(2, 1, 1, 1)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate until converged
    while not (converged):
        # Update nearby power source term
        tools._nearby_fission_source(flux_old, xs_fission, fission_source, \
                                     residual, medium_map, info, keff[0])

        # Solve for scalar flux
        flux = mg.multi_group(flux_old, xs_total, xs_scatter, fission_source, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)

        # Update keffective
        keff[0] = tools._update_keffective(flux, flux_old, xs_fission, \
                                           medium_map, info, keff[0])

        # Normalize flux
        tools._normalize_flux(flux, info)

        # Check for convergence
        change = tools.group_convergence(flux, flux_old, info)
        logger.info(f"Count: {str(count).zfill(3)}\tKeff: {keff[0]:.8f}")
        converged = (change < info.tol_keff) or (count >= info.max_iter_keff)
        count += 1

        # Update old flux
        flux_old[:,:,:] = flux[:,:,:]

    logger.info(f"Convergence: {change:.6e}")
    return flux[:,:,:]
