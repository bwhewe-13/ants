########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One-Dimensional Nearby Problems
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
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm

from ants import critical1d, fixed1d
from ants.utils.interp1d import Interpolation
from ants.utils.pytools import average_array

from ants cimport cytools_1d as tools
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import SourceData, create_params

logger = logging.getLogger(__name__)


def fixed_source(materials, sources, geometry, quadrature, solver, knots_x, **kwargs):
    quintic = kwargs.get("quintic", True)
    numerical_flux = kwargs.get("numerical_flux", None)
    scalar_residual = kwargs.get("scalar_residual", False)
    save_residual = kwargs.get("save_residual", True)
    run_nearby = kwargs.get("return_residual", False)

    # Convert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_nearby1d_fixed_source(info, materials.total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    x_splits = kwargs.get("x_splits", np.zeros((0,), dtype=np.int32))

    # Run Numerical Solution (angular flux required for curve fitting)
    logger.info("Calculating Numerical Solution...")
    numerical_flux = calculate_numerical_flux(numerical_flux, materials, sources, \
                                            geometry, quadrature, solver, info)

    # Calculate curve fit and residual
    logger.info("Calculating Curve Fit and Residual...")
    curve_fit_flux, curve_fit_boundary_x, residual = calculate_residual(numerical_flux, \
                    materials, sources, geometry, quadrature, x_splits, knots_x, \
                    info, block, quintic, scalar_residual, save_residual)

    if not run_nearby:
        return numerical_flux, curve_fit_flux, curve_fit_boundary_x, residual

    # Run Nearby Problem
    logger.info("Calculating Nearby Solution...")
    if kwargs.get("zero_bounds", False):
        logger.info("Removing Analytical Boundary Conditions...")
        nearby_boundary_x = sources.boundary_x.copy()
    else:
        nearby_boundary_x = np.asarray(curve_fit_boundary_x)
    nearby_sources = SourceData(np.asarray(sources.external) + np.asarray(residual), \
                                nearby_boundary_x)
    scalar_solver = replace(solver, angular=False, flux_at_edges=0)
    nearby_flux = fixed1d.fixed_source(materials, nearby_sources, geometry, \
                                       quadrature, scalar_solver)

    return numerical_flux, curve_fit_flux, curve_fit_boundary_x, residual, nearby_flux


def calculate_numerical_flux(numerical_flux, materials, sources, geometry, \
        quadrature, solver, info):
    """Calculate numerical flux for nearby fixed source problem."""
    # Unpack Python DataTypes to Cython memoryviews (used by ndim==2 path)
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    # Restore C struct from Python object that crosses the def boundary
    cdef params _info = info
    # Ensure solving for angular flux at cell centers (fixed1d solver requirement)
    angular_solver = replace(solver, angular=True, flux_at_edges=0)
    # No numerical flux provided, compute with fixed source solver (default)
    if numerical_flux is None:
        return fixed1d.fixed_source(materials, sources, geometry, \
                                            quadrature, angular_solver)
    # Scalar flux provided, compute angular flux with single transport sweep
    elif numerical_flux.ndim == 2:
        xs_matrix = tools.array_3d(_info.materials, _info.groups, _info.groups)
        tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, _info)
        angular_params = create_params(materials, quadrature, geometry, angular_solver)
        return fixed1d.known_flux(numerical_flux, xs_total, xs_matrix, sources.external, \
                            sources.boundary_x, geometry, quadrature, angular_params)
    # Angular flux provided, no change needed
    return numerical_flux


def calculate_residual(numerical_flux, materials, sources, geometry, quadrature, \
        x_splits, knots_x, info, block, quintic, scalar_residual, save_residual):
    """Calculate curve fit and residual for nearby fixed source problem."""
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:] external = sources.external
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Restore the C struct from the Python dict that crosses the def boundary
    cdef params _info = info

    # Initialize curve fit arrays
    if scalar_residual:
        # Initialize curve fit and residual (scalar: cells_x * groups)
        curve_fit_boundary_x = tools.array_2d(2, _info.groups)
        curve_fit_flux = tools.array_2d(_info.cells_x, _info.groups)
        residual = tools.array_2d(_info.cells_x, _info.groups)
        # Build cell-edge grid (scalar residual requires center knots)
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)

        logger.info("Calculating Scalar Curve Fit Solution and Residual...")
        _scalar_curve_fit_residual(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, residual, xs_total, xs_scatter, \
                xs_fission, external, medium_map, delta_x, knots_x, edges_x, \
                x_splits, angle_x, angle_w, block, quintic, _info)

    else:
        curve_fit_boundary_x = tools.array_3d(2, _info.angles, _info.groups)
        curve_fit_flux = tools.array_2d(_info.cells_x, _info.groups)
        residual = tools.array_3d(_info.cells_x, _info.angles, _info.groups)

        # Initialize curve fit integrals
        int_angular = tools.array_3d(_info.cells_x, _info.angles, _info.groups)
        int_dx_angular = tools.array_3d(_info.cells_x, _info.angles, _info.groups)
        int_scalar = tools.array_2d(_info.cells_x, _info.groups)

        # Calculate curve fit (dispatches to center or edge knots internally)
        logger.info("Calculating Angular Curve Fit Solution...")
        _angular_curve_fit(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                int_angular, int_dx_angular, int_scalar, medium_map, x_splits, \
                knots_x, delta_x, angle_w, block, quintic, _info)

        # Calculate residual for each cell
        logger.info("Calculating Angular Residual...")
        _angular_residual(residual, int_angular, int_dx_angular, int_scalar, \
                xs_total, xs_scatter, xs_fission, external, medium_map, \
                delta_x, angle_x, _info)

    curve_fit_flux = np.asarray(curve_fit_flux)
    residual = np.asarray(residual)
    curve_fit_boundary_x = np.asarray(curve_fit_boundary_x)

    if save_residual:
        fangles = str(_info.angles).zfill(2)
        fcells = str(_info.cells_x).zfill(3)
        np.save(f"curve_fit_flux_x{fcells}_n{fangles}.npy", curve_fit_flux)
        np.save(f"nearby_residual_x{fcells}_n{fangles}.npy", residual)
        np.save(f"nearby_boundary_x_x{fcells}_n{fangles}.npy", curve_fit_boundary_x)
    return curve_fit_flux, curve_fit_boundary_x, residual


def k_criticality(materials, geometry, quadrature, solver, knots_x, **kwargs):
    quintic = kwargs.get("quintic", True)
    numerical_flux = kwargs.get("numerical_flux", None)
    numerical_keff = kwargs.get("numerical_keff", None)
    save_residual = kwargs.get("save_residual", True)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_w = quadrature.angle_w

    # Convert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_nearby1d_criticality(info)
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    x_splits = kwargs.get("x_splits", np.zeros((0,), dtype=np.int32))

    # Run Numerical Solution (angular flux required for curve fitting)
    logger.info("Calculating Numerical Solution...")
    numerical_flux, numerical_keff = calculate_k_numerical_flux(numerical_flux, \
                                                    numerical_keff, materials, \
                                                    geometry, quadrature, solver)
    # Initialize curve fit arrays
    curve_fit_boundary_x = tools.array_3d(2, info.angles, info.groups)
    curve_fit_scalar = tools.array_2d(info.cells_x, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_dx_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_scalar = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit (dispatches to center or edge knots internally)
    logger.info("Calculating Analytical Solution...")
    _angular_curve_fit(numerical_flux, curve_fit_scalar, curve_fit_boundary_x, \
            int_angular, int_dx_angular, int_scalar, medium_map, x_splits, \
            knots_x, delta_x, angle_w, block, quintic, info)

    # Create curve fit fission source, curve fit keff, and nearby reaction rate
    curve_fit_source = tools.array_2d(info.cells_x, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(int_angular, \
                            int_dx_angular, int_scalar, xs_total, xs_scatter, \
                            xs_fission, curve_fit_source, medium_map, delta_x, \
                            angle_x, angle_w, info)

    # Calculate residual for each cell
    logger.info("Calculating Residual...")
    residual = np.zeros((info.cells_x, info.angles, info.groups))
    _angular_residual_critical(residual, int_angular, int_dx_angular, \
            int_scalar, xs_total, xs_scatter, curve_fit_source, medium_map, \
            angle_x, curve_fit_keff, info)
    if save_residual:
        fangles = str(info.angles).zfill(2)
        fcells = str(info.cells_x).zfill(3)
        np.save(f"nearby_residual_x{fcells}_n{fangles}.npy", np.asarray(residual))

    # Run Nearby Problem
    logger.info("Calculating Nearby Solution...")
    nearby_scalar, nearby_keff = critical1d.nearby_power_iteration( \
                                    np.asarray(residual), nearby_rate, \
                                    materials, geometry, quadrature, solver)

    # Convert numerical angular flux to scalar flux
    numerical_scalar = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

    return numerical_scalar, numerical_keff, np.asarray(curve_fit_scalar), \
            curve_fit_keff, nearby_scalar, nearby_keff


def calculate_k_numerical_flux(numerical_flux, numerical_keff, materials, \
        geometry, quadrature, solver):
    """Calculate numerical flux for nearby k criticality problem."""
    # Ensure solving for angular flux at cell centers (fixed1d solver requirement)
    angular_solver = replace(solver, angular=True, flux_at_edges=0)
    # No numerical flux provided: run power iteration for keff then get angular flux
    if numerical_flux is None:
        scalar_solver = replace(solver, angular=False, flux_at_edges=0)
        scalar_flux, numerical_keff = critical1d.k_criticality(materials, geometry, \
                                                quadrature, scalar_solver)
        angular_params = create_params(materials, quadrature, geometry, angular_solver)
        numerical_flux = critical1d.known_flux(scalar_flux, numerical_keff, materials, \
                                               geometry, quadrature, angular_params)
        return numerical_flux, numerical_keff
    # Scalar flux provided, compute angular flux with single transport sweep
    elif numerical_flux.ndim == 2:
        angular_params = create_params(materials, quadrature, geometry, angular_solver)
        return critical1d.known_flux(numerical_flux, numerical_keff, materials, \
                                     geometry, quadrature, angular_params), numerical_keff
    # Angular flux provided, no change needed
    return numerical_flux, numerical_keff


cdef void _angular_curve_fit(double[:,:,:]& flux, double[:,:]& curve_fit, \
        double[:,:,:]& boundary_x, double[:,:,:]& int_angular, \
        double[:,:,:]& int_dx_angular, double[:,:]& int_scalar, \
        int[:]& medium_map, int[:]& x_splits, double[:]& knots_x, \
        double[:]& delta_x, double[:]& angle_w, bint block, bint quintic, \
        params info):
    """Dispatch to center- or edge-knot curve fit based on knot array size."""
    cdef double[:] edges_x, centers_x
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _angular_curve_fit_impl(flux, curve_fit, boundary_x, int_angular, \
                int_dx_angular, int_scalar, medium_map, x_splits, knots_x, \
                knots_x, edges_x, angle_w, block, quintic, True, info)
    else:
        centers_x = average_array(np.asarray(knots_x))
        _angular_curve_fit_impl(flux, curve_fit, boundary_x, int_angular, \
                int_dx_angular, int_scalar, medium_map, x_splits, knots_x, \
                centers_x, knots_x, angle_w, block, quintic, False, info)


cdef void _angular_curve_fit_impl(double[:,:,:]& flux, \
        double[:,:]& curve_fit, double[:,:,:]& boundary_x, \
        double[:,:,:]& int_angular, double[:,:,:]& int_dx_angular, \
        double[:,:]& int_scalar, int[:]& medium_map, int[:]& x_splits, \
        double[:]& knots_x, double[:]& eval_x, double[:]& bound_edges, \
        double[:]& angle_w, bint block, bint quintic, \
        bint center_knots, params info):

    # Initialize angle and group
    cdef int nn, gg

    # Initialize angular and group specific interpolations
    cdef double[:] spline, int_psi, int_dx

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", \
                ascii=True, position=0):
        # Iterate over angles
        for nn in tqdm(range(info.angles), desc="Curve Fit Angles", \
                    ascii=True, position=1, leave=False):
            # Create function
            approx = Interpolation(flux[:,nn,gg], knots_x, medium_map, \
                                   x_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(eval_x)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate x boundary
            boundary_x[0,nn,gg] = approx.interpolate(bound_edges[0])[0]
            boundary_x[1,nn,gg] = approx.interpolate(bound_edges[info.cells_x])[0]

            # Calculate integrals
            if center_knots:
                int_psi, int_dx = approx.integrate_centers(bound_edges)
            else:
                int_psi, int_dx = approx.integrate_edges()
            int_angular[:,nn,gg] = int_psi[:]
            int_dx_angular[:,nn,gg] = int_dx[:]

    # Populate int_scalar scalar flux
    tools._angular_to_scalar(int_angular, int_scalar, angle_w, info)


cdef void _angular_residual(double[:,:,:]& residual, double[:,:,:]& psi, \
        double[:,:,:]& int_dx, double[:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:,:]& external, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, params info):

    # Initialize angle, group and cell
    cdef int ii, nn, nn_q, og, ig, og_q, mat

    # Initialize off-scattering term
    cdef double off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(info.groups):
            og_q = 0 if external.shape[2] == 1 else og

            off_scatter = 0.0
            for ig in range(info.groups):
                off_scatter += phi[ii,ig] * (xs_scatter[mat,og,ig] \
                                + xs_fission[mat,og,ig])

            # Iterate over angles
            for nn in range(info.angles):
                nn_q = 0 if external.shape[1] == 1 else nn

                residual[ii,nn,og] = (angle_x[nn] * int_dx[ii,nn,og] \
                        + psi[ii,nn,og] * xs_total[mat,og]) - off_scatter \
                        - external[ii,nn_q,og_q] * delta_x[ii]


cdef void _scalar_curve_fit_residual(double[:,:,:]& flux, double[:,:]& curve_fit, \
        double[:,:]& boundary_x, double[:,:]& residual, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:,:]& external, int[:]& medium_map, double[:]& delta_x, \
        double[:]& knots_x, double[:]& edges_x, int[:]& x_splits, \
        double[:]& angle_x, double[:]& angle_w, bint block, bint quintic, \
        params info):

    # Initialize angle, group and cell
    cdef int nn, nn_q, gg

    # Initialize per-angle integrals (reused each iteration)
    int_angular = tools.array_1d(info.cells_x)
    int_dx_angular = tools.array_1d(info.cells_x)
    int_scalar = tools.array_2d(info.cells_x, info.groups)

    # Initialize group specific interpolations
    cdef double[:] spline, int_psi, int_dx

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", \
                ascii=True, position=0):

        # Iterate over angles
        for nn in tqdm(range(info.angles), desc="Curve Fit Angles", \
                    ascii=True, position=1, leave=False):

            nn_q = 0 if external.shape[1] == 1 else nn

            # Create interpolant for this angle/group
            approx = Interpolation(flux[:,nn,gg], knots_x, medium_map, \
                                   x_splits, block, quintic)

            # Accumulate scalar curve fit (angle-weighted sum)
            spline = approx.interpolate(knots_x)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Accumulate scalar boundary (angle-weighted sum)
            boundary_x[0,gg] += approx.interpolate(edges_x[0])[0] * angle_w[nn]
            boundary_x[1,gg] += approx.interpolate(edges_x[info.cells_x])[0] * angle_w[nn]

            # Compute cell integrals and accumulate into int_scalar
            int_psi, int_dx = approx.integrate_centers(edges_x)
            int_angular = int_psi[:]
            int_dx_angular = int_dx[:]
            tools._nearby_flux_to_scalar(int_scalar, int_psi, angle_w[nn], gg, info)

            # Accumulate on-scatter residual contribution for this angle/group
            tools._nearby_on_scatter(residual, int_angular, int_dx_angular, \
                    xs_total, external[:,nn_q,:], medium_map, delta_x, \
                    angle_x[nn], angle_w[nn], gg, gg, info)

    # Subtract off-scatter term (scatter + fission) from residual
    tools._nearby_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, medium_map, info)


cdef (double, double) _curve_fit_fission_source(double[:,:,:]& int_psi, \
        double[:,:,:]& int_dx, double[:,:]& int_phi, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:]& fission_source, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, double[:]& angle_w, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, nn, og, ig, mat

    # Initialize needed terms
    cdef double nearby_rate = 0.0
    cdef double curve_fit_keff = 0.0
    cdef double left_hand = 0.0
    cdef double right_hand = 0.0
    cdef double left_hand_off, right_hand_off

    # Zero out fission source
    fission_source[:,:] = 0.0

    # Iterate over cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(info.groups):
            right_hand_off = 0.0
            left_hand_off = 0.0
            for ig in range(info.groups):
                right_hand_off += int_phi[ii,ig] * xs_fission[mat,og,ig]
                left_hand_off += int_phi[ii,ig] * xs_scatter[mat,og,ig]

            # Create curve fit source with only one angle
            fission_source[ii,og] = right_hand_off

            # Update nearby fission rate
            nearby_rate += right_hand_off / delta_x[ii]

            # Iterate over angles
            for nn in range(info.angles):
                right_hand += angle_w[nn] * right_hand_off
                left_hand += angle_w[nn] * (angle_x[nn] * int_dx[ii,nn,og] \
                            + int_psi[ii,nn,og] * xs_total[mat,og] - left_hand_off)

    curve_fit_keff = right_hand / left_hand
    return nearby_rate, curve_fit_keff


cdef void _angular_residual_critical(double[:,:,:]& residual, \
        double[:,:,:]& int_psi, double[:,:,:]& int_dx, double[:,:]& int_phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:]& source, int[:]& medium_map, double[:]& angle_x, \
        double keff, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, nn, og, ig, mat

    # Initialize off-scattering term
    cdef double off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(info.groups):
            off_scatter = 0.0
            for ig in range(info.groups):
                off_scatter += int_phi[ii,ig] * xs_scatter[mat,og,ig]

            # Iterate over angles
            for nn in range(info.angles):
                residual[ii,nn,og] = (angle_x[nn] * int_dx[ii,nn,og] \
                                    + int_psi[ii,nn,og] * xs_total[mat,og]) \
                                    - off_scatter - source[ii,og] / keff
