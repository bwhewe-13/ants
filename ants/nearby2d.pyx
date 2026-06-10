########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two-Dimensional Nearby Problems
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

from ants import critical2d, fixed2d
from ants.datatypes import SourceData, create_params
from ants.utils.interp2d import Interpolation
from ants.utils.pytools import average_array

from ants cimport cytools_2d as tools
from ants cimport multi_group_2d as mg
from ants cimport parameters
from ants.parameters cimport params

logger = logging.getLogger(__name__)


def fixed_source(materials, sources, geometry, quadrature, solver, **kwargs):
    """Compute the nearby fixed-source residual and optionally run the nearby solve.

    Consolidates fixed_source / fixed_source_angular_residual /
    fixed_source_scalar_residual into a single entry point.

    kwargs
    ------
    numerical_flux : ndarray or None
        Pre-computed flux.  None -> run source_iteration.
        Shape (cells_x, cells_y, groups) -> convert to angular via known_flux.
        Shape (cells_x, cells_y, NN, groups) -> use directly.
    scalar_residual : bool, default False
        Use scalar curve fit instead of angular.
    return_nearby : bool, default False
        If True, run the nearby fixed-source solve and return its scalar flux.
    save_residual : bool, default True
        Write .npy files for the residual and boundaries.
    x_splits, y_splits : int arrays
        Custom spatial interpolation splits.
    quintic : bool, default True
    block : bool, default True (False when materials == 1)
    zero_bounds : bool, default False
        Replace curve-fit boundaries with the original vacuum boundaries.

    Returns (5-tuple)
    -----------------
    numerical_flux, curve_fit_flux, curve_fit_boundary_x,
    curve_fit_boundary_y, residual

    With return_nearby=True (6-tuple)
    ----------------------------------
    ... + nearby_flux
    """
    quintic = kwargs.get("quintic", True)
    scalar_residual = kwargs.get("scalar_residual", False)
    save_residual = kwargs.get("save_residual", True)
    return_nearby = kwargs.get("return_nearby", False)
    numerical_flux = kwargs.get("numerical_flux", None)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef double[:,:,:,:] external = sources.external
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Convert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Angular directions
    cdef int NN = info.angles * info.angles

    # Spatial splits (empty -> no splits)
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)
    else:
        x_splits = kwargs.get("x_splits")
        y_splits = kwargs.get("y_splits", np.zeros((0,), dtype=np.int32))

    # Grid coordinates (always from geometry)
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)

    # --- Numerical flux ---
    logger.info("Calculating Numerical Solution...")
    angular_solver = replace(solver, angular=True, flux_at_edges=0)
    if numerical_flux is None:
        numerical_flux = fixed2d.fixed_source(materials, sources, geometry,
                                              quadrature, angular_solver)
    elif np.asarray(numerical_flux).ndim == 3:
        angular_params = create_params(materials, quadrature, geometry, angular_solver)
        xs_matrix = np.asarray(xs_scatter) + np.asarray(xs_fission)
        numerical_flux = fixed2d.known_flux(numerical_flux, xs_total, xs_matrix,
                                            sources.external, sources.boundary_x,
                                            sources.boundary_y, geometry, quadrature,
                                            angular_params)
    # else: 4-D angular flux provided, use directly

    # --- Curve fit + residual ---
    if scalar_residual:
        curve_fit_boundary_x = tools.array_3d(2, info.cells_y, info.groups)
        curve_fit_boundary_y = tools.array_3d(2, info.cells_x, info.groups)
        curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
        residual = tools.array_3d(info.cells_x, info.cells_y, info.groups)

        logger.info("Calculating Scalar Curve Fit Solution and Residual...")
        _scalar_curve_fit_residual(numerical_flux, curve_fit_flux,
                curve_fit_boundary_x, curve_fit_boundary_y, residual,
                xs_total, xs_scatter, xs_fission, external, medium_map,
                delta_x, delta_y, centers_x, centers_y, edges_x, edges_y,
                x_splits, y_splits, angle_x, angle_y, angle_w, block, quintic, info)
    else:
        curve_fit_boundary_x = tools.array_4d(2, info.cells_y, NN, info.groups)
        curve_fit_boundary_y = tools.array_4d(2, info.cells_x, NN, info.groups)
        curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
        int_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
        int_dx_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
        int_dy_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
        int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

        logger.info("Calculating Angular Curve Fit Solution...")
        _angular_curve_fit(numerical_flux, curve_fit_flux, curve_fit_boundary_x,
                curve_fit_boundary_y, int_angular, int_dx_angular, int_dy_angular,
                int_scalar, medium_map, centers_x, centers_y, edges_x, edges_y,
                x_splits, y_splits, angle_w, block, quintic, info)

        logger.info("Calculating Angular Residual...")
        residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
        _angular_residual(residual, int_angular, int_dx_angular, int_dy_angular,
                int_scalar, xs_total, xs_scatter, xs_fission, external,
                medium_map, delta_x, delta_y, angle_x, angle_y, info)

    if save_residual:
        fangles = str(info.angles).zfill(2)
        fcells = str(info.cells_x).zfill(3)
        np.save(f"nearby_residual_x{fcells}_n{fangles}.npy", np.asarray(residual))
        np.save(f"nearby_boundary_x_x{fcells}_n{fangles}.npy", np.asarray(curve_fit_boundary_x))
        np.save(f"nearby_boundary_y_x{fcells}_n{fangles}.npy", np.asarray(curve_fit_boundary_y))

    if not return_nearby:
        return numerical_flux, np.asarray(curve_fit_flux), \
               np.asarray(curve_fit_boundary_x), np.asarray(curve_fit_boundary_y), \
               np.asarray(residual)

    # --- Nearby solve ---
    logger.info("Calculating Nearby Solution...")
    if kwargs.get("zero_bounds", False):
        logger.info("Removing Curve Fit Boundary Conditions...")
        nearby_boundary_x = np.asarray(sources.boundary_x).copy()
        nearby_boundary_y = np.asarray(sources.boundary_y).copy()
    else:
        nearby_boundary_x = np.asarray(curve_fit_boundary_x)
        nearby_boundary_y = np.asarray(curve_fit_boundary_y)

    nearby_sources = SourceData(
        np.asarray(sources.external) + np.asarray(residual),
        nearby_boundary_x,
        nearby_boundary_y,
    )
    scalar_solver = replace(solver, angular=False, flux_at_edges=0)
    nearby_flux = fixed2d.fixed_source(materials, nearby_sources, geometry,
                                       quadrature, scalar_solver)

    return numerical_flux, np.asarray(curve_fit_flux), \
           np.asarray(curve_fit_boundary_x), np.asarray(curve_fit_boundary_y), \
           np.asarray(residual), nearby_flux


def k_criticality(materials, geometry, quadrature, solver, **kwargs):
    """Compute the nearby k-criticality residual and run the nearby power iteration.

    Consolidates criticality / criticality_angular_residual /
    criticality_scalar_residual into a single entry point.

    kwargs
    ------
    numerical_flux : ndarray or None
        Pre-computed flux.  None -> run power iteration.
        Shape (cells_x, cells_y, groups) -> convert to angular via known_flux.
        Shape (cells_x, cells_y, NN, groups) -> use directly.
    numerical_keff : float or None
        Required when numerical_flux is provided.
    scalar_residual : bool, default False
        Use scalar curve-fit residual path instead of angular.
    save_residual : bool, default True
        Write .npy files for the residual.
    x_splits, y_splits : int arrays
        Custom spatial interpolation splits.
    quintic : bool, default True
    block : bool, default True (False when materials == 1)

    Returns (7-tuple)
    -----------------
    numerical_scalar, numerical_keff, curve_fit_scalar, curve_fit_keff,
    nearby_scalar, nearby_keff, nearby_rate
    """
    quintic = kwargs.get("quintic", True)
    scalar_residual = kwargs.get("scalar_residual", False)
    save_residual = kwargs.get("save_residual", True)
    numerical_flux = kwargs.get("numerical_flux", None)
    numerical_keff = kwargs.get("numerical_keff", None)

    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total = materials.total
    cdef double[:,:,:] xs_scatter = materials.scatter
    cdef double[:,:,:] xs_fission = tools._fission_matrix(materials.fission, materials.chi)
    cdef int[:,:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] delta_y = geometry.delta_y
    cdef double[:] angle_x = quadrature.angle_x
    cdef double[:] angle_y = quadrature.angle_y
    cdef double[:] angle_w = quadrature.angle_w

    # Convert ProblemParameters to type params
    params = create_params(materials, quadrature, geometry, solver)
    info = parameters._to_params(params)
    parameters._check_nearby2d_criticality(info)
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Angular directions
    cdef int NN = info.angles * info.angles

    # Spatial splits (empty -> no splits)
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)
    else:
        x_splits = kwargs.get("x_splits")
        y_splits = kwargs.get("y_splits", np.zeros((0,), dtype=np.int32))

    # Grid coordinates
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)

    # --- Numerical flux + keff ---
    logger.info("Calculating Numerical Solution...")
    angular_solver = replace(solver, angular=True, flux_at_edges=0)
    angular_params = create_params(materials, quadrature, geometry, angular_solver)

    if numerical_flux is None:
        scalar_solver = replace(solver, angular=False, flux_at_edges=0)
        scalar_flux, numerical_keff = critical2d.k_criticality(materials, geometry,
                                                               quadrature, scalar_solver)
        numerical_flux = critical2d.known_flux(scalar_flux, numerical_keff, materials,
                                               geometry, quadrature, angular_params)
    elif np.asarray(numerical_flux).ndim == 3:
        numerical_flux = critical2d.known_flux(numerical_flux, numerical_keff, materials,
                                               geometry, quadrature, angular_params)
    # else: 4-D angular flux provided, use directly

    if scalar_residual:
        # Scalar curve-fit + residual (no boundary arrays)
        curve_fit_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
        residual = tools.array_3d(info.cells_x, info.cells_y, info.groups)
        nearby_array = tools.array_1d(2)

        logger.info("Calculating Scalar Curve Fit Solution and Residual...")
        _scalar_curve_fit_residual_no_bcs(numerical_flux, curve_fit_scalar,
                residual, nearby_array, xs_total, xs_scatter, xs_fission,
                medium_map, delta_x, delta_y, centers_x, centers_y, edges_x, edges_y,
                x_splits, y_splits, angle_x, angle_y, angle_w, block, quintic, info)

        nearby_rate_s = float(nearby_array[0])
        curve_fit_keff_s = float(nearby_array[1])

        if save_residual:
            fangles = str(info.angles).zfill(2)
            fcells = str(info.cells_x).zfill(3)
            np.save(f"nearby_residual_x{fcells}_n{fangles}.npy", np.asarray(residual))

        # Reshape 3-D scalar residual -> 4-D for nearby_power_iteration
        scalar_residual_4d = np.asarray(residual).reshape(
            info.cells_x, info.cells_y, 1, info.groups)

        logger.info("Calculating Nearby Solution...")
        nearby_scalar, nearby_keff = critical2d.nearby_power_iteration(
            scalar_residual_4d, nearby_rate_s, materials, geometry, quadrature, solver)

        numerical_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
        tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

        return np.asarray(numerical_scalar), numerical_keff, np.asarray(curve_fit_scalar), \
               curve_fit_keff_s, nearby_scalar, nearby_keff, nearby_rate_s

    # --- Angular path (default) ---
    curve_fit_boundary_x = np.zeros((2, info.cells_y, NN, info.groups))
    curve_fit_boundary_y = np.zeros((2, info.cells_x, NN, info.groups))
    curve_fit_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    int_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    logger.info("Calculating Angular Curve Fit Solution...")
    _angular_curve_fit(numerical_flux, curve_fit_scalar, curve_fit_boundary_x,
            curve_fit_boundary_y, int_angular, int_dx_angular, int_dy_angular,
            int_scalar, medium_map, centers_x, centers_y, edges_x, edges_y,
            x_splits, y_splits, angle_w, block, quintic, info)

    # Curve-fit keff and nearby reaction rate
    fission_source = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    nearby_rate, curve_fit_keff = _angular_curve_fit_source(int_angular,
                            int_dx_angular, int_dy_angular, int_scalar,
                            xs_total, xs_scatter, xs_fission, fission_source,
                            medium_map, delta_x, delta_y, angle_x, angle_y,
                            angle_w, info)

    logger.info("Calculating Residual...")
    angular_residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _angular_residual_critical(angular_residual, int_angular, int_dx_angular,
            int_dy_angular, int_scalar, xs_total, xs_scatter, fission_source,
            medium_map, angle_x, angle_y, curve_fit_keff, info)

    if save_residual:
        fangles = str(info.angles).zfill(2)
        fcells = str(info.cells_x).zfill(3)
        np.save(f"nearby_residual_x{fcells}_n{fangles}.npy", np.asarray(angular_residual))

    logger.info("Calculating Nearby Solution...")
    scalar_residual_arr = tools.array_4d(info.cells_x, info.cells_y, 1, info.groups)
    tools._nearby_angular_to_scalar(angular_residual, scalar_residual_arr, angle_w, info)

    nearby_scalar, nearby_keff = critical2d.nearby_power_iteration(
                        scalar_residual_arr, nearby_rate, materials, geometry,
                        quadrature, solver)

    numerical_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

    return np.asarray(numerical_scalar), numerical_keff, np.asarray(curve_fit_scalar), \
           curve_fit_keff, nearby_scalar, nearby_keff, nearby_rate


def off_scatter_corrector(double[:,:,:]& residual, double[:,:,:]& scalar_flux, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, int[:,:]& medium_map):
    """ When calculating one energy group of nearby problems, it corrects
    for the off-scattering term by using the scalar flux integral
    Arguments:
        residual (float [cells_x, cells_y, 1]): single (uncorrected)
                                                energy group residual
        scalar_flux (float [cells_x, cells_y, groups]): scalar flux integral
        xs_scatter (float [materials, groups, groups]): problem scattering xs
        xs_fission (float [materials, groups, groups]): problem fission xs
        medium_map (int [cells_x, cells_y]): problem medium map
        group (int): Specific energy group residual is part of
    Returns:
        corrected residual (float [cells_x, cells_y, 1])
    """
    # Initialize iterables
    cdef int ii, jj, mat, og, ig

    # Initialize off-scattering term
    cdef double off_scatter

    cells_x = residual.shape[0]
    cells_y = residual.shape[1]
    groups = residual.shape[2]

    # Iterate over spatial cells
    for ii in range(cells_x):
        for jj in range(cells_y):
            mat = medium_map[ii,jj]
            # Iterate over groups
            for og in range(groups):
                off_scatter = 0.0
                for ig in range(groups):
                    off_scatter += scalar_flux[ii,jj,ig] \
                                * (xs_scatter[mat,og,ig] + xs_fission[mat,og,ig])
                residual[ii,jj,og] -= (off_scatter)

    return residual


cdef void _angular_curve_fit(double[:,:,:,:]& flux, double[:,:,:]& curve_fit, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& int_angular, double[:,:,:,:]& int_dx_angular, \
        double[:,:,:,:]& int_dy_angular, double[:,:,:]& int_scalar, \
        int[:,:]& medium_map, double[:]& knots_x, double[:]& knots_y, \
        double[:]& edges_x, double[:]& edges_y, int[:]& x_splits, \
        int[:]& y_splits, double[:]& angle_w, bint block, bint quintic, \
        params info):

    # Initialize cell, angle, and group
    cdef int ii, jj, nn, gg

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize angular and group specific interpolations
    cdef double[:,:] spline, int_psi, int_dx, int_dy, boundary
    cdef double[2] bounds_x = [edges_x[0], edges_x[info.cells_x]]
    cdef double[2] bounds_y = [edges_y[0], edges_y[info.cells_y]]

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", ascii=True, position=0):
        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True, position=1, leave=False):
            # Create function
            approx = Interpolation(flux[:,:,nn,gg], knots_x, knots_y, \
                        medium_map, x_splits, y_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(knots_x, knots_y)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate y boundary
            boundary = approx.interpolate(knots_x, bounds_y)
            boundary_y[...,nn,gg] = boundary[:,:].T

            # Interpolate x boundary
            boundary = approx.interpolate(bounds_x, knots_y)
            boundary_x[...,nn,gg] = boundary[:,:]

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
            int_angular[...,nn,gg] = int_psi[:,:]
            int_dx_angular[...,nn,gg] = int_dx[:,:]
            int_dy_angular[...,nn,gg] = int_dy[:,:]

    # Populate int_scalar flux
    tools._angular_to_scalar(int_angular, int_scalar, angle_w, info)


cdef void _angular_residual(double[:,:,:,:]& residual, double[:,:,:,:]& psi, \
        double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, double[:,:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, params info):

    # Initialize angle, group and cell
    cdef int ii, jj, nn, og, ig, mat, nn_q, gg_q

    # Initialize off-scattering term
    cdef double off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            # Iterate over groups
            for og in range(info.groups):
                gg_q = 0 if external.shape[3] == 1 else og
                off_scatter = 0.0
                for ig in range(info.groups):
                    off_scatter += phi[ii,jj,ig] * (xs_scatter[mat,og,ig] \
                                    + xs_fission[mat,og,ig])

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    nn_q = 0 if external.shape[2] == 1 else nn
                    residual[ii,jj,nn,og] = (angle_x[nn] * dxpsi[ii,jj,nn,og]) \
                                + (angle_y[nn] * dypsi[ii,jj,nn,og]) - off_scatter \
                                + (psi[ii,jj,nn,og] * xs_total[mat,og]) \
                                - (external[ii,jj,nn_q,gg_q] * delta_x[ii] * delta_y[jj])


cdef void _scalar_curve_fit_residual(double[:,:,:,:]& flux, \
        double[:,:,:]& curve_fit, double[:,:,:]& boundary_x, \
        double[:,:,:]& boundary_y, double[:,:,:]& residual,
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& knots_x, double[:]& knots_y, double[:]& edges_x, \
        double[:]& edges_y, int[:]& x_splits, int[:]& y_splits, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        bint block, bint quintic, params info):

    # Initialize angle, group and cell
    cdef int ii, jj, nn, nn_q, gg

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize integrals
    int_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dx_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dy_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize group specific interpolations
    cdef double[:,:] spline, int_psi, int_dx, int_dy, boundary
    cdef double[2] bounds_x = [edges_x[0], edges_x[info.cells_x]]
    cdef double[2] bounds_y = [edges_y[0], edges_y[info.cells_y]]

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", ascii=True, position=0):

        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True, position=1, leave=False):

            nn_q = 0 if external.shape[2] == 1 else nn

            # Create function
            approx = Interpolation(flux[:,:,nn,gg], knots_x, knots_y, \
                        medium_map, x_splits, y_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(knots_x, knots_y)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate y boundary
            boundary = approx.interpolate(knots_x, bounds_y)
            tools._nearby_boundary_to_scalar(boundary_y, boundary[:,:].T, \
                                            angle_w[nn], gg, info)

            # Interpolate x boundary
            boundary = approx.interpolate(bounds_x, knots_y)
            tools._nearby_boundary_to_scalar(boundary_x, boundary, angle_w[nn], gg, info)

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
            int_angular = int_psi[:,:]
            int_dx_angular = int_dx[:,:]
            int_dy_angular = int_dy[:,:]
            tools._nearby_flux_to_scalar(int_scalar, int_psi, angle_w[nn], gg, info)

            # Update Residual - On scatter
            tools._nearby_on_scatter(residual, int_angular, int_dx_angular, \
                    int_dy_angular, xs_total, external[:,:,nn_q,:], \
                    medium_map, delta_x, delta_y, angle_x[nn], angle_y[nn], \
                    angle_w[nn], gg, gg, info)

    tools._nearby_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, medium_map, info)


cdef void _scalar_curve_fit_residual_no_bcs(double[:,:,:,:]& flux, \
        double[:,:,:]& curve_fit, double[:,:,:]& residual, \
        double[:]& nearby_array, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& knots_x, double[:]& knots_y, double[:]& edges_x, \
        double[:]& edges_y, int[:]& x_splits, int[:]& y_splits, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        bint block, bint quintic, params info):

    # Initialize cell, angle, and group
    cdef int ii, jj, nn, gg

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize integrals
    int_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dx_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dy_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize fission source
    fission_source = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize angular and group specific interpolations
    cdef double[:,:] spline, int_psi, int_dx, int_dy

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", ascii=True, position=0):
        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True, position=1, leave=False):

            # Create function
            approx = Interpolation(flux[:,:,nn,gg], knots_x, knots_y, \
                        medium_map, x_splits, y_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(knots_x, knots_y)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
            int_angular = int_psi[:,:]
            int_dx_angular = int_dx[:,:]
            int_dy_angular = int_dy[:,:]
            tools._nearby_flux_to_scalar(int_scalar, int_psi, angle_w[nn], gg, info)

            # Update Residual - On scatter
            tools._nearby_critical_on_scatter(residual, int_angular, \
                    int_dx_angular, int_dy_angular, xs_total, medium_map, \
                    angle_x[nn], angle_y[nn], angle_w[nn], gg, gg, info)

    # Update Residual - Off scatter
    # Calculate curve fit keff / nearby reaction rate (nearby_array)
    tools._nearby_critical_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, fission_source, nearby_array, medium_map, delta_x, \
            delta_y, angle_w, info)

    # Update Residual - Fission term
    tools._nearby_critical_residual_source(residual, fission_source, \
            nearby_array[1], info)


cdef (double, double) _angular_curve_fit_source(double[:,:,:,:]& int_angular, \
        double[:,:,:,:]& int_dx_angular, double[:,:,:,:]& int_dy_angular, \
        double[:,:,:]& int_scalar, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:,:]& fission_source, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, jj, nn, og, ig, mat

    # Initialize needed terms
    cdef double nearby_rate = 0.0
    cdef double curve_fit_keff = 0.0
    cdef double left_hand = 0.0
    cdef double right_hand = 0.0
    cdef double left_hand_off, right_hand_off

    # Zero out fission source
    fission_source[:,:,:] = 0.0

    # Iterate over cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over groups
            for og in range(info.groups):
                right_hand_off = 0.0
                left_hand_off = 0.0
                for ig in range(info.groups):
                    right_hand_off += int_scalar[ii,jj,ig] * xs_fission[mat,og,ig]
                    left_hand_off += int_scalar[ii,jj,ig] * xs_scatter[mat,og,ig]

                # Create curve fit source with only one angle
                fission_source[ii,jj,og] = right_hand_off

                # Update nearby fission rate
                nearby_rate += right_hand_off / (delta_x[ii] * delta_y[jj])

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    right_hand += angle_w[nn] * right_hand_off
                    left_hand += angle_w[nn] * (angle_x[nn] * int_dx_angular[ii,jj,nn,og] \
                                + angle_y[nn] * int_dy_angular[ii,jj,nn,og] \
                                + int_angular[ii,jj,nn,og] * xs_total[mat,og] \
                                - left_hand_off)

    curve_fit_keff = right_hand / left_hand
    return nearby_rate, curve_fit_keff


cdef void _angular_residual_critical(double[:,:,:,:]& residual, \
        double[:,:,:,:]& psi, double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& fission_source, int[:,:]& medium_map, double[:]& angle_x, \
        double[:]& angle_y, double keff, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, jj, nn, og, ig, mat

    # Initialize off-scattering term
    cdef double off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]

            # Iterate over groups
            for og in range(info.groups):
                off_scatter = 0.0
                for ig in range(info.groups):
                    off_scatter += phi[ii,jj,ig] * xs_scatter[mat,og,ig]

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    residual[ii,jj,nn,og] = (angle_x[nn] * dxpsi[ii,jj,nn,og] \
                            + angle_y[nn] * dypsi[ii,jj,nn,og] \
                            + psi[ii,jj,nn,og] * xs_total[mat,og]) \
                            - off_scatter - fission_source[ii,jj,og] / keff
