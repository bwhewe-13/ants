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
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

import numpy as np
from tqdm.auto import tqdm

from ants import fixed2d, critical2d
from ants.utils.interp2d import Interpolation
from ants.utils.pytools import average_array

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants cimport parameters
from ants.parameters cimport params


def fixed_source(xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, edges_x, edges_y, \
        angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Angular directions
    cdef int NN = info.angles * info.angles
    
    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Numerical Solution...")
    numerical_flux = fixed2d.source_iteration(xs_total, xs_scatter, \
                            xs_fission, external, boundary_x, boundary_y, \
                            medium_map, delta_x, delta_y, angle_x, angle_y, \
                            angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = tools.array_4d(2, info.cells_y, NN, info.groups)
    curve_fit_boundary_y = tools.array_4d(2, info.cells_x, NN, info.groups)
    # curve_fit_flux = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Calculate curve fit at knots
    print("Calculating Angular Curve Fit Solution...")    
    # Knots at cell centers
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)
    _angular_curve_fit(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
            curve_fit_boundary_y, int_angular, int_dx_angular, int_dy_angular, \
            int_scalar, medium_map, centers_x, centers_y, edges_x, edges_y, \
            x_splits, y_splits, angle_w, block, quintic, info)

    # Calculate residual for each cell
    print("Calculating Angular Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _angular_residual(residual, int_angular, int_dx_angular, int_dy_angular, \
            int_scalar, xs_total, xs_scatter, xs_fission, external, \
            medium_map, delta_x, delta_y, angle_x, angle_y, info)
    
    fangles = str(info.angles).zfill(2)
    fcells = str(info.cells_x).zfill(3)
    np.save(f"nearby_residual_x{fcells}_n{fangles}", np.asarray(residual))
    np.save(f"nearby_boundary_x_x{fcells}_n{fangles}", np.asarray(curve_fit_boundary_x))
    np.save(f"nearby_boundary_y_x{fcells}_n{fangles}", np.asarray(curve_fit_boundary_y))

    # Run Nearby Problem
    print("Calculating Nearby Solution...")
    info.edges = 0
    if kwargs.get("zero_bounds", False):
        print("Removing Curve Fit Boundary Conditions...")
        curve_fit_boundary_x = boundary_x.copy()
        curve_fit_boundary_y = boundary_y.copy()

    nearby_flux = fixed2d.source_iteration(xs_total, xs_scatter, xs_fission, \
                                (external + residual), curve_fit_boundary_x, \
                                curve_fit_boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

    return numerical_flux, np.asarray(curve_fit_flux), nearby_flux


def fixed_source_angular_residual(scalar_flux, xs_total, xs_scatter, xs_fission, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        edges_x, edges_y, angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)
    group = kwargs.get("group", -1)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)
    
    # Angular directions
    cdef int NN = info.angles * info.angles
    
    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Angular Flux Solution...")
    numerical_flux = fixed2d.known_source_calculation(scalar_flux, xs_total, \
                            xs_scatter + xs_fission, external, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, params_dict)

    # Initialize curve fit
    curve_fit_boundary_x = tools.array_4d(2, info.cells_y, NN, info.groups)
    curve_fit_boundary_y = tools.array_4d(2, info.cells_x, NN, info.groups)
    # curve_fit_flux = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    print("Calculating Angular Curve Fit Solution...")
    # Knots at cell centers
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)
    _angular_curve_fit(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
            curve_fit_boundary_y, int_angular, int_dx_angular, int_dy_angular, \
            int_scalar, medium_map, centers_x, centers_y, edges_x, edges_y, \
            x_splits, y_splits, angle_w, block, quintic, info)
        
    # Calculate residual for each cell
    print("Calculating Angular Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _angular_residual(residual, int_angular, int_dx_angular, int_dy_angular, \
            int_scalar, xs_total, xs_scatter, xs_fission, external, \
            medium_map, delta_x, delta_y, angle_x, angle_y, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
                np.asarray(curve_fit_boundary_y), np.asarray(residual)


def fixed_source_scalar_residual(scalar_flux, xs_total, xs_scatter, xs_fission, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        edges_x, edges_y, angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)
    group = kwargs.get("group", -1)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Angular directions
    cdef int NN = info.angles * info.angles

    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)
    else:
        x_splits = kwargs.get("x_splits")
        y_splits = kwargs.get("y_splits")

    # Initialize curve fit and residual
    num_gg = 1 if (group > -1) else info.groups
    curve_fit_boundary_x = tools.array_3d(2, info.cells_y, num_gg)
    curve_fit_boundary_y = tools.array_3d(2, info.cells_x, num_gg)
    curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, num_gg)
    residual = tools.array_3d(info.cells_x, info.cells_y, num_gg)

    # Knots at cell centers
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)

    # Run Numerical Solution
    print("Calculating Angular Flux Solution...")
    if group == -1:
        numerical_flux = fixed2d.known_source_calculation(scalar_flux, \
                            xs_total, xs_scatter + xs_fission, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, params_dict)
        
        print("Calculating Scalar Curve Fit Solution and Residual...")
        _scalar_curve_fit_residual(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, curve_fit_boundary_y, residual, \
                xs_total, xs_scatter, xs_fission, external, medium_map, \
                delta_x, delta_y, centers_x, centers_y, edges_x, edges_y, \
                x_splits, y_splits, angle_x, angle_y, angle_w, block, \
                quintic, info)
    else:
        numerical_flux = fixed2d.known_source_single(scalar_flux, xs_total, \
                            xs_scatter + xs_fission, external, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, group, params_dict)

        print("Calculating Scalar Curve Fit Solution and Residual...")
        _scalar_curve_fit_residual_single(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, curve_fit_boundary_y, residual, \
                xs_total, xs_scatter, xs_fission, external, medium_map, \
                delta_x, delta_y, centers_x, centers_y, edges_x, edges_y, \
                x_splits, y_splits, angle_x, angle_y, angle_w, group, \
                block, quintic, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
                np.asarray(curve_fit_boundary_y), np.asarray(residual)


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
            # curve_fit[...,nn,gg] = spline[:,:]
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
    cdef float off_scatter

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
                int_dy_angular, xs_total, external[:,:,nn_q,:], medium_map, \
                delta_x, delta_y, angle_x[nn], angle_y[nn], angle_w[nn], \
                gg, gg, info)

    tools._nearby_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, medium_map, info)


cdef void _scalar_curve_fit_residual_single(double[:,:,:,:]& flux, \
        double[:,:,:]& curve_fit, double[:,:,:]& boundary_x, \
        double[:,:,:]& boundary_y, double[:,:,:]& residual,
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& knots_x, double[:]& knots_y, double[:]& edges_x, \
        double[:]& edges_y, int[:]& x_splits, int[:]& y_splits, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        int group, bint block, bint quintic, params info):

    # Initialize angle, group and cell
    cdef int ii, jj, nn, nn_q

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize integrals
    int_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dx_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dy_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, 1)

    # Initialize group specific interpolations
    cdef double[:,:] spline, int_psi, int_dx, int_dy, boundary
    cdef double[2] bounds_x = [edges_x[0], edges_x[info.cells_x]]
    cdef double[2] bounds_y = [edges_y[0], edges_y[info.cells_y]]

    print(f"Calculating for Group: {group}")
    # Iterate over angles
    for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True, position=0):

        nn_q = 0 if external.shape[2] == 1 else nn

        # Create function
        approx = Interpolation(flux[:,:,nn,0], knots_x, knots_y, \
                        medium_map, x_splits, y_splits, block, quintic)

        # Interpolate the knots
        spline = approx.interpolate(knots_x, knots_y)
        tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], 0, info)

        # Interpolate y boundary
        boundary = approx.interpolate(knots_x, bounds_y)
        tools._nearby_boundary_to_scalar(boundary_y, boundary[:,:].T, \
                                        angle_w[nn], 0, info)

        # Interpolate x boundary
        boundary = approx.interpolate(bounds_x, knots_y)
        tools._nearby_boundary_to_scalar(boundary_x, boundary, angle_w[nn], 0, info)

        # Calculate integrals
        int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
        int_angular = int_psi[:,:]
        int_dx_angular = int_dx[:,:]
        int_dy_angular = int_dy[:,:]
        tools._nearby_flux_to_scalar(int_scalar, int_psi, angle_w[nn], 0, info)

        # Update Residual - On scatter
        tools._nearby_on_scatter(residual, int_angular, int_dx_angular, \
                            int_dy_angular, xs_total, external[:,:,nn_q,:], \
                            medium_map, delta_x, delta_y, angle_x[nn], \
                            angle_y[nn], angle_w[nn], 0, group, info)

    np.save(f"scalar_flux_integral_x{str(info.cells_x).zfill(3)}_" \
            f"n{str(info.angles).zfill(2)}_gg{str(group).zfill(3)}", \
            np.asarray(int_scalar))


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
    cdef float off_scatter

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


def criticality(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        delta_y, edges_x, edges_y, angle_x, angle_y, angle_w, \
        params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_criticality(info)
    block = False if (info.materials == 1) else kwargs.get("block", True)
    
    # Angular directions
    cdef int NN = info.angles * info.angles

    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Numerical Solution...")
    numerical_flux, numerical_keff = critical2d.power_iteration(xs_total, \
                    xs_scatter, xs_fission, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.cells_y, NN, info.groups))
    curve_fit_boundary_y = np.zeros((2, info.cells_x, NN, info.groups))
    # curve_fit_flux = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    curve_fit_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy_angular = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Calculate curve fit at knots
    print("Calculating Angular Curve Fit Solution...")
    # Knots at cell centers
    centers_x = average_array(edges_x)
    centers_y = average_array(edges_y)
    _angular_curve_fit(numerical_flux, curve_fit_scalar, curve_fit_boundary_x, \
            curve_fit_boundary_y, int_angular, int_dx_angular, int_dy_angular, \
            int_scalar, medium_map, centers_x, centers_y, edges_x, edges_y, \
            x_splits, y_splits, angle_w, block, quintic, info)
    
    # Create curve fit source, curve fit keff, nearby reaction rate
    fission_source = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(int_angular, \
                            int_dx_angular, int_dy_angular, int_scalar, \
                            xs_total, xs_scatter, xs_fission, fission_source, \
                            medium_map, delta_x, delta_y, angle_x, angle_y, \
                            angle_w, info)

    # Calculate residual for each cell
    print("Calculating Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _angular_residual_critical(residual, int_angular, int_dx_angular, \
            int_dy_angular, int_scalar, xs_total, xs_scatter, fission_source, \
            medium_map, angle_x, angle_y, curve_fit_keff, info)
    
    fangles = str(info.angles).zfill(2)
    fcells = str(info.cells_x).zfill(3)
    np.save(f"nearby_residual_x{fcells}_n{fangles}", np.asarray(residual))

    # Run Nearby Problem
    print("Calculating Nearby Solution...")
    info.edges = 0
    nearby_scalar, nearby_keff = critical2d.nearby_power(xs_total, xs_scatter, \
                        xs_fission, residual, medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, nearby_rate, info)

    # Convert numerical_flux to scalar flux
    numerical_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

    # Return numerical, curve fit, and nearby data
    return numerical_scalar, numerical_keff, np.asarray(curve_fit_scalar), \
            curve_fit_keff, nearby_scalar, nearby_keff, nearby_rate


cdef (double, double) _curve_fit_fission_source(double[:,:,:,:]& psi, \
        double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, double[:,:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:]& fission_source, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):

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
                    right_hand_off += phi[ii,jj,ig] * xs_fission[mat,og,ig]
                    left_hand_off += phi[ii,jj,ig] * xs_scatter[mat,og,ig]

                # Create curve fit source with only one angle
                fission_source[ii,jj,og] = right_hand_off

                # Update nearby fission rate
                nearby_rate += right_hand_off / (delta_x[ii] * delta_y[jj])

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    right_hand += angle_w[nn] * right_hand_off
                    left_hand += angle_w[nn] * (angle_x[nn] * dxpsi[ii,jj,nn,og] \
                                + angle_y[nn] * dypsi[ii,jj,nn,og] \
                                + psi[ii,jj,nn,og] * xs_total[mat,og] \
                                - left_hand_off)

    curve_fit_keff = right_hand / left_hand
    return nearby_rate, curve_fit_keff


cdef void _angular_residual_critical(double[:,:,:,:]& residual, \
        double[:,:,:,:]& psi, double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& source, int[:,:]& medium_map, double[:]& angle_x, \
        double[:]& angle_y, double keff, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, jj, nn, og, ig, mat

    # Initialize off-scattering term
    cdef float off_scatter

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
                            + angle_y[nn] * dypsi[ii,jj,nn,og]
                            + psi[ii,jj,nn,og] * xs_total[mat,og]) \
                            - off_scatter - source[ii,jj,og] / keff