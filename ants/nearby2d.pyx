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
        boundary_y, medium_map, delta_x, delta_y, knots_x, knots_y, \
        angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = True if kwargs.get("block", True) or (info.materials != 1) else False

    # Angular directions
    cdef int NN = info.angles * info.angles
    
    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("1. Calculating Numerical Solution...")
    numerical_flux = fixed2d.source_iteration(xs_total, xs_scatter, \
                                xs_fission, external, boundary_x, boundary_y, \
                                medium_map, delta_x, delta_y, angle_x, \
                                angle_y, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.cells_y, NN, info.groups))
    curve_fit_boundary_y = np.zeros((2, info.cells_x, NN, info.groups))
    curve_fit_flux = np.zeros((info.cells_x, info.cells_y, NN, info.groups))

    # Initialize curve fit integrals
    int_psi = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_phi = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Calculate curve fit at knots
    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        edges_y = np.insert(np.cumsum(delta_y), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, edges_x, edges_y, x_splits, \
                    y_splits, angle_w, block, quintic, info)
    # Knots at cell edges
    elif knots_x.shape[0] == (info.cells_x + 1):
        centers_x = average_array(knots_x)
        centers_y = average_array(knots_y)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, centers_x, centers_y, \
                    x_splits, y_splits, angle_w, block, quintic, info)
    
    # Knots at cell centers and medium edges
    elif knots_x.shape[0] == (info.cells_x + 2):
        print("{xx}\n* This method has not been verified\n{xx}".format(xx="*"*50))
        modified_flux = tools.array_4d(info.cells_x + 2, info.cells_y + 2, \
                                        NN, info.groups)
        # Calculate modified flux
        _modified_flux_centers(modified_flux, numerical_flux, xs_total, \
                xs_scatter, xs_fission, external, boundary_x, boundary_y, \
                medium_map, delta_x, delta_y, angle_x, angle_y, angle_w, info)

        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        edges_y = np.insert(np.cumsum(delta_y), 0, 0)
        # Run modified centers
        _curve_fit_modified_centers(modified_flux, curve_fit_flux, \
                curve_fit_boundary_x, curve_fit_boundary_y, int_psi, \
                int_dx, int_dy, int_phi, medium_map, knots_x, knots_y, \
                edges_x, edges_y, x_splits, y_splits, angle_w, block, \
                quintic, info)

    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _residual_integral(residual, int_psi, int_dx, int_dy, int_phi, xs_total, \
                       xs_scatter, xs_fission, external, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, info)
    fangles = str(info.angles).zfill(2)
    fcells = str(info.cells_x).zfill(3)
    np.save(f"nearby_residual_x{fcells}_s{fangles}", np.asarray(residual))
    np.save(f"nearby_boundary_x_x{fcells}_s{fangles}", np.asarray(curve_fit_boundary_x))
    np.save(f"nearby_boundary_y_x{fcells}_s{fangles}", np.asarray(curve_fit_boundary_y))

    # Run Nearby Problem
    print("4. Calculating Nearby Solution...")
    info.edges = 0
    if kwargs.get("zero_bounds", False):
        print("Removing Analytical Boundary Conditions...")
        curve_fit_boundary_x = boundary_x.copy()
        curve_fit_boundary_y = boundary_y.copy()
    nearby_flux = fixed2d.source_iteration(xs_total, xs_scatter, xs_fission, \
                                (external + residual), curve_fit_boundary_x, \
                                curve_fit_boundary_y, medium_map, delta_x, \
                                delta_y, angle_x, angle_y, angle_w, info)

    return numerical_flux, np.asarray(curve_fit_flux), nearby_flux


def fixed_source_residual(scalar_flux, xs_total, xs_scatter, xs_fission, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        knots_x, knots_y, angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = True if kwargs.get("block", True) or (info.materials != 1) else False
    
    # Angular directions
    cdef int NN = info.angles * info.angles
    
    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("1. Calculating Angular Flux Solution...")
    numerical_flux = fixed2d.known_source_calculation(scalar_flux, xs_total, \
                            xs_scatter + xs_fission, external, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, params_dict)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.cells_y, NN, info.groups))
    curve_fit_boundary_y = np.zeros((2, info.cells_x, NN, info.groups))
    curve_fit_flux = np.zeros((info.cells_x, info.cells_y, NN, info.groups))

    # Initialize curve fit integrals
    int_psi = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_phi = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        edges_y = np.insert(np.cumsum(delta_y), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, edges_x, edges_y, x_splits, \
                    y_splits, angle_w, block, quintic, info)
    # Knots at cell edges
    elif knots_x.shape[0] == (info.cells_x + 1):
        centers_x = average_array(knots_x)
        centers_y = average_array(knots_y)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, centers_x, centers_y, \
                    x_splits, y_splits, angle_w, block, quintic, info)
        
    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _residual_integral(residual, int_psi, int_dx, int_dy, int_phi, xs_total, \
                       xs_scatter, xs_fission, external, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
                np.asarray(curve_fit_boundary_y), np.asarray(residual)


def fixed_source_residual_lite(scalar_flux, xs_total, xs_scatter, xs_fission, \
        external, boundary_x, boundary_y, medium_map, delta_x, delta_y, \
        knots_x, knots_y, angle_x, angle_y, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    block = True if kwargs.get("block", True) or (info.materials != 1) else False

    # Angular directions
    cdef int NN = info.angles * info.angles

    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)
    else:
        x_splits = kwargs.get("x_splits")
        y_splits = kwargs.get("y_splits")

    # Run Numerical Solution
    print("Calculating Angular Flux Solution...")
    numerical_flux = fixed2d.known_source_calculation(scalar_flux, xs_total, \
                            xs_scatter + xs_fission, external, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, params_dict)

    # Initialize curve fit and residual
    curve_fit_boundary_x = tools.array_3d(2, info.cells_y, info.groups)
    curve_fit_boundary_y = tools.array_3d(2, info.cells_x, info.groups)
    curve_fit_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    residual = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    print("Calculating Analytical Solution and Residual...")
    # Knots at cell centers
    edges_x = np.insert(np.cumsum(delta_x), 0, 0)
    edges_y = np.insert(np.cumsum(delta_y), 0, 0)
    _curve_fit_centers_residual_lite(numerical_flux, curve_fit_flux, \
            curve_fit_boundary_x, curve_fit_boundary_y, residual, xs_total, \
            xs_scatter, xs_fission, external, medium_map, delta_x, delta_y, \
            knots_x, knots_y, edges_x, edges_y, x_splits, y_splits, angle_x, \
            angle_y, angle_w, block, quintic, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
                np.asarray(curve_fit_boundary_y), np.asarray(residual)


cdef void _curve_fit_centers(double[:,:,:,:]& flux, double[:,:,:,:]& curve_fit, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& integral, double[:,:,:,:]& dxintegral, \
        double[:,:,:,:]& dyintegral, double[:,:,:]& sintegral, \
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
            curve_fit[...,nn,gg] = spline[:,:]

            # Interpolate y boundary
            boundary = approx.interpolate(knots_x, bounds_y)
            boundary_y[...,nn,gg] = boundary[:,:].T

            # Interpolate x boundary
            boundary = approx.interpolate(bounds_x, knots_y)
            boundary_x[...,nn,gg] = boundary[:,:]

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
            integral[...,nn,gg] = int_psi[:,:]
            dxintegral[...,nn,gg] = int_dx[:,:]
            dyintegral[...,nn,gg] = int_dy[:,:]

    # Populate sintegral scalar flux
    tools._angular_to_scalar(integral, sintegral, angle_w, info)


cdef void _modified_flux_centers(double[:,:,:,:]& modified_flux, \
        double[:,:,:,:]& numerical_flux, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:,:,:]& external, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    ####################################################################
    # For angular flux at edges
    info.angular = True
    info.edges = 1
    ####################################################################
    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)

    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)
    scalar_flux = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    tools._source_total(source, scalar_flux, xs_matrix, medium_map, external, info)
    tools._angular_to_scalar(numerical_flux, scalar_flux, angle_w, info)

    # Solve for angular flux cell interfaces
    flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                                 info.angles * info.angles, info.groups)
    flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                                 info.angles * info.angles, info.groups)

    mg._interface_angular(flux_edge_x, flux_edge_y, xs_total, source, \
                         boundary_x, boundary_y, medium_map, delta_x, \
                         delta_y, angle_x, angle_y, angle_w, info)

    # Collect outer edges
    tools._nearby_populate_outer(modified_flux, numerical_flux, \
                                flux_edge_x, flux_edge_y, info)


cdef void _curve_fit_modified_centers(double[:,:,:,:]& flux, \
        double[:,:,:,:]& curve_fit, double[:,:,:,:]& boundary_x, \
        double[:,:,:,:]& boundary_y, double[:,:,:,:]& integral, \
        double[:,:,:,:]& dxintegral, double[:,:,:,:]& dyintegral, \
        double[:,:,:]& sintegral, int[:,:]& medium_map, \
        double[:]& knots_x, double[:]& knots_y, double[:]& edges_x, \
        double[:]& edges_y, int[:]& x_splits, int[:]& y_splits, \
        double[:]& angle_w, bint block, bint quintic, params info):

    # Initialize cell, angle, and group
    cdef int ii, jj, nn, gg

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize angular and group specific interpolations
    cdef double[:,:] spline, int_psi, int_dx, int_dy, boundary
    cdef double[2] bounds_x = [edges_x[0], edges_x[info.cells_x]]
    cdef double[2] bounds_y = [edges_y[0], edges_y[info.cells_y]]

    # Adjust medium map
    modified_medium_map = tools._nearby_adjust_medium(medium_map, info)

    # Iterate over groups
    for gg in range(info.groups):
        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True):
            # Create function
            approx = Interpolation(flux[:,:,nn,gg], knots_x, knots_y, \
                                modified_medium_map, x_splits, y_splits, \
                                block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(knots_x, knots_y)
            curve_fit[...,nn,gg] = tools._nearby_sum_2d(spline, info)

            # Interpolate y boundary
            boundary = approx.interpolate(knots_x, bounds_y)
            boundary_y[...,nn,gg] = tools._nearby_sum_bc(boundary[:,:].T, info)

            # Interpolate x boundary
            boundary = approx.interpolate(bounds_x, knots_y)
            boundary_x[...,nn,gg] = tools._nearby_sum_bc(boundary, info)

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_centers(edges_x, edges_y)
            integral[...,nn,gg] = tools._nearby_sum_2d(int_psi, info)
            dxintegral[...,nn,gg] = tools._nearby_sum_2d(int_dx, info)
            dyintegral[...,nn,gg] = tools._nearby_sum_2d(int_dy, info)

    # Populate sintegral scalar flux
    tools._angular_to_scalar(integral, sintegral, angle_w, info)


cdef void _curve_fit_edges(double[:,:,:,:]& flux, double[:,:,:,:]& curve_fit, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& integral, double[:,:,:,:]& dxintegral, \
        double[:,:,:,:]& dyintegral, double[:,:,:]& sintegral, \
        int[:,:]& medium_map, double[:]& knots_x, double[:]& knots_y, \
        double[:]& centers_x, double[:]& centers_y, int[:]& x_splits, \
        int[:]& y_splits, double[:]& angle_w, bint block, bint quintic, \
        params info):

    # Initialize angle and group
    cdef int ii, jj, nn, gg

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Iterate over groups
    for gg in tqdm(range(info.groups), desc="Curve Fit Groups", ascii=True, position=0):
        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True, position=1, leave=False):
            # Create function
            approx = Interpolation(flux[:,:,nn,gg], knots_x, knots_y, \
                        medium_map, x_splits, y_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(centers_x, centers_y)

            # Calculate integrals
            int_psi, int_dx, int_dy = approx.integrate_edges()

            # Iterate over material areas
            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    curve_fit[ii,jj,nn,gg] = spline[ii,jj]
                    integral[ii,jj,nn,gg] = int_psi[ii,jj]
                    dxintegral[ii,jj,nn,gg] = int_dx[ii,jj]
                    dyintegral[ii,jj,nn,gg] = int_dy[ii,jj]
                    sintegral[ii,jj,gg] += angle_w[nn] * int_psi[ii,jj]

                    # Correct boundary conditions - x direction
                    if ii == 0:
                        boundary_x[0,jj,nn,gg] = approx.interpolate(knots_x[0], centers_y[jj])
                    elif ii == (info.cells_x - 1):
                        boundary_x[1,jj,nn,gg] = approx.interpolate( \
                                        knots_x[info.cells_x], centers_y[jj])

                    # Correct boundary conditions - y direction
                    if jj == 0:
                        boundary_y[0,ii,nn,gg] = approx.interpolate(centers_x[ii], knots_y[0])
                    elif jj == (info.cells_y - 1):
                        boundary_y[1,ii,nn,gg] = approx.interpolate( \
                                        centers_x[ii], knots_y[info.cells_y])
                

cdef void _residual_integral(double[:,:,:,:]& residual, double[:,:,:,:]& psi, \
        double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, double[:,:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:,:]& external, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, params info):

    # Initialize angle, group and cell
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
                    off_scatter += phi[ii,jj,ig] * (xs_scatter[mat,og,ig] \
                                    + xs_fission[mat,og,ig])

                # Iterate over angles
                for nn in range(info.angles * info.angles):
                    residual[ii,jj,nn,og] = (angle_x[nn] * dxpsi[ii,jj,nn,og]) \
                                + (angle_y[nn] * dypsi[ii,jj,nn,og]) - off_scatter \
                                + (psi[ii,jj,nn,og] * xs_total[mat,og]) \
                                - (external[ii,jj,nn,og] * delta_x[ii] * delta_y[jj])


cdef void _curve_fit_centers_residual_lite(double[:,:,:,:]& flux, \
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
    cdef int ii, jj, nn, gg, mat

    # Initialize angular directions
    cdef int NN = info.angles * info.angles

    # Initialize integrals
    int_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dx_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_dy_angular = tools.array_2d(info.cells_x, info.cells_y)
    int_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Initialize group specific interpolations
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
            tools._nearby_boundary_to_scalar(boundary_y, boundary[:,:].T, angle_w[nn], gg, info)

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
                int_dy_angular, xs_total, external[:,:,nn,:], medium_map, \
                delta_x, delta_y, angle_x[nn], angle_y[nn], angle_w[nn], \
                gg, info)

    tools._nearby_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, medium_map, info)


def criticality(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        delta_y, knots_x, knots_y, angle_x, angle_y, angle_w, \
        params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_criticality(info)
    block = True if kwargs.get("block", True) or (info.materials != 1) else False
    
    # Angular directions
    cdef int NN = info.angles * info.angles

    # Check for custom x/y splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)
        y_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("1. Calculating Numerical Solution...")
    numerical_flux, numerical_keff = critical2d.power_iteration(xs_total, \
                    xs_scatter, xs_fission, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.cells_y, NN, info.groups))
    curve_fit_boundary_y = np.zeros((2, info.cells_x, NN, info.groups))
    curve_fit_flux = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)

    # Initialize curve fit integrals
    int_psi = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dx = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_dy = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    int_phi = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Calculate curve fit at knots
    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        edges_y = np.insert(np.cumsum(delta_y), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, edges_x, edges_y, x_splits, \
                    y_splits, angle_w, block, quintic, info)
    
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        centers_y = average_array(knots_y)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                    curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                    medium_map, knots_x, knots_y, centers_x, centers_y, \
                    x_splits, y_splits, angle_w, block, quintic, info)

    # Create curve fit source, curve fit keff, nearby reaction rate
    curve_fit_source = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(int_psi, int_dx, \
                                int_dy, int_phi, xs_total, xs_scatter, \
                                xs_fission, curve_fit_source, medium_map, \
                                delta_x, delta_y, angle_x, angle_y, \
                                angle_w, info)

    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _residual_integral_critical(residual, int_psi, int_dx, int_dy, int_phi, \
            xs_total, xs_scatter, curve_fit_source, medium_map, angle_x, \
            angle_y, curve_fit_keff, info)
    fangles = str(info.angles).zfill(2)
    fcells = str(info.cells_x).zfill(3)
    np.save(f"nearby_residual_x{fcells}_s{fangles}", np.asarray(residual))

    # Run Nearby Problem
    print("4. Calculating Nearby Solution...")
    info.edges = 0
    nearby_scalar, nearby_keff = critical2d.nearby_power(xs_total, xs_scatter, \
                        xs_fission, residual, medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, nearby_rate, info)

    # Convert numerical_flux to scalar flux
    numerical_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

    # Convert curve_fit_flux to scalar flux
    curve_fit_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(curve_fit_flux, curve_fit_scalar, angle_w, info)

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


cdef void _residual_integral_critical(double[:,:,:,:]& residual, \
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