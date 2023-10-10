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
from ants.utils.interp2d import BlockInterpolation as Block
from ants.utils.interp2d import QuinticHermite as Hermite

from ants cimport cytools_2d as tools
from ants cimport parameters
from ants.parameters cimport params


def fixed_source(xs_total, xs_scatter, xs_fission, external, boundary_x, \
        boundary_y, medium_map, delta_x, delta_y, knots_x, knots_y, \
        angle_x, angle_y, angle_w, params_dict, zero_bounds=False):
    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_fixed_source(info, xs_total.shape[0])
    # Angular directions
    cdef int NN = info.angles * info.angles
    print("1. Calculating Numerical Solution...")
    # Run Numerical Solution
    numerical_flux = fixed2d.source_iteration(xs_total, xs_scatter, \
                        xs_fission, external.flatten(), boundary_x.flatten(), \
                        boundary_y.flatten(), medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, info)
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
                curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, medium_map, \
                knots_x, knots_y, edges_x, edges_y, angle_w, info)
    # Knots at cell edges
    else:
        centers_x = 0.5 * (knots_x[1:] + knots_x[:-1])
        centers_y = 0.5 * (knots_y[1:] + knots_y[:-1])
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                medium_map, knots_x, knots_y, centers_x, centers_y, angle_w, info)
    print("3. Calculating Residual...")
    # Calculate residual for each cell
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _residual_integral(residual, int_psi, int_dx, int_dy, int_phi, xs_total, \
                       xs_scatter, xs_fission, external, medium_map, \
                       delta_x, delta_y, angle_x, angle_y, info)
    np.save(f"nearby_residual_s{info.angles}", np.asarray(residual))
    np.save(f"nearby_boundary_x_s{info.angles}", np.asarray(curve_fit_boundary_x))
    np.save(f"nearby_boundary_y_s{info.angles}", np.asarray(curve_fit_boundary_y))
    # Run Nearby Problem
    print("4. Calculating Nearby Solution...")
    info.edges = 0
    if zero_bounds:
        print("Removing Analytical Boundary Conditions...")
        curve_fit_boundary_x = boundary_x.copy()
        curve_fit_boundary_y = boundary_y.copy()
    nearby_flux = fixed2d.source_iteration(xs_total, xs_scatter, \
                        xs_fission, (external + residual).flatten(), \
                        curve_fit_boundary_x.flatten(), \
                        curve_fit_boundary_y.flatten(), medium_map, \
                        delta_x, delta_y, angle_x, angle_y, angle_w, info)
    return numerical_flux, np.asarray(curve_fit_flux), nearby_flux


cdef void _curve_fit_centers(double[:,:,:,:]& flux, double[:,:,:,:]& curve_fit, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& integral, double[:,:,:,:]& dxintegral, \
        double[:,:,:,:]& dyintegral, double[:,:,:]& sintegral, \
        int[:,:]& medium_map, double[:]& knots_x, double[:]& knots_y, \
        double[:]& edges_x, double[:]& edges_y, double[:]& angle_w, \
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
    for gg in range(info.groups):
        # Iterate over angles
        for nn in tqdm(range(NN), desc="Curve Fit Angles", ascii=True):
        # for nn in range(NN):
            # Create function
            if info.materials == 1:
                approx = Hermite(flux[:,:,nn,gg], knots_x, knots_y)
            else:
                approx = Block(Hermite, flux[:,:,nn,gg], knots_x, \
                                knots_y, medium_map)
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
    tools._angular_to_scalar(integral, sintegral, angle_w, info)


cdef void _curve_fit_edges(double[:,:,:,:]& flux, double[:,:,:,:]& curve_fit, \
        double[:,:,:,:]& boundary_x, double[:,:,:,:]& boundary_y, \
        double[:,:,:,:]& integral, double[:,:,:,:]& dxintegral, \
        double[:,:,:,:]& dyintegral, double[:,:,:]& sintegral, \
        int[:,:]& medium_map, double[:]& knots_x, double[:]& knots_y, \
        double[:]& centers_x, double[:]& centers_y, double[:]& angle_w, \
        params info):
    # Initialize angle and group
    cdef int ii, jj, nn, gg
    # Initialize angular directions
    cdef int NN = info.angles * info.angles
    # Iterate over groups
    for gg in range(info.groups):
        # Iterate over angles
        for nn in range(NN):
            # Create function
            approx = Block(Hermite, flux[:,:,nn,gg], knots_x, knots_y, medium_map)
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
    # Initialize angular directions
    cdef int NN = info.angles * info.angles
    # Initialize off-scattering term
    cdef float off_scatter
    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over angles
            for nn in range(info.angles):
                # Iterate over groups
                for og in range(info.groups):
                    off_scatter = 0.0
                    for ig in range(info.groups):
                        off_scatter += phi[ii,jj,ig] * (xs_scatter[mat,og,ig] \
                                        + xs_fission[mat,og,ig])
                    residual[ii,jj,nn,og] = (angle_x[nn] * dxpsi[ii,jj,nn,og]) \
                                + (angle_y[nn] * dypsi[ii,jj,nn,og]) - off_scatter \
                                + (psi[ii,jj,nn,og] * xs_total[mat,og]) \
                                - (external[ii,jj,nn,og] * delta_x[ii] * delta_y[jj])


def criticality(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        delta_y, knots_x, knots_y, angle_x, angle_y, angle_w, params_dict):
    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby2d_criticality(info)
    # Angular directions
    cdef int NN = info.angles * info.angles
    # Run Numerical Solution
    info.qdim = 2
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
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        edges_y = np.insert(np.cumsum(delta_y), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, medium_map, \
                knots_x, knots_y, edges_x, edges_y, angle_w, info)
    # Knots at cell edges
    else:
        centers_x = 0.5 * (knots_x[1:] + knots_x[:-1])
        centers_y = 0.5 * (knots_y[1:] + knots_y[:-1])
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                curve_fit_boundary_y, int_psi, int_dx, int_dy, int_phi, \
                medium_map, knots_x, knots_y, centers_x, centers_y, angle_w, info)
    # Create curve fit source, curve fit keff, nearby reaction rate
    curve_fit_source = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(int_psi, int_dx, \
                                int_dy, int_phi, xs_total, xs_scatter, \
                                xs_fission, curve_fit_source, medium_map, \
                                delta_x, delta_y, angle_x, angle_y, \
                                angle_w, info)
    # Calculate residual for each cell
    residual = tools.array_4d(info.cells_x, info.cells_y, NN, info.groups)
    _residual_integral_critical(residual, int_psi, int_dx, int_dy, int_phi, \
            xs_total, xs_scatter, curve_fit_source, medium_map, angle_x, \
            angle_y, curve_fit_keff, info)
    # Run Nearby Problem
    info.edges = 0
    info.qdim = 3
    info.angular = False
    nearby_scalar, nearby_keff = critical2d.nearby_power(xs_total, xs_scatter, \
                    xs_fission, residual.flatten(), medium_map, delta_x, \
                    delta_y, angle_x, angle_y, angle_w, nearby_rate, info)
    # Convert numerical_flux to scalar flux
    numerical_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)
    # Convert curve_fit_flux to scalar flux
    curve_fit_scalar = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    tools._angular_to_scalar(curve_fit_flux, curve_fit_scalar, angle_w, info)
    # Normalize nearby and curve fit fluxes
    # tools._normalize_flux(numerical_scalar, info)
    # tools._normalize_flux(nearby_scalar, info)
    # tools._normalize_flux(curve_fit_scalar, info)
    # Return numerical, curve fit, and nearby data
    return numerical_scalar, numerical_keff, np.asarray(curve_fit_scalar), \
            curve_fit_keff, nearby_scalar, nearby_keff


cdef (double, double) _curve_fit_fission_source(double[:,:,:,:]& psi, \
        double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, double[:,:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& xs_fission, double[:,:,:]& fission_source, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    # Initialize cell, angle, and group iterables
    cdef int ii, jj, nn, og, ig, mat
    # Angular directions
    cdef int NN = info.angles * info.angles
    # Initialize needed terms
    cdef double nearby_rate = 0.0
    cdef double curve_fit_keff = 0.0
    cdef double left_hand = 0.0
    cdef double right_hand = 0.0
    cdef double left_hand_off, right_hand_off, rate_temp
    # Iterate over cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over angles
            for nn in range(NN):
                rate_temp = 0.0
                # Iterate over groups
                for og in range(info.groups):
                    right_hand_off = 0.0
                    left_hand_off = 0.0
                    for ig in range(info.groups):
                        right_hand_off += phi[ii,jj,ig] * xs_fission[mat,og,ig]
                        left_hand_off += phi[ii,jj,ig] * xs_scatter[mat,og,ig]
                    right_hand += angle_w[nn] * right_hand_off
                    left_hand += angle_w[nn] * (angle_x[nn] * dxpsi[ii,jj,nn,og] \
                                + angle_y[nn] * dypsi[ii,jj,nn,og] \
                                + psi[ii,jj,nn,og] * xs_total[mat,og] \
                                - left_hand_off)
                    # Create curve fit source with only one angle
                    if (nn == 0):
                        fission_source[ii,jj,og] = right_hand_off
                    rate_temp += right_hand_off / (delta_x[ii] * delta_y[jj])
                # Update nearby fission rate
                if (nn == 0):
                    nearby_rate += rate_temp
    curve_fit_keff = right_hand / left_hand
    return nearby_rate, curve_fit_keff


cdef void _residual_integral_critical(double[:,:,:,:]& residual, \
        double[:,:,:,:]& psi, double[:,:,:,:]& dxpsi, double[:,:,:,:]& dypsi, \
        double[:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:,:]& source, int[:,:]& medium_map, double[:]& angle_x, \
        double[:]& angle_y, double keff, params info):
    # Initialize cell, angle, and group iterables
    cdef int ii, jj, nn, og, ig, mat
    # Angular directions
    cdef int NN = info.angles * info.angles
    # Initialize off-scattering term
    cdef float off_scatter
    # Iterate over spatial cells
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            # Iterate over angles
            for nn in range(NN):
                # Iterate over groups
                for og in range(info.groups):
                    off_scatter = 0.0
                    for ig in range(info.groups):
                        off_scatter += phi[ii,jj,ig] * xs_scatter[mat,og,ig]
                    residual[ii,jj,nn,og] = (angle_x[nn] * dxpsi[ii,jj,nn,og] \
                            + angle_y[nn] * dypsi[ii,jj,nn,og]
                            + psi[ii,jj,nn,og] * xs_total[mat,og]) \
                            - off_scatter - source[ii,jj,og] / keff