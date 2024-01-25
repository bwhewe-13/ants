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
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

import numpy as np

from ants import fixed1d, critical1d
from ants.utils.interp1d import Interpolation
from ants.utils.pytools import average_array

from ants cimport cytools_1d as tools
from ants cimport parameters
from ants.parameters cimport params


def fixed_source(xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, knots_x, angle_x, angle_w, params_dict, \
        quintic=True, zero_bounds=False):

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_fixed_source(info, xs_total.shape[0])

    # Create boundaries for each material
    splits = tools._material_index(medium_map, info)

    # Run Numerical Solution
    print("1. Calculating Numerical Solution...")
    numerical_flux = fixed1d.source_iteration(xs_total, xs_scatter, \
                            xs_fission, external, boundary_x, medium_map, \
                            delta_x, angle_x, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.angles, info.groups))
    curve_fit_flux = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize curve fit integrals
    psi = tools.array_3d(info.cells_x, info.angles, info.groups)
    dpsi = tools.array_3d(info.cells_x, info.angles, info.groups)
    phi = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                psi, dpsi, phi, splits, knots_x, edges_x, angle_w, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
            psi, dpsi, phi, splits, knots_x, centers_x, angle_w, quintic, info)

    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = tools.array_3d(info.cells_x, info.angles, info.groups)
    _residual_integral(residual, psi, dpsi, phi, xs_total, xs_scatter, \
                       xs_fission, external, medium_map, delta_x, \
                       angle_x, info)
    fangles = str(info.angles).zfill(2)
    np.save(f"nearby_residual_s{fangles}", np.asarray(residual))
    np.save(f"nearby_boundary_x_s{fangles}", np.asarray(curve_fit_boundary_x))

    # Run Nearby Problem
    print("4. Calculating Nearby Solution...")
    info.edges = 0
    if zero_bounds:
        print("Removing Analytical Boundary Conditions...")
        curve_fit_boundary_x = boundary_x.copy()
    nearby_flux = fixed1d.source_iteration(xs_total, xs_scatter, xs_fission, \
                            (external + residual), curve_fit_boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    return numerical_flux, np.asarray(curve_fit_flux), nearby_flux


def residual_fixed(numerical_flux, xs_total, xs_scatter, xs_fission, \
        external, boundary_x, medium_map, delta_x, knots_x, angle_x, \
        angle_w, params_dict, quintic=True):

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_fixed_source(info, xs_total.shape[0])

    # Create boundaries for each material
    splits = tools._material_index(medium_map, info)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.angles, info.groups))
    curve_fit_flux = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize curve fit integrals
    psi = tools.array_3d(info.cells_x, info.angles, info.groups)
    dpsi = tools.array_3d(info.cells_x, info.angles, info.groups)
    phi = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                psi, dpsi, phi, splits, knots_x, edges_x, angle_w, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
            psi, dpsi, phi, splits, knots_x, centers_x, angle_w, quintic, info)

    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = tools.array_3d(info.cells_x, info.angles, info.groups)
    _residual_integral(residual, psi, dpsi, phi, xs_total, xs_scatter, \
                       xs_fission, external, medium_map, delta_x, \
                       angle_x, info)

    return np.asarray(curve_fit_flux), np.asarray(residual), \
            np.asarray(curve_fit_boundary_x)


cdef void _curve_fit_centers(double[:,:,:]& flux, double[:,:,:]& curve_fit, \
        double[:,:,:]& boundary_x, double[:,:,:]& integral, \
        double[:,:,:]& dintegral, double[:,:]& sintegral, int[:]& splits, \
        double[:]& knots_x, double[:]& edges_x, double[:]& angle_w, \
        bint quintic, params info):
    # Initialize angle and group
    cdef int nn, gg, ii, iix, idx1, idx2, local
    # Initialize cell divisions
    cdef int split_length = splits.shape[0] - 1
    # Iterate over groups
    for gg in range(info.groups):
        # Iterate over angles
        for nn in range(info.angles):
            # Iterate over material areas
            for ii in range(split_length):
                # Create material edge index
                idx1 = splits[ii]
                idx2 = splits[ii+1]
                # Create function
                spline = Interpolation(flux[idx1:idx2,nn,gg], \
                                       knots_x[idx1:idx2], quintic)
                psi, dpsi = spline.integrate_centers(edges_x[idx1:idx2+1])
                local = 0
                # Interpolate at cell centers
                for iix in range(idx1, idx2):
                    curve_fit[iix,nn,gg] = spline.interpolate(knots_x[iix])
                    integral[iix,nn,gg] = psi[local]
                    dintegral[iix,nn,gg] = dpsi[local]
                    sintegral[iix,gg] += angle_w[nn] * psi[local]
                    local += 1
                # Correct boundary conditions
                if ii == 0:
                    boundary_x[0,nn,gg] = spline.interpolate(edges_x[0])
                if ii == (split_length - 1):
                    boundary_x[1,nn,gg] = spline.interpolate(edges_x[info.cells_x])


cdef void _curve_fit_edges(double[:,:,:]& flux, double[:,:,:]& curve_fit, \
        double[:,:,:]& boundary_x, double[:,:,:]& integral, \
        double[:,:,:]& dintegral, double[:,:]& sintegral, int[:]& splits, \
        double[:]& knots_x, double[:]& centers_x, double[:]& angle_w, \
        bint quintic, params info):
    # Initialize angle and group
    cdef int nn, gg, ii, iix, idx1, idx2
    # Initialize integral terms
    cdef double psi, dpsi
    # Initialize cell divisions
    cdef int split_length = splits.shape[0] - 1
    # Iterate over groups
    for gg in range(info.groups):
        # Iterate over angles
        for nn in range(info.angles):
            # Iterate over material areas
            for ii in range(split_length):
                # Create material edge index
                idx1 = splits[ii]
                idx2 = splits[ii+1] + 1
                # Create function
                spline = Interpolation(flux[idx1:idx2,nn,gg], \
                                       knots_x[idx1:idx2], quintic)
                # Interpolate at cell centers
                for iix in range(idx1, idx2 - 1):
                    curve_fit[iix,nn,gg] = spline.interpolate(centers_x[iix])
                    psi, dpsi = spline.integrate_edge(knots_x[iix], knots_x[iix+1])
                    integral[iix,nn,gg] = psi
                    dintegral[iix,nn,gg] = dpsi
                    sintegral[iix,gg] += angle_w[nn] * psi
                # Correct boundary conditions
                if ii == 0:
                    boundary_x[0,nn,gg] = spline.interpolate(0.0)
                if ii == (split_length - 1):
                    boundary_x[1,nn,gg] = spline.interpolate(knots_x[info.cells_x])
                

cdef void _residual_integral(double[:,:,:]& residual, double[:,:,:]& psi, \
        double[:,:,:]& dpsi, double[:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        double[:,:,:]& external, int[:]& medium_map, double[:]& delta_x, \
        double[:]& angle_x, params info):

    # Initialize angle, group and cell
    cdef int ii, nn, og, ig, mat

    # Initialize off-scattering term
    cdef float off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(info.groups):
            off_scatter = 0.0
            for ig in range(info.groups):
                off_scatter += phi[ii,ig] * (xs_scatter[mat,og,ig] \
                                + xs_fission[mat,og,ig])

            # Iterate over angles
            for nn in range(info.angles):
                residual[ii,nn,og] = (angle_x[nn] * dpsi[ii,nn,og] \
                        + psi[ii,nn,og] * xs_total[mat,og]) - off_scatter \
                        - external[ii,nn,og] * delta_x[ii]


def criticality(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        knots_x, angle_x, angle_w, params_dict, quintic=True):

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_criticality(info)

    # Create boundaries for each material
    splits = tools._material_index(medium_map, info)

    # Run Numerical Solution
    print("1. Calculating Numerical Solution...")
    numerical_flux, numerical_keff = critical1d.power_iteration(xs_total, \
                                    xs_scatter, xs_fission, medium_map, \
                                    delta_x, angle_x, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = tools.array_3d(2, info.angles, info.groups)
    curve_fit_flux = tools.array_3d(info.cells_x, info.angles, info.groups)

    # Initialize curve fit integrals
    psi = tools.array_3d(info.cells_x, info.angles, info.groups)
    dpsi = tools.array_3d(info.cells_x, info.angles, info.groups)
    phi = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("2. Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _curve_fit_centers(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
                psi, dpsi, phi, splits, knots_x, edges_x, angle_w, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _curve_fit_edges(numerical_flux, curve_fit_flux, curve_fit_boundary_x, \
            psi, dpsi, phi, splits, knots_x, centers_x, angle_w, quintic, info)

    # Create curve fit source, curve fit keff, nearby reaction rate
    curve_fit_source = tools.array_2d(info.cells_x, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(psi, dpsi, \
                                phi, xs_total, xs_scatter, xs_fission, \
                                curve_fit_source, medium_map, delta_x, \
                                angle_x, angle_w, info)

    # Calculate residual for each cell
    print("3. Calculating Residual...")
    residual = np.zeros((info.cells_x, info.angles, info.groups))
    _residual_integral_critical(residual, psi, dpsi, phi, xs_total, \
                                xs_scatter, curve_fit_source, medium_map, \
                                angle_x, curve_fit_keff, info)
    fangles = str(info.angles).zfill(2)
    np.save(f"nearby_residual_s{fangles}", np.asarray(residual))

    # Run Nearby Problem
    print("4. Calculating Nearby Solution...")
    info.edges = 0

    nearby_scalar, nearby_keff = critical1d.nearby_power(xs_total, xs_scatter, \
                                    xs_fission, residual, medium_map, delta_x, \
                                    angle_x, angle_w, nearby_rate, info)

    # Convert numerical_flux to scalar flux
    numerical_scalar = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)
    # Convert curve_fit_flux to scalar flux
    curve_fit_scalar = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(curve_fit_flux, curve_fit_scalar, angle_w, info)

    return numerical_scalar, numerical_keff, np.asarray(curve_fit_scalar), \
            curve_fit_keff, nearby_scalar, nearby_keff


cdef (double, double) _curve_fit_fission_source(double[:,:,:]& psi, \
        double[:,:,:]& dpsi, double[:,:]& phi, double[:,:]& xs_total, \
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
                right_hand_off += phi[ii,ig] * xs_fission[mat,og,ig]
                left_hand_off += phi[ii,ig] * xs_scatter[mat,og,ig]

            # Create curve fit source with only one angle
            fission_source[ii,og] = right_hand_off

            # Update nearby fission rate
            nearby_rate += right_hand_off / delta_x[ii]

            # Iterate over angles
            for nn in range(info.angles):
                right_hand += angle_w[nn] * right_hand_off
                left_hand += angle_w[nn] * (angle_x[nn] * dpsi[ii,nn,og] \
                            + psi[ii,nn,og] * xs_total[mat,og] - left_hand_off)

    curve_fit_keff = right_hand / left_hand
    return nearby_rate, curve_fit_keff


cdef void _residual_integral_critical(double[:,:,:]& residual, \
        double[:,:,:]& psi, double[:,:,:]& dpsi, double[:,:]& phi, \
        double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:,:]& source, int[:]& medium_map, double[:]& angle_x, \
        double keff, params info):

    # Initialize cell, angle, and group iterables
    cdef int ii, nn, og, ig, mat

    # Initialize off-scattering term
    cdef float off_scatter

    # Iterate over spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(info.groups):
            off_scatter = 0.0
            for ig in range(info.groups):
                off_scatter += phi[ii,ig] * xs_scatter[mat,og,ig]

            # Iterate over angles
            for nn in range(info.angles):
                residual[ii,nn,og] = (angle_x[nn] * dpsi[ii,nn,og] \
                                    + psi[ii,nn,og] * xs_total[mat,og]) \
                                    - off_scatter - source[ii,og] / keff