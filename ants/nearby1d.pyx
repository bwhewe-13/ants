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
from tqdm.auto import tqdm

from ants import fixed1d, critical1d
from ants.utils.interp1d import Interpolation
from ants.utils.pytools import average_array

from ants cimport cytools_1d as tools
from ants cimport parameters
from ants.parameters cimport params


def fixed_source(xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, knots_x, angle_x, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Numerical Solution...")
    numerical_flux = fixed1d.source_iteration(xs_total, xs_scatter, \
                            xs_fission, external, boundary_x, medium_map, \
                            delta_x, angle_x, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = tools.array_3d(2, info.angles, info.groups)
    curve_fit_flux = tools.array_2d(info.cells_x, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_dx_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_scalar = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("Calculating Angular Cuvre Fit Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _angular_curve_fit_centers(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, edges_x, \
                angle_w, block, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _angular_curve_fit_edges(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, centers_x, \
                angle_w, block, quintic, info)

    # Calculate residual for each cell
    print("Calculating Angular Residual...")
    residual = tools.array_3d(info.cells_x, info.angles, info.groups)
    _angular_residual(residual, int_angular, int_dx_angular, int_scalar, \
            xs_total, xs_scatter, xs_fission, external, medium_map, \
            delta_x, angle_x, info)

    fangles = str(info.angles).zfill(2)
    fcells = str(info.cells_x).zfill(3)
    np.save(f"nearby_residual_x{fcells}_n{fangles}", np.asarray(residual))
    np.save(f"nearby_boundary_x_x{fcells}_n{fangles}", np.asarray(curve_fit_boundary_x))

    # Run Nearby Problem
    print("Calculating Nearby Solution...")
    info.edges = 0
    if kwargs.get("zero_bounds", False):
        print("Removing Analytical Boundary Conditions...")
        curve_fit_boundary_x = boundary_x.copy()
    nearby_flux = fixed1d.source_iteration(xs_total, xs_scatter, xs_fission, \
                            (external + residual), curve_fit_boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, info)

    return numerical_flux, np.asarray(curve_fit_flux), nearby_flux


def fixed_source_angular_residual(scalar_flux, xs_total, xs_scatter, \
        xs_fission, external, boundary_x, medium_map, delta_x, knots_x, \
        angle_x, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Angular Flux Solution...")
    numerical_flux = fixed1d.known_source_calculation(scalar_flux, xs_total, \
                            xs_scatter + xs_fission, external, boundary_x, \
                            medium_map, delta_x, angle_x, angle_w, params_dict)

    # Initialize curve fit
    curve_fit_boundary_x = np.zeros((2, info.angles, info.groups))
    curve_fit_flux = tools.array_2d(info.cells_x, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_dx_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_scalar = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("Calculating Angular Curve Fit Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _angular_curve_fit_centers(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, edges_x, \
                angle_w, block, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _angular_curve_fit_edges(numerical_flux, curve_fit_flux, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, centers_x, \
                angle_w, block, quintic, info)

    # Calculate residual for each cell
    print("Calculating Angular Residual...")
    residual = tools.array_3d(info.cells_x, info.angles, info.groups)
    _angular_residual(residual, int_angular, int_dx_angular, int_scalar, \
            xs_total, xs_scatter, xs_fission, external, medium_map, \
            delta_x, angle_x, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
            np.asarray(residual)


def fixed_source_scalar_residual(scalar_flux, xs_total, xs_scatter, \
        xs_fission, external, boundary_x, medium_map, delta_x, knots_x, \
        angle_x, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_fixed_source(info, xs_total.shape[0])
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Angular Flux Solution...")
    numerical_flux = fixed1d.known_source_calculation(scalar_flux, \
                            xs_total, xs_scatter + xs_fission, external, \
                            boundary_x, medium_map, delta_x, angle_x, \
                            angle_w, params_dict)

    # Initialize curve fit and residual
    curve_fit_boundary_x = tools.array_2d(2, info.groups)
    curve_fit_flux = tools.array_2d(info.cells_x, info.groups)
    residual = tools.array_2d(info.cells_x, info.groups)

    # Knots at cell centers
    edges_x = np.insert(np.cumsum(delta_x), 0, 0)

    print("Calculating Scalar Curve Fit Solution and Residual...")
    _scalar_curve_fit_residual(numerical_flux, curve_fit_flux, \
            curve_fit_boundary_x, residual, xs_total, xs_scatter, \
            xs_fission, external, medium_map, delta_x, knots_x, edges_x, \
            x_splits, angle_x, angle_w, block, quintic, info)

    return np.asarray(curve_fit_flux), np.asarray(curve_fit_boundary_x), \
            np.asarray(residual)
            

cdef void _angular_curve_fit_centers(double[:,:,:]& flux, \
        double[:,:]& curve_fit, double[:,:,:]& boundary_x, \
        double[:,:,:]& int_angular, double[:,:,:]& int_dx_angular, \
        double[:,:]& int_scalar, int[:]& medium_map, int[:]& x_splits, \
        double[:]& knots_x, double[:]& edges_x, double[:]& angle_w, \
        bint block, bint quintic, params info):

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
            spline = approx.interpolate(knots_x)
            # curve_fit[:,nn,gg] = spline[:]
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate x boundary
            boundary_x[0,nn,gg] = approx.interpolate(edges_x[0])
            boundary_x[1,nn,gg] = approx.interpolate(edges_x[info.cells_x])

            # Calculate integrals
            int_psi, int_dx = approx.integrate_centers(edges_x)
            int_angular[:,nn,gg] = int_psi[:]
            int_dx_angular[:,nn,gg] = int_dx[:]

    # Populate int_scalar scalar flux
    tools._angular_to_scalar(int_angular, int_scalar, angle_w, info)


cdef void _angular_curve_fit_edges(double[:,:,:]& flux, \
        double[:,:]& curve_fit, double[:,:,:]& boundary_x, \
        double[:,:,:]& int_angular, double[:,:,:]& int_dx_angular, \
        double[:,:]& int_scalar, int[:]& medium_map, int[:]& x_splits, \
        double[:]& knots_x, double[:]& centers_x, double[:]& angle_w, \
        bint block, bint quintic, params info):

    # Initialize angle and group
    cdef int ii, nn, gg

    # Initialize integral terms
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
            spline = approx.interpolate(centers_x)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate x boundary
            boundary_x[0,nn,gg] = approx.interpolate(knots_x[0])
            boundary_x[1,nn,gg] = approx.interpolate(knots_x[info.cells_x])

            # Calculate integrals
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
    cdef float off_scatter

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
    cdef int ii, nn, nn_q, gg

    # Initialize integrals
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

            # Create function
            approx = Interpolation(flux[:,nn,gg], knots_x, medium_map, \
                                   x_splits, block, quintic)

            # Interpolate the knots
            spline = approx.interpolate(knots_x)
            tools._nearby_flux_to_scalar(curve_fit, spline, angle_w[nn], gg, info)

            # Interpolate x boundary
            boundary_x[0,gg] += approx.interpolate(edges_x[0]) * angle_w[nn]
            boundary_x[1,gg] += approx.interpolate(edges_x[info.cells_x]) * angle_w[nn]

            # Calculate integrals
            int_psi, int_dx = approx.integrate_centers(edges_x)
            int_angular = int_psi[:]
            int_dx_angular = int_dx[:]
            tools._nearby_flux_to_scalar(int_scalar, int_psi, angle_w[nn], gg, info)

            # Update Residual - On scatter
            tools._nearby_on_scatter(residual, int_angular, int_dx_angular, \
                    xs_total, external[:,nn_q,:], medium_map, delta_x, \
                    angle_x[nn], angle_w[nn], gg, gg, info)

    tools._nearby_off_scatter(residual, int_scalar, xs_scatter, \
            xs_fission, medium_map, info)


def criticality(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        knots_x, angle_x, angle_w, params_dict, **kwargs):

    # Keyword arguments
    quintic = kwargs.get("quintic", True)

    # Convert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_nearby1d_criticality(info)
    block = False if (info.materials == 1) else kwargs.get("block", True)

    # Check for custom x splits
    if kwargs.get("x_splits", None) is None:
        x_splits = np.zeros((0,), dtype=np.int32)

    # Run Numerical Solution
    print("Calculating Numerical Solution...")
    numerical_flux, numerical_keff = critical1d.power_iteration(xs_total, \
                                    xs_scatter, xs_fission, medium_map, \
                                    delta_x, angle_x, angle_w, info)

    # Initialize curve fit
    curve_fit_boundary_x = tools.array_3d(2, info.angles, info.groups)
    curve_fit_scalar = tools.array_2d(info.cells_x, info.groups)

    # Initialize curve fit integrals
    int_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_dx_angular = tools.array_3d(info.cells_x, info.angles, info.groups)
    int_scalar = tools.array_2d(info.cells_x, info.groups)

    # Calculate curve fit at knots
    print("Calculating Analytical Solution...")
    # Knots at cell centers
    if knots_x.shape[0] == info.cells_x:
        edges_x = np.insert(np.cumsum(delta_x), 0, 0)
        _angular_curve_fit_centers(numerical_flux, curve_fit_scalar, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, edges_x, \
                angle_w, block, quintic, info)
    # Knots at cell edges
    else:
        centers_x = average_array(knots_x)
        _angular_curve_fit_edges(numerical_flux, curve_fit_scalar, \
                curve_fit_boundary_x, int_angular, int_dx_angular, \
                int_scalar, medium_map, x_splits, knots_x, centers_x, \
                angle_w, block, quintic, info)

    # Create curve fit source, curve fit keff, nearby reaction rate
    curve_fit_source = tools.array_2d(info.cells_x, info.groups)
    nearby_rate, curve_fit_keff = _curve_fit_fission_source(int_angular, \
                            int_dx_angular, int_scalar, xs_total, xs_scatter, \
                            xs_fission, curve_fit_source, medium_map, delta_x, \
                            angle_x, angle_w, info)

    # Calculate residual for each cell
    print("Calculating Residual...")
    residual = np.zeros((info.cells_x, info.angles, info.groups))
    _angular_residual_critical(residual, int_angular, int_dx_angular, \
            int_scalar, xs_total, xs_scatter, curve_fit_source, medium_map, \
            angle_x, curve_fit_keff, info)
    fangles = str(info.angles).zfill(2)
    np.save(f"nearby_residual_n{fangles}", np.asarray(residual))

    # Run Nearby Problem
    print("Calculating Nearby Solution...")
    info.edges = 0
    nearby_scalar, nearby_keff = critical1d.nearby_power(xs_total, xs_scatter, \
                                    xs_fission, residual, medium_map, delta_x, \
                                    angle_x, angle_w, nearby_rate, info)

    # Convert numerical_flux to scalar flux
    numerical_scalar = tools.array_2d(info.cells_x, info.groups)
    tools._angular_to_scalar(numerical_flux, numerical_scalar, angle_w, info)

    return numerical_scalar, numerical_keff, np.asarray(curve_fit_scalar), \
            curve_fit_keff, nearby_scalar, nearby_keff


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
    cdef float off_scatter

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