########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Spatial sweeps for one-dimensional neutron transport problems.
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

from libc.math cimport pow, fabs, tanh, M_PI
# from cython.parallel import prange

from ants cimport cytools_1d as tools
from ants.parameters cimport params


cdef void discrete_ordinates(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:,:]& external, double[:,:]& boundary_x, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        params info):
    # One-dimensional slab
    if info.geometry == 1:
        slab_ordinates(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                       external, boundary_x, medium_map, delta_x, angle_x, \
                       angle_w, info)
    # One-dimensional sphere
    elif info.geometry == 2:
        sphere_ordinates(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, boundary_x, medium_map, delta_x, \
                         angle_x, angle_w, info)


cdef void slab_ordinates(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:,:]& external, double[:,:]& boundary_x, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        params info):
    
    # Initialize external and boundary indices, iterables
    cdef int nn, qq, bc
    
    # Initialize unknown cell edge
    cdef double edge = 0.0
    
    # Add reflector array
    reflector = tools.array_1d(info.angles)
    
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    
    # Iterate over angles until converged
    while not (converged):
    
        # Zero out the scalar flux
        flux[:] = 0.0
    
        # Zero out reflector collector
        reflector[:] = 0.0
    
        # Iterate over angles
        for nn in range(info.angles):
            
            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary_x.shape[1] == 1 else nn

            # Perform spatial sweep
            edge = slab_sweep(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                        external[:,qq], boundary_x[:,bc], medium_map, \
                        delta_x, angle_x[nn], angle_w[nn], reflector[nn], info)
            
            # Update reflected direction
            reflector_corrector(reflector, angle_x, edge, nn, info)
        
        # Calculate L2 change
        change = tools.angle_convergence(flux, flux_old, info)
        # Check for convergence
        converged = (change < info.change_nn) or (count >= info.count_nn)
        # Add one to the iteration count
        count += 1
        
        # Update flux_old
        flux_old[:] = flux[:]


cdef void reflector_corrector(double[:]& reflector, double[:]& angle_x, \
        double edge, int angle, params info):
    # Reflected angle index
    cdef int reflected_idx = info.angles - angle - 1
    # Add edge to boundary condition if reflected surface
    if ((angle_x[angle] > 0.0) and (info.bc_x[1] == 1)) \
            or ((angle_x[angle] < 0.0) and (info.bc_x[0] == 1)):
        reflector[reflected_idx] = edge


cdef double slab_sweep(double[:]& flux, double[:]& flux_old, double[:]& xs_total, \
        double[:]& xs_scatter, double[:]& off_scatter, double[:]& external, \
        double[:]& boundary_x, int[:]& medium_map, double[:]& delta_x, \
        double angle_x, double angle_w, double edge1, params info):
    # Iterate from 0 -> I
    if angle_x > 0.0:
        edge1 += boundary_x[0]
        return slab_forward(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                    external, edge1, medium_map, delta_x, angle_x, angle_w, info)
    # Iterate from I -> 0
    elif angle_x < 0.0:
        edge1 += boundary_x[1]
        return slab_backward(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                    external, edge1, medium_map, delta_x, angle_x, angle_w, info)


cdef float spatial_coef(int spatial):
    if (spatial == 1):
        return 1.0
    elif (spatial == 2):
        return 0.0
    return 0.0


cdef double slab_forward(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge1, int[:]& medium_map, \
        double[:]& delta_x, double angle_x, double angle_w, params info):
    # Initialize cell and material iteration index
    cdef int ii, mat
    # Initialize unknown cell edge
    cdef double edge2 = 0.0
    # Initialize discretization constants
    cdef double tau = 0.0
    cdef float alpha1 = 0.5 * (1.0 - spatial_coef(info.spatial))
    cdef float alpha2 = 0.5 * (1.0 + spatial_coef(info.spatial))
    # Determine flux edge
    if info.edges:
        flux[0] += angle_w * edge1
    # Iterate over cells from 0 -> I, edge1 is known
    for ii in range(info.cells_x):
        # For determining the material cross sections
        mat = medium_map[ii]
        if info.spatial == 3:
            tau = xs_total[mat] * delta_x[ii] / angle_x
            alpha1 = 0.5 * (1.0 - (1.0 / tanh(0.5 * tau) - 2.0 / tau))
            alpha2 = 0.5 * (1.0 + (1.0 / tanh(0.5 * tau) - 2.0 / tau))
        # Calculate cell edge unknown
        edge2 = (xs_scatter[mat] * flux_old[ii] + external[ii] + off_scatter[ii] \
                + edge1 * (fabs(angle_x) / delta_x[ii] - alpha1 * xs_total[mat])) \
                * 1 / (fabs(angle_x) / delta_x[ii] + alpha2 * xs_total[mat])
        # Update flux with cell edges
        if info.edges:
            flux[ii+1] += angle_w * edge2
        # Update flux with cell centers
        else:
            flux[ii] += angle_w * (alpha1 * edge1 + edge2 * alpha2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = I
    return edge1


cdef double slab_backward(double[:]& flux, double[:]& flux_old, double[:]& xs_total, \
        double[:]& xs_scatter, double[:]& off_scatter, double[:]& external, \
        double edge1, int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, params info):
    # Initialize cell and material iterables
    cdef int ii, mat
    # Initialize unknown cell edges
    cdef double edge2 = 0.0
    # Initialize discretization constants
    cdef double tau = 0.0
    cdef float alpha1 = 0.5 * (1.0 - spatial_coef(info.spatial))
    cdef float alpha2 = 0.5 * (1.0 + spatial_coef(info.spatial))
    # Determine flux edge
    if info.edges:
        flux[info.cells_x] += angle_w * edge1
    # Iterate over cells from I -> 0, edge1 is known
    for ii in range(info.cells_x-1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]
        # Step Characteristic
        if info.spatial == 3:
            tau = xs_total[mat] * delta_x[ii] / (angle_x)
            alpha1 = 0.5 * (1.0 - (1.0 / tanh(0.5 * tau) - 2.0 / tau))
            alpha2 = 0.5 * (1.0 + (1.0 / tanh(0.5 * tau) - 2.0 / tau))
        # Calculate cell edge unknown
        edge2 = (xs_scatter[mat] * flux_old[ii] + external[ii] + off_scatter[ii] \
                + edge1 * (fabs(angle_x) / delta_x[ii] - alpha1 * xs_total[mat])) \
                * 1 / (fabs(angle_x) / delta_x[ii] + alpha2 * xs_total[mat])
        # Update flux with cell edges
        if info.edges:
            flux[ii] += angle_w * edge2
        # Update flux with cell centers
        else:
            flux[ii] += angle_w * (alpha1 * edge1 + edge2 * alpha2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = 0
    return edge1


cdef void sphere_ordinates(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:,:]& external, double[:,:]& boundary_x, int[:]& medium_map, \
        double[:]& delta_x, double[:]& angle_x, double[:]& angle_w, \
        params info):
    # Initialize external and boundary indices, iterables
    cdef int nn, qq, bc
    # Initialize sphere specific terms
    cdef double angle_minus, angle_plus, tau
    cdef double alpha_minus, alpha_plus
    half_angle = tools.array_1d(info.cells_x)
    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0
    # Iterate over angles until converged
    while not (converged):
        # Initialize the half angle coefficient
        angle_minus = -1.0
        # Initialize the angular differencing coefficient
        alpha_minus = 0.0
        # Zero out the flux
        flux[:] = 0.0
        # Calculate the initial half angle
        initialize_half_angle(flux_old, half_angle, xs_total, xs_scatter, \
                              off_scatter, external[:,0], medium_map, \
                              delta_x, boundary_x[1,0], info)
        # Iterate over all the discrete ordinates
        for nn in range(info.angles):
            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary_x.shape[1] == 1 else nn
            # Calculate the half angle coefficient
            angle_plus = angle_minus + 2 * angle_w[nn]
            # Calculate the weighted diamond
            tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
            # Calculate the angular differencing coefficient
            alpha_plus = angle_coef_corrector(alpha_minus, angle_x[nn], \
                                              angle_w[nn], nn, info)
            # Iterate over spatial cells
            sphere_sweep(flux, flux_old, half_angle, xs_total, xs_scatter, \
                         off_scatter, external[:,qq], boundary_x[:,bc], \
                         medium_map, delta_x, angle_x[nn], angle_w[nn], \
                         angle_w[nn], tau, alpha_plus, alpha_minus, info)
            # Update the angular differencing coefficient
            alpha_minus = alpha_plus
            # Update the half angle
            angle_minus = angle_plus
        change = tools.angle_convergence(flux, flux_old, info)
        converged = (change < info.change_nn) or (count >= info.count_nn)
        count += 1
        flux_old[:] = flux[:]


cdef double angle_coef_corrector(double alpha_minus, double angle_x, \
        double angle_w, int angle, params info):
    # For calculating angular differencing coefficient
    if angle != info.angles - 1:
        return alpha_minus - angle_x * angle_w
    return 0.0


cdef void initialize_half_angle(double[:]& flux, double[:]& half_angle, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, int[:]& medium_map, double[:]& delta_x, \
        double angle_plus, params info):
    # Initialize cell and material iteration index
    cdef int ii, mat
    # Zero out half angle
    half_angle[:] = 0.0
    # Iterate from sphere surface to center
    for ii in range(info.cells_x-1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]
        # Calculate angular flux half angle
        half_angle[ii] = (2 * angle_plus + delta_x[ii] * (external[ii] \
                        + off_scatter[ii] + xs_scatter[mat] * flux[ii])) \
                        / (2 + xs_total[mat] * delta_x[ii])
        # Update half angle coefficient
        angle_plus = 2 * half_angle[ii] - angle_plus


cdef void sphere_sweep(double[:]& flux, double[:]& flux_old, \
        double[:]& half_angle, double[:]& xs_total, double[:]& xs_scatter, \
        double[:]& off_scatter, double[:]& external, double[:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double weight, double tau, double alpha_plus, \
        double alpha_minus, params info):
    # Iterate from I -> 0 (sphere edge to center)
    if angle_x < 0.0:
        sphere_backward(flux, flux_old, half_angle, xs_total, xs_scatter, \
            off_scatter, external, boundary_x[1], medium_map, delta_x, \
            angle_x, angle_w, weight, tau, alpha_plus, alpha_minus, info)
    # Iterate from 0 -> I (sphere center to edge)
    elif angle_x > 0.0:
        sphere_forward(flux, flux_old, half_angle, xs_total, xs_scatter, \
            off_scatter, external, medium_map, delta_x, angle_x, angle_w, \
            weight, tau, alpha_plus, alpha_minus, info)


cdef void sphere_forward(double[:]& flux, double[:]& flux_old, \
        double[:]& half_angle, double[:]& xs_total, double[:]& xs_scatter, \
        double[:]& off_scatter, double[:]& external, int[:]& medium_map, \
        double[:]& delta_x, double angle_x, double angle_w, double weight, \
        double tau, double alpha_plus, double alpha_minus, params info):
    # Initialize cell and material iteration index
    cdef int ii, mat
    # Initialize known cell edge
    cdef double edge1 = half_angle[0]
    # Determine flux edge
    if info.edges:
        flux[0] += weight * edge1
    # Initialize surface area on cell edges, cell volume, flux center
    cdef double area1, area2, center, volume
    # Iterate over cells from 0 -> I (center to edge)
    for ii in range(info.cells_x):
        # For determining the material cross sections
        mat = medium_map[ii]
        # Calculate surface area at known cell edge
        area1 = edge_surface_area(ii * delta_x[ii])
        # Calculate surface area at unknown cell edge
        area2 = edge_surface_area((ii + 1) * delta_x[ii])
        # Calculate volume of cell
        volume = cell_volume((ii + 1) * delta_x[ii], ii * delta_x[ii])
        # Calculate flux at cell center
        center = (angle_x * (area2 + area1) * edge1 + 1 / angle_w * (area2 - area1) \
                * (alpha_plus + alpha_minus) * (half_angle[ii]) + volume \
                * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])) \
                / (2 * angle_x * area2 + 2 / angle_w * (area2 - area1) * alpha_plus \
                + xs_total[mat] * volume)
        # Update flux with cell edges
        if info.edges:
            flux[ii+1] += weight * (2 * center - edge1)
        # Update flux with cell centers
        else:
            flux[ii] += weight * center
        # Update known cell edge with step method
        if info.spatial == 1:
            edge1 = center
        # Update known cell edge with diamond difference
        elif info.spatial == 2:
            edge1 = 2 * center - edge1
        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


cdef void sphere_backward(double[:]& flux, double[:]& flux_old, \
        double[:]& half_angle, double[:]& xs_total, double[:]& xs_scatter, \
        double[:]& off_scatter, double[:]& external, double boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double weight, double tau, double alpha_plus, \
        double alpha_minus, params info):
    # Initialize cell and material iteration index
    cdef int ii, mat
    # Initialize known cell edge
    cdef double edge1 = boundary_x
    # Determine flux edge
    if info.edges:
        flux[info.cells_x] += weight * edge1
    # Initialize surface area on cell edges, cell volume, flux center
    cdef double area1, area2, center, volume
    # Iterate over cells from I -> 0 (edge to center)
    for ii in range(info.cells_x-1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]
        # Calculate the surface area at known cell edge
        area1 = edge_surface_area(ii * delta_x[ii])
        # Calculate the surface area at unknown cell edge
        area2 = edge_surface_area((ii + 1) * delta_x[ii])
        # Calculate volume of the cell
        volume = cell_volume((ii + 1) * delta_x[ii], ii * delta_x[ii])
        # Calculate the flux at the cell center
        center = (fabs(angle_x) * (area2 + area1) * edge1 + 1 / angle_w * (area2 - area1) \
                * (alpha_plus + alpha_minus) * (half_angle[ii]) + volume \
                * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])) \
                / (2 * fabs(angle_x) * area1 + 2 / angle_w * (area2 - area1) * alpha_plus \
                + xs_total[mat] * volume)
        # Update flux with cell edges
        if info.edges:
            flux[ii+1] += weight * (2 * center - edge1)
        # Update flux with cell centers
        else:
            flux[ii] += weight * center
        # Update known cell edge with step method
        if info.spatial == 1:
            edge1 = center
        # Update known cell edge with diamond difference
        elif info.spatial == 2:
            edge1 = 2 * center - edge1
        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


cdef double edge_surface_area(double rho):
    # return 4 * M_PI * pow(rho, 2)
    return 4 * M_PI * rho * rho


cdef double cell_volume(double rho_plus, double rho_minus):
    # return 4 * M_PI / 3 * (pow(rho_plus, 3) - pow(rho_minus, 3))
    return 4/3. * M_PI * ((rho_plus * rho_plus * rho_plus) \
                        - (rho_minus * rho_minus * rho_minus))


########################################################################
# Known Source Spatial Sweeps
########################################################################

cdef void _known_sweep(double[:,:]& flux, double[:]& xs_total, \
        double[:]& zero, double[:,:]& source, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    # One-dimensional slab
    if info.geometry == 1:
        _known_slab(flux, xs_total, zero, source, boundary_x, medium_map, \
                    delta_x, angle_x, angle_w, info)
    # One-dimensional sphere
    elif info.geometry == 2:
        _known_sphere(flux, xs_total, zero, source, boundary_x, medium_map, \
                      delta_x, angle_x, angle_w, info)


cdef void _known_slab(double[:,:]& flux, double[:]& xs_total, \
        double[:]& zero, double[:,:]& source, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    
    # Initialize external and boundary indices, iterables
    cdef int nn, qq, bc
    
    # Initialize unknown cell edge
    cdef double edge = 0.0
    
    # Add dummy dimension to run both (I x N) and (I) fluxes
    cdef int xdim = flux.shape[1]
    
    # Add reflector array initialized to zero
    reflector = tools.array_1d(info.angles)
    
    # Iterate over all the discrete ordinates
    for nn in range(info.angles):
        
        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[1] == 1 else nn
        bc = 0 if boundary_x.shape[1] == 1 else nn
        
        # Perform spatial sweep on scalar flux
        if (xdim == 1):
            edge = slab_sweep(flux[:,0], zero, xs_total, zero, zero, \
                    source[:,qq], boundary_x[:,bc], medium_map, \
                    delta_x, angle_x[nn], angle_w[nn], reflector[nn], info)
        
        # Perform spatial sweep on angular flux
        else:
            edge = slab_sweep(flux[:,nn], zero, xs_total, zero, zero, \
                    source[:,qq], boundary_x[:,bc], medium_map, \
                    delta_x, angle_x[nn], 1.0, reflector[nn], info)
        
        # Update reflected direction
        reflector_corrector(reflector, angle_x, edge, nn, info)


cdef void _known_sphere(double[:,:]& flux, double[:]& xs_total, \
        double[:]& zero, double[:,:]& source, double[:,:]& boundary_x, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_x, \
        double[:]& angle_w, params info):
    
    # Initialize external and boundary indices, iterables
    cdef int nn, qq, bc
    
    # Initialize sphere specific terms
    cdef double angle_minus, angle_plus, tau
    cdef double alpha_minus, alpha_plus
    half_angle = tools.array_1d(info.cells_x)
    
    # Add dummy dimension to run both (I x N) and (I) fluxes
    cdef int xdim = flux.shape[1]
    
    # Initialize the half angle coefficient
    angle_minus = -1.0
    
    # Initialize the angular differencing coefficient
    alpha_minus = 0.0
    
    # Calculate the initial half angle
    initialize_half_angle(zero, half_angle, xs_total, zero, zero, \
                source[:,0], medium_map, delta_x, boundary_x[1,0], info)
    
    # Iterate over all the discrete ordinates
    for nn in range(info.angles):
    
        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[1] == 1 else nn
        bc = 0 if boundary_x.shape[1] == 1 else nn
    
        # Calculate the half angle coefficient
        angle_plus = angle_minus + 2 * angle_w[nn]
    
        # Calculate the weighted diamond
        tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
    
        # Calculate the angular differencing coefficient
        alpha_plus = angle_coef_corrector(alpha_minus, angle_x[nn], \
                                          angle_w[nn], nn, info)
    
        # Iterate over spatial cells
        if (xdim == 1):
            sphere_sweep(flux[:,0], zero, half_angle, xs_total, zero, \
                        zero, source[:,qq], boundary_x[:,bc], medium_map, \
                        delta_x, angle_x[nn], angle_w[nn], angle_w[nn], \
                        tau, alpha_plus, alpha_minus, info)
        else:
            sphere_sweep(flux[:,nn], zero, half_angle, xs_total, zero, \
                        zero, source[:,qq], boundary_x[:,bc], medium_map, \
                        delta_x, angle_x[nn], angle_w[nn], 1.0, tau, \
                        alpha_plus, alpha_minus, info)
        
        # Update the angular differencing coefficient
        alpha_minus = alpha_plus
        
        # Update the half angle
        angle_minus = angle_plus