########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Spatial sweeps for two-dimensional neutron transport problems.
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

# from libc.math cimport fabs
# from cython.parallel import prange

from ants cimport cytools_2d as tools
from ants.parameters cimport params
from ants.constants import *


cdef void discrete_ordinates(double[:,:]& flux, double[:,:]& flux_old, 
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary_x, \
        double[:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params info):
    # Rectangular spatial cells
    if info.geometry == 1:
        square_ordinates(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, boundary_x, boundary_y, medium_map, \
                         delta_x, delta_y, angle_x, angle_y, angle_w, info)


cdef void square_ordinates(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary_x, \
        double[:,:,:]& boundary_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_w, params info):

    # Initialize indices etc
    cdef int nn, qq, bcx, bcy

    # Add reflector array
    known_y = tools.array_1d(info.cells_x)
    reflected_y = tools.array_3d(2, info.cells_x, info.angles * info.angles)
    known_x = tools.array_1d(info.cells_y)
    reflected_x = tools.array_3d(2, info.cells_y, info.angles * info.angles)

    # Set convergence limits
    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    # Iterate over angles until converged
    while not (converged):

        # Zero out the scalar flux
        flux[:,:] = 0.0

        # Iterate over angles
        for nn in range(info.angles * info.angles):
        
            # Determine dimensions of external and boundary sources
            qq = 0 if external.shape[2] == 1 else nn
            bcx = 0 if boundary_x.shape[2] == 1 else nn
            bcy = 0 if boundary_y.shape[2] == 1 else nn        

            # Initialize known x and y
            tools.initialize_known_y(known_y, boundary_y[:,:,bcy], \
                                     reflected_y, angle_y, nn, info)
            tools.initialize_known_x(known_x, boundary_x[:,:,bcx], \
                                     reflected_x, angle_x, nn, info)

            # Perform spatial sweep
            square_sweep(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                    external[:,:,qq], known_x, known_y, medium_map, delta_x, \
                    delta_y, angle_x[nn], angle_y[nn], angle_w[nn], info)

            # Save known_x, known_y into reflected
            tools.update_reflector(known_x, reflected_x, angle_x, known_y, \
                                   reflected_y, angle_y, nn, info)

        # Check for convergence
        change = tools.angle_convergence(flux, flux_old, info)
        converged = (change < EPSILON_ANGULAR) or (count >= MAX_ANGULAR)
        count += 1

        # Update old flux
        flux_old[:,:] = flux[:,:]


cdef void square_sweep(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    if angle_y > 0.0:
        square_forward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                         external, known_x, known_y, medium_map, delta_x, \
                         delta_y, angle_x, angle_y, angle_w, info)
    elif angle_y < 0.0:
        square_backward_y(flux, flux_old, xs_total, xs_scatter, off_scatter, \
                          external, known_x, known_y, medium_map, delta_x, \
                          delta_y, angle_x, angle_y, angle_w, info)


cdef void square_forward_y(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    
    # Initialize iterables
    cdef int jj
    cdef double coef_y, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over Y spatial cells
    for jj in range(info.cells_y):

        # Angular coefficient
        coef_y = const * angle_y / delta_y[jj]

        # Set direction of sweep
        if angle_x > 0.0:
            known_x[jj] = square_forward_x(flux[:,jj], flux_old[:,jj], \
                                    xs_total, xs_scatter, off_scatter[:,jj], \
                                    external[:,jj], known_x[jj], known_y, \
                                    medium_map[:,jj], delta_x, angle_x, \
                                    angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = square_backward_x(flux[:,jj], flux_old[:,jj], \
                                    xs_total, xs_scatter, off_scatter[:,jj], \
                                    external[:,jj], known_x[jj], known_y, \
                                    medium_map[:,jj], delta_x, angle_x, \
                                    angle_w, coef_y, info)


cdef void square_backward_y(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    
    # Initialize iterable
    cdef int jj
    cdef double coef_y, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over Y spatial cells
    for jj in range(info.cells_y-1, -1, -1):

        # Angular coefficient
        coef_y = -const * angle_y / delta_y[jj]

        # Set direction of sweep
        if angle_x > 0.0:
            known_x[jj] = square_forward_x(flux[:,jj], flux_old[:,jj], \
                                    xs_total, xs_scatter, off_scatter[:,jj], \
                                    external[:,jj], known_x[jj], known_y, \
                                    medium_map[:,jj], delta_x, angle_x, \
                                    angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = square_backward_x(flux[:,jj], flux_old[:,jj], \
                                    xs_total, xs_scatter, off_scatter[:,jj], \
                                    external[:,jj], known_x[jj], known_y, \
                                    medium_map[:,jj], delta_x, angle_x, \
                                    angle_w, coef_y, info)


cdef double square_forward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double coef_y, params info):
    
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over X spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        coef_x = (const * angle_x / delta_x[ii])

        # Calculate flux center
        center = (coef_x * edge_x + coef_y * edge_y[ii] + xs_scatter[mat] \
                    * flux_old[ii] + external[ii] + off_scatter[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)
        
        # Update flux with cell centers
        flux[ii] += angle_w * center
        
        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]
    
    return edge_x


cdef double square_backward_x(double[:]& flux, double[:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:]& off_scatter, \
        double[:]& external, double edge_x, double[:]& edge_y, \
        int[:]& medium_map, double[:]& delta_x, double angle_x, \
        double angle_w, double coef_y, params info):
    
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over X spatial cells
    for ii in range(info.cells_x-1, -1, -1):
        mat = medium_map[ii]
        coef_x = (-const * angle_x / delta_x[ii])
    
        # Calculate flux center
        center = (coef_x * edge_x + coef_y * edge_y[ii] + xs_scatter[mat] \
                    * flux_old[ii] + external[ii] + off_scatter[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)

        # Update flux with cell centers
        flux[ii] += angle_w * center

        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]

    return edge_x


########################################################################
# Known Source Spatial Sweeps
########################################################################

cdef void _known_center_sweep(double[:,:,:]& flux, double[:]& xs_total, \
        double[:,:]& zero_2d, double[:,:,:]& source, \
        double[:,:,:]& boundary_x, double[:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    # if info.geometry == 1:
    # Rectangular spatial cells
    _known_square(flux, xs_total, zero_2d, source, boundary_x, \
                  boundary_y, medium_map, delta_x, delta_y, angle_x, \
                  angle_y, angle_w, info)


cdef void _known_square(double[:,:,:]& flux, double[:]& xs_total, \
        double[:,:]& zero_2d, double[:,:,:]& source, \
        double[:,:,:]& boundary_x, double[:,:,:]& boundary_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    
    # Initialize indices etc
    cdef int nn, qq, bcx, bcy

    # Add dummy dimension to run both (I x J x N) and (I x J) fluxes
    cdef int xdim = flux.shape[2]

    # Add reflector array
    known_y = tools.array_1d(info.cells_x)
    reflected_y = tools.array_3d(2, info.cells_x, info.angles * info.angles)
    known_x = tools.array_1d(info.cells_y)
    reflected_x = tools.array_3d(2, info.cells_y, info.angles * info.angles)

    # Add zero placeholder
    zero_1d = tools.array_1d(info.materials)

    # Iterate over angles
    for nn in range(info.angles * info.angles):
    
        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[2] == 1 else nn
        bcx = 0 if boundary_x.shape[2] == 1 else nn
        bcy = 0 if boundary_y.shape[2] == 1 else nn

        # Initialize known x and y
        tools.initialize_known_y(known_y, boundary_y[:,:,bcy], \
                                 reflected_y, angle_y, nn, info)
        tools.initialize_known_x(known_x, boundary_x[:,:,bcx], \
                                 reflected_x, angle_x, nn, info)

        if (xdim == 1):
            # Perform spatial sweep - scalar flux
            square_sweep(flux[:,:,0], zero_2d, xs_total, zero_1d, zero_2d, \
                source[:,:,qq], known_x, known_y, medium_map, delta_x, \
                delta_y, angle_x[nn], angle_y[nn], angle_w[nn], info)
        else:
            # Perform spatial sweep - angular flux
            square_sweep(flux[:,:,nn], zero_2d, xs_total, zero_1d, zero_2d, \
                source[:,:,qq], known_x, known_y, medium_map, delta_x, \
                delta_y, angle_x[nn], angle_y[nn], 1.0, info)

        # Save known_x, known_y into reflected
        tools.update_reflector(known_x, reflected_x, angle_x, known_y, \
                               reflected_y, angle_y, nn, info)


cdef void _known_interface_sweep(double[:,:,:]& flux_edge_x, \
        double[:,:,:]& flux_edge_y, double[:]& xs_total, \
        double[:,:,:]& source, double[:,:,:]& boundary_x, \
        double[:,:,:]& boundary_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    
    # Initialize indices etc
    cdef int nn, qq, bcx, bcy
    
    # Add dummy dimension to run both (I x J x N) and (I x J) fluxes
    cdef int xdim = flux_edge_x.shape[2]

    # Add reflector array
    known_y = tools.array_1d(info.cells_x)
    reflected_y = tools.array_3d(2, info.cells_x, info.angles * info.angles)
    known_x = tools.array_1d(info.cells_y)
    reflected_x = tools.array_3d(2, info.cells_y, info.angles * info.angles)

    # Iterate over angles
    for nn in range(info.angles * info.angles):
    
        # Determine dimensions of external and boundary sources
        qq = 0 if source.shape[2] == 1 else nn
        bcx = 0 if boundary_x.shape[2] == 1 else nn
        bcy = 0 if boundary_y.shape[2] == 1 else nn
        
        # Initialize known x and y
        tools.initialize_known_y(known_y, boundary_y[:,:,bcy], \
                                 reflected_y, angle_y, nn, info)
        tools.initialize_known_x(known_x, boundary_x[:,:,bcx], \
                                 reflected_x, angle_x, nn, info)

        if (xdim == 1):
            # Perform spatial sweep - scalar flux
            interface_sweep(flux_edge_x[:,:,0], flux_edge_y[:,:,0], xs_total, \
                    source[:,:,qq], known_x, known_y, medium_map, delta_x, \
                    delta_y, angle_x[nn], angle_y[nn], angle_w[nn], info)
        else:
            # Perform spatial sweep - angular flux
            interface_sweep(flux_edge_x[:,:,nn], flux_edge_y[:,:,nn], \
                    xs_total, source[:,:,qq], known_x, known_y, medium_map, \
                    delta_x, delta_y, angle_x[nn], angle_y[nn], 1.0, info)

        # Save known_x, known_y into reflected
        tools.update_reflector(known_x, reflected_x, angle_x, known_y, \
                               reflected_y, angle_y, nn, info)


cdef void interface_sweep(double[:,:]& flux_edge_x, double[:,:]& flux_edge_y, \
        double[:]& xs_total, double[:,:]& external, double[:]& known_x, \
        double[:]& known_y, int[:,:]& medium_map, double[:]& delta_x, \
        double[:]& delta_y, double angle_x, double angle_y, double angle_w, \
        params info):
    
    if (angle_y > 0.0):
        interface_forward_y(flux_edge_x, flux_edge_y, xs_total, external, \
                            known_x, known_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info)
    elif (angle_y < 0.0):
        interface_backward_y(flux_edge_x, flux_edge_y, xs_total, external, \
                            known_x, known_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, info)


cdef void interface_forward_y(double[:,:]& flux_edge_x, \
        double[:,:]& flux_edge_y, double[:]& xs_total, \
        double[:,:]& external, double[:]& known_x, double[:]& known_y, \
        int[:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double angle_x, double angle_y, double angle_w, params info):
    
    # Initialize iterables
    cdef int ii, jj
    cdef double coef_y, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over Y spatial cells
    for jj in range(info.cells_y):
    
        coef_y = const * angle_y / delta_y[jj]
        if angle_x > 0.0:
            known_x[jj] = interface_forward_x(flux_edge_x[:,jj], \
                                flux_edge_y[:,jj], xs_total, external[:,jj], \
                                known_x[jj], known_y, medium_map[:,jj], \
                                delta_x, angle_x, angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = interface_backward_x(flux_edge_x[:,jj], \
                                flux_edge_y[:,jj], xs_total, external[:,jj], \
                                known_x[jj], known_y, medium_map[:,jj], \
                                delta_x, angle_x, angle_w, coef_y, info)

    # Solve for top boundary (flux_edge_y)
    for ii in range(info.cells_x):
        flux_edge_y[ii,info.cells_y] += angle_w * known_y[ii]


cdef void interface_backward_y(double[:,:]& flux_edge_x, \
        double[:,:]& flux_edge_y, double[:]& xs_total, double[:,:]& external, \
        double[:]& known_x, double[:]& known_y, int[:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double angle_x, \
        double angle_y, double angle_w, params info):
    
    # Initialize iterable
    cdef int ii, jj
    cdef double coef_y, const
    
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    
    # Iterate over Y spatial cells
    for jj in range(info.cells_y-1, -1, -1):
        coef_y = -const * angle_y / delta_y[jj]
        if angle_x > 0.0:
            known_x[jj] = interface_forward_x(flux_edge_x[:,jj], \
                                flux_edge_y[:,jj+1], xs_total, external[:,jj], \
                                known_x[jj], known_y, medium_map[:,jj], \
                                delta_x, angle_x, angle_w, coef_y, info)
        elif angle_x < 0.0:
            known_x[jj] = interface_backward_x(flux_edge_x[:,jj], \
                                flux_edge_y[:,jj+1], xs_total, external[:,jj], \
                                known_x[jj], known_y, medium_map[:,jj], \
                                delta_x, angle_x, angle_w, coef_y, info)
    # Solve for bottom boundary (flux_edge_y)
    for ii in range(info.cells_x):
        flux_edge_y[ii,0] += angle_w * known_y[ii]


cdef double interface_forward_x(double[:]& flux_edge_x, double[:]& flux_edge_y, \
        double[:]& xs_total, double[:]& external, double edge_x, \
        double[:]& edge_y, int[:]& medium_map, double[:]& delta_x, \
        double angle_x, double angle_w, double coef_y, params info):
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Start with initial edge (i-1/2, j)
    flux_edge_x[0] += angle_w * edge_x
    # Iterate over X spatial cells
    for ii in range(info.cells_x):
        mat = medium_map[ii]
        coef_x = (const * angle_x / delta_x[ii])
        center = (coef_x * edge_x + coef_y * edge_y[ii] + external[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)
        # Update flux_edge_y (i, j-1/2)
        flux_edge_y[ii] += angle_w * edge_y[ii]
        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]
        # Update flux_edge_x (i+1/2, j)
        flux_edge_x[ii+1] += angle_w * edge_x
    return edge_x


cdef double interface_backward_x(double[:]& flux_edge_x, double[:]& flux_edge_y, \
        double[:]& xs_total, double[:]& external, double edge_x, \
        double[:]& edge_y, int[:]& medium_map, double[:]& delta_x, \
        double angle_x, double angle_w, double coef_y, params info):
    # Initialize iterables
    cdef int ii, mat
    cdef double center, coef_x, const
    # Step vs Diamond
    const = 2.0 if info.spatial == 2 else 1.0
    # Start with initial edge (i+1/2, j)
    flux_edge_x[info.cells_x] += angle_w * edge_x
    # Iterate over X spatial cells
    for ii in range(info.cells_x-1, -1, -1):
        mat = medium_map[ii]
        coef_x = (-const * angle_x / delta_x[ii])
        center = (coef_x * edge_x + coef_y * edge_y[ii] + external[ii]) \
                    / (xs_total[mat] + coef_x + coef_y)
        # Update flux_edge_y (i, j-1/2)
        flux_edge_y[ii] += angle_w * edge_y[ii]
        # Update known flux with step method
        if info.spatial == 1:
            edge_x = center
            edge_y[ii] = center
        # Update known flux with diamond difference
        elif info.spatial == 2:
            edge_x = 2 * center - edge_x
            edge_y[ii] = 2 * center - edge_y[ii]
        # Update flux_edge_x to (i-1/2, j)
        flux_edge_x[ii] += angle_w * edge_x
    return edge_x