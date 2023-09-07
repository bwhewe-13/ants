########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Functions needed for both fixed source, criticality, and 
# time-dependent problems in one-dimensional neutron transport 
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

from libc.math cimport sqrt, pow, erfc, ceil
from cython.view cimport array as cvarray

from ants.parameters cimport params

########################################################################
# Memoryview functions
########################################################################
cdef double[:] array_1d(int dim1):
    dd1 = cvarray((dim1,), itemsize=sizeof(double), format="d")
    cdef double[:] arr = dd1
    arr[:] = 0.0
    return arr


cdef double[:,:] array_2d(int dim1, int dim2):
    dd2 = cvarray((dim1, dim2), itemsize=sizeof(double), format="d")
    cdef double[:,:] arr = dd2
    arr[:,:] = 0.0
    return arr


cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3):
    dd3 = cvarray((dim1, dim2, dim3), itemsize=sizeof(double), format="d")
    cdef double[:,:,:] arr = dd3
    arr[:,:,:] = 0.0
    return arr


cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4):
    dd4 = cvarray((dim1, dim2, dim3, dim4), itemsize=sizeof(double), format="d")
    cdef double[:,:,:,:] arr = dd4
    arr[:,:,:,:] = 0.0
    return arr

########################################################################
# Convergence functions
########################################################################
cdef double group_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info):
    # Calculate the L2 convergence of the scalar flux in the energy loop
    cdef int ii, jj, gg
    cdef int cells = info.cells_x * info.cells_y
    cdef double change = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for gg in range(info.groups):
                if arr1[ii,jj,gg] == 0.0:
                    continue
                change += pow((arr1[ii,jj,gg] - arr2[ii,jj,gg]) \
                              / arr1[ii,jj,gg] / cells, 2)
    change = sqrt(change)
    return change


cdef double angle_convergence(double[:,:]& arr1, double[:,:]& arr2, params info):
    # Calculate the L2 convergence of the scalar flux in the ordinates loop
    cdef int ii, jj
    cdef int cells = info.cells_x * info.cells_y
    cdef double change = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            if arr1[ii,jj] == 0.0:
                continue
            change += pow((arr1[ii,jj] - arr2[ii,jj]) / arr1[ii,jj] / cells, 2)
    change = sqrt(change)
    return change

########################################################################
# Multigroup functions
########################################################################

cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
    double[:,:,:]& mat3, params info):
    # Initialize iterables
    cdef int ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig] = (mat2[mat,og,ig] + mat3[mat,og,ig])


cdef void _off_scatter(double[:,:,:]& flux, double[:,:,:]& flux_old, \
        int[:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:]& off_scatter, params info, int group):
    # Initialize iterables
    cdef int ii, jj, mat, og
    # Zero out previous values
    off_scatter[:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(0, group):
                off_scatter[ii,jj] += xs_matrix[mat,group,og] * flux[ii,jj,og]
            for og in range(group + 1, info.groups):
                off_scatter[ii,jj] += xs_matrix[mat,group,og] * flux_old[ii,jj,og]


cdef void _source_total(double[:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:]& medium_map, \
        double[:]& external, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, nn, ig, og, mat, loc, NN
    NN = info.angles * info.angles
    # Zero out previous values
    source[:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for nn in range(NN):
                for og in range(info.groups):
                    loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                    for ig in range(info.groups):
                        source[loc] += flux[ii,jj,ig] * xs_matrix[mat,og,ig]
                    source[loc] += external[loc]


cdef void _angular_to_scalar(double[:,:,:,:]& angular_flux, \
        double[:,:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    scalar_flux[ii,jj,gg] += angular_flux[ii,jj,nn,gg] * angle_w[nn]
    # return scalar_flux


cdef void _initialize_edge_y(double[:]& known_y, double[:]& boundary_y, \
        double[:]& angle_y, double[:]& angle_x, int nn, params info):
    # This is for adding boundary conditions to known edge x
    # Initialize location
    cdef int loc, width
    # Keep flux from previous
    if (angle_y[nn] == -angle_y[nn-1]) and (angle_x[nn] == angle_x[nn-1]) \
            and (nn > 0) and (((angle_y[nn] > 0.0) and (info.bc_y[0] == 1)) \
            or ((angle_y[nn] < 0.0) and (info.bc_y[1] == 1))):
        return
    # Zero out flux
    known_y[:] = 0.0
    # Pick left / right location
    loc = 0 if angle_y[nn] > 0.0 else 1
    # Populate known edge value
    if info.bcdim_y == 1:
        known_y[:] = boundary_y[loc]
    else:
        width = info.cells_x + info.edges
        known_y[:] = boundary_y[loc * width:(loc+1) * width]


cdef void _initialize_edge_x(double[:]& known_x, double[:]& boundary_x, \
        double[:]& angle_x, double[:]& angle_y, int nn, params info):
    # This is for adding boundary conditions to known edge x
    # Initialize location
    cdef int loc, width
    # Keep flux from previous
    if (angle_x[nn] == -angle_x[nn-1]) and (angle_y[nn] == angle_y[nn-1]) \
            and (nn > 0) and (((angle_x[nn] > 0.0) and (info.bc_x[0] == 1)) \
            or ((angle_x[nn] < 0.0) and (info.bc_x[1] == 1))):
        return
    # Zero out flux
    known_x[:] = 0.0
    # Pick left / right location
    loc = 0 if angle_x[nn] > 0.0 else 1
    # Populate known edge value
    if info.bcdim_x == 1:
        known_x[:] = boundary_x[loc]
    else:
        width = info.cells_y + info.edges
        known_x[:] = boundary_x[loc * width:(loc+1) * width]


########################################################################
# Time Dependent functions
########################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, params info):
    # Create sigma_t + 1 / (v * dt)
    cdef int mm, gg
    for gg in range(info.groups):
        for mm in range(info.materials):
            xs_total[mm,gg] += 1 / (velocity[gg] * info.dt)


cdef void _time_source_total(double[:]& source, double[:,:,:]& scalar_flux, \
        double[:,:,:,:]& angular_flux, double[:,:,:]& xs_matrix, \
        double[:]& velocity, int[:,:]& medium_map, double[:]& external, \
        params info):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Initialize iterables
    cdef int ii, jj, nn, NN, ig, og, mat, loc
    NN = info.angles * info.angles
    # Zero out previous values
    source[:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii, jj]
            for nn in range(NN):
                for og in range(info.groups):
                    loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                    for ig in range(info.groups):
                        source[loc] += scalar_flux[ii,jj,ig] * xs_matrix[mat,og,ig]
                    source[loc] += external[loc] + angular_flux[ii,jj,nn,og] \
                                    * 1 / (velocity[og] * info.dt)


cdef void _time_source_star(double[:,:,:,:]& angular_flux, double[:]& q_star, \
        double[:]& external, double[:]& velocity, params info):
    # Combining the source (I x J x N^2 x G) with the angular flux (I x J x N^2 x G)
    # Initialize iterables
    cdef int ii, jj, nn, NN, gg, loc
    NN = info.angles * info.angles
    # Zero out previous values
    q_star[:] = 0.0
    for gg in range(info.groups):
        for nn in range(NN):
            for jj in range(info.cells_y):
                for ii in range(info.cells_x):
                    loc = gg + info.groups * (nn + NN * (jj + ii * info.cells_y))
                    q_star[loc] = external[loc] + angular_flux[ii,jj,nn,gg] \
                                    * 1 / (velocity[gg] * info.dt)


cdef void boundary_decay(double[:]& boundary_x, double[:]& boundary_y, \
        int step, params info):
    # Calculate elapsed time
    cdef double t = info.dt * step
    cdef bint switch = True
    # Cycle through different decay processes for x boundaries
    if info.bcdecay_x == 0: # Do nothing
        pass
    elif info.bcdecay_x == 1: # Turn off after one step
        switch = (step > 0)
        _decay_x_switch(boundary_x, switch, info)
    elif info.bcdecay_x == 2: # Step decay
        _decay_x_02(boundary_x, t, info)
    elif info.bcdecay_x == 3: # Turn off after 10 microseconds
        switch = (t > 10e-6)
        _decay_x_switch(boundary_x, switch, info)
    # Cycle through different decay processes for y boundaries
    if info.bcdecay_y == 0: # Do nothing
        pass
    elif info.bcdecay_y == 1: # Turn off after one step
        switch = (step > 0)
        _decay_y_switch(boundary_y, switch, info)
    elif info.bcdecay_y == 2: # Step decay
        _decay_y_02(boundary_y, t, info)
    elif info.bcdecay_y == 3: # Turn off after 10 microseconds
        switch = (t > 10e-6)
        _decay_y_switch(boundary_y, switch, info)


cdef int _boundary_length(int bcdim, int cells, params info):
    # Initialize boundary length
    cdef int bc_length
    if bcdim == 1:
        bc_length = 2
    elif bcdim == 2:
        bc_length = 2 * cells
    elif bcdim == 3:
        bc_length = 2 * cells * info.groups
    elif bcdim == 3:
        bc_length = 2 * cells * info.angles * info.angles * info.groups
    return bc_length


cdef void _decay_x_switch(double[:]& boundary_x, bint switch, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_x, info.cells_y, info)
    cdef double magnitude = 0.0 if switch else 1.0
    for cell in range(bc_length):
        if boundary_x[cell] == 0.0:
            continue
        else:
            boundary_x[cell] = magnitude


cdef void _decay_x_02(double[:]& boundary_x, double t, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_x, info.cells_y, info)
    cdef double k, err_arg
    t *= 1e6 # Convert elapsed time
    for cell in range(bc_length):
        if boundary_x[cell] == 0.0:
            continue
        if t < 0.2:
            boundary_x[cell] = 1.
        else:
            k = ceil((t - 0.2) / 0.1)
            err_arg = (t - 0.1 * (1 + k)) / (0.01)
            boundary_x[cell] = pow(0.5, k) * (1 + 2 * erfc(err_arg))


cdef void _decay_y_switch(double[:]& boundary_y, bint switch, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_y, info.cells_x, info)
    cdef double magnitude = 0.0 if switch else 1.0
    for cell in range(bc_length):
        if boundary_y[cell] == 0.0:
            continue
        else:
            boundary_y[cell] = magnitude


cdef void _decay_y_02(double[:]& boundary_y, double t, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_y, info.cells_x, info)
    cdef double k, err_arg
    t *= 1e6 # Convert elapsed time
    for cell in range(bc_length):
        if boundary_y[cell] == 0.0:
            continue
        if t < 0.2:
            boundary_y[cell] = 1.
        else:
            k = ceil((t - 0.2) / 0.1)
            err_arg = (t - 0.1 * (1 + k)) / (0.01)
            boundary_y[cell] = pow(0.5, k) * (1 + 2 * erfc(err_arg))


########################################################################
# Criticality functions
########################################################################

cdef void _normalize_flux(double[:,:,:]& flux, params info):
    cdef int ii, jj, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for jj in range(info.cells_y):
            for ii in range(info.cells_x):
                keff += (flux[ii,jj,gg] * flux[ii,jj,gg])
    keff = sqrt(keff)
    for gg in range(info.groups):
        for jj in range(info.cells_y):
            for ii in range(info.cells_x):
                flux[ii,jj,gg] /= keff


cdef void _fission_source(double[:,:,:] flux, double[:,:,:] xs_fission, \
        double[:] source, int[:,:] medium_map, params info, double keff):
    # Calculate the fission source (I x G) for the power iteration
    # (keff^{-1} * sigma_f * phi)
    # Initialize iterables
    cdef int ii, jj, mat, ig, og, loc
    # Zero out previous power source
    source[:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                loc = og + info.groups * (jj + ii * info.cells_y)
                for ig in range(info.groups):
                    source[loc] += flux[ii,jj,ig] * xs_fission[mat,og,ig]
                source[loc] /= keff


cdef double _update_keffective(double[:,:,:] flux_new, double[:,:,:] flux_old, \
        double[:,:,:] xs_fission, int[:,:] medium_map, params info, double keff):
    # Initialize iterables
    cdef int ii, jj, mat, ig, og
    # Initialize fission rates for 2 fluxes
    cdef double rate_new = 0.0
    cdef double rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                for ig in range(info.groups):
                    rate_new += flux_new[ii,jj,ig] * xs_fission[mat,og,ig]
                    rate_old += flux_old[ii,jj,ig] * xs_fission[mat,og,ig]
    return (rate_new * keff) / rate_old


cdef void _source_total_critical(double[:]& source, double[:,:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:]& xs_fission, \
        int[:,:]& medium_map, double keff, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, ig, og, mat, loc
    # Zero out previous values
    source[:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info.groups):
                loc = og + info.groups * (jj + ii * info.cells_y)
                for ig in range(info.groups):
                    source[loc] += (flux[ii,jj,ig] * xs_fission[mat,og,ig]) / keff \
                                 + (flux[ii,jj,ig] * xs_scatter[mat,og,ig])

########################################################################
# Nearby Problems Criticality functions
########################################################################

cdef void _nearby_fission_source(double[:,:,:]& flux, \
        double[:,:,:]& xs_fission, double[:]& source, double[:]& residual, \
        int[:,:]& medium_map, params info, double keff):
    # Initialize iterables
    cdef int ii, jj, mat, nn, NN, ig, og, loc
    NN = info.angles * info.angles
    # Zero out previous power iteration
    source[:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            mat = medium_map[ii,jj]
            for nn in range(NN):
                for og in range(info.groups):
                    loc = og + info.groups * (nn + NN * (jj + ii * info.cells_y))
                    for ig in range(info.groups):
                        source[loc] += flux[ii,jj,ig] / keff \
                                        * xs_fission[mat,og,ig]
                    # Add nearby residual
                    source[loc] += residual[loc]


cdef double _nearby_keffective(double[:,:,:]& flux, double rate, params info):
    cdef int ii, jj, gg
    cdef double keff = 0.0
    for gg in range(info.groups):
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                keff += rate * flux[ii,jj,gg]
    return keff

########################################################################
# Hybrid Method Time Dependent Problems
########################################################################

cdef void _hybrid_source_collided(double[:,:,:]& flux, double[:,:,:]& xs_scatter, \
        double[:]& source_c, int[:,:]& medium_map, int[:]& index_c, \
        params info_u, params info_c):
    # Initialize iterables
    cdef int ii, jj, mat, og, ig, loc
    # Zero out previous source
    source_c[:] = 0.0
    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            mat = medium_map[ii,jj]
            for og in range(info_u.groups):
                loc = index_c[og] + info_c.groups * (jj + ii * info_c.cells_y)
                for ig in range(info_u.groups):
                    source_c[loc] += flux[ii,jj,ig] * xs_scatter[mat,og,ig]


cdef void _hybrid_source_total(double[:,:,:]& flux_t, double[:,:,:]& flux_u, \
        double[:,:,:]& xs_matrix, double[:]& source, int[:,:]& medium_map, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
    # Initialize iterables
    cdef int ii, jj, mat, nn, NN, ig, og, loc
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    # Get all angular directions
    NN = info_u.angles * info_u.angles
    # source[:] = 0.0
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_x):
            mat = medium_map[ii,jj]
            for nn in range(NN):
                for og in range(info_u.groups):
                    loc = og + info_u.groups * (nn + NN * (jj + ii * info_u.cells_y))
                    for ig in range(info_u.groups):
                        source[loc] += (flux_t[ii,jj,ig] + flux_u[ii,jj,ig]) \
                                        * xs_matrix[mat,og,ig]


cdef void _expand_hybrid_source(double[:,:,:]& flux_t, double[:,:,:]& flux_c, \
        int[:]& index_u, double[:]& factor_u, params info_u, params info_c):
    # Initialize iterables
    cdef int ii, jj, gu, gc
    flux_t[:,:,:] = 0.0
    # Create uncollided flux size
    for ii in range(info_c.cells_x):
        for jj in range(info_c.cells_y):
            for gc in range(info_c.groups):
                for gu in range(index_u[gc], index_u[gc+1]):
                    flux_t[ii,jj,gu] = flux_c[ii,jj,gc] * factor_u[gu]
