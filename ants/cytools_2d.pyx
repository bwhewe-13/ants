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
    cdef int cells = (info.cells_x + info.edges) + (info.cells_y + info.edges)
    cdef double change = 0.0
    for ii in range(info.cells_x + info.edges):
        for jj in range(info.cells_y + info.edges):
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
    cdef int cells = (info.cells_x + info.edges) + (info.cells_y + info.edges)
    cdef double change = 0.0
    for ii in range(info.cells_x + info.edges):
        for jj in range(info.cells_y + info.edges):
            if arr1[ii,jj] == 0.0:
                continue
            change += pow((arr1[ii,jj] - arr2[ii,jj]) / arr1[ii,jj] / cells, 2)
    change = sqrt(change)
    return change

########################################################################
# Multigroup functions
########################################################################

cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, params info):
    # Initialize iterables
    cdef int ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig] += mat2[mat,og,ig]


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


cdef double[:,:,:] _angular_to_scalar(double[:,:,:,:]& angular_flux,
        double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, nn, gg
    # Initialize scalar flux term
    scalar_flux = array_3d(info.cells_x + info.edges, info.cells_y \
                            + info.edges, info.groups)
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x + info.edges):
        for jj in range(info.cells_y + info.edges):
            for nn in range(info.angles * info.angles):
                for gg in range(info.groups):
                    scalar_flux[ii,jj,gg] += angular_flux[ii,jj,nn,gg] * angle_w[nn]
    return scalar_flux


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
    # Cycle through different decay processes for x boundaries
    if info.bcdecay_x == 0: # Do nothing
        pass
    elif info.bcdecay_x == 1: # Turn off after one step
        _decay_x_01(boundary_x, step, info)
    elif info.bcdecay_x == 2: # Step decay
        _decay_x_02(boundary_x, t, info)
    # Cycle through different decay processes for y boundaries
    if info.bcdecay_y == 0: # Do nothing
        pass
    elif info.bcdecay_y == 1: # Turn off after one step
        _decay_y_01(boundary_y, step, info)
    elif info.bcdecay_y == 2: # Step decay
        _decay_y_02(boundary_y, t, info)


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


cdef void _decay_x_01(double[:]& boundary_x, int step, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_x, info.cells_y, info)
    cdef double magnitude = 0.0 if step > 0 else 1.0
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


cdef void _decay_y_01(double[:]& boundary_y, int step, params info):
    cdef int cell, bc_length
    bc_length = _boundary_length(info.bcdim_y, info.cells_x, info)
    cdef double magnitude = 0.0 if step > 0 else 1.0
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


# cdef void combine_self_scattering(double[:,:,:] xs_matrix, \
#         double[:,:,:] xs_scatter, double[:,:,:] xs_fission, params2d params):
#     cdef size_t mat, ig, og
#     for mat in range(params.materials):
#         for ig in range(params.groups):
#             for og in range(params.groups):
#                 xs_matrix[mat,ig,og] = xs_scatter[mat,ig,og] \
#                                         + xs_fission[mat,ig,og]

# cdef void combine_total_velocity(double[:,:]& xs_total_star, \
#         double[:,:]& xs_total, double[:]& velocity, params2d params):
#     cdef size_t mat, group
#     for group in range(params.groups):
#         for mat in range(params.materials):
#             xs_total_star[mat,group] = xs_total[mat,group] \
#                                     + 1 / (velocity[group] * params.dt)

# cdef void combine_source_flux(double[:,:,:]& flux_last, double[:]& source_star, \
#         double[:]& source, double[:]& velocity, params2d params):
#     cdef size_t cell, angle, group, start
#     cdef size_t stop = params.groups * params.angles
#     for group in range(params.groups):
#         for angle in range(params.angles):
#             for cell in range(params.cells_x * params.cells_y):
#                 start = group + angle * params.groups
#                 source_star[start::stop][cell] = source[start::stop][cell] \
#                         + flux_last[cell,angle,group] \
#                         * 1 / (velocity[group] * params.dt)





# # cdef double[:] group_flux(params2d params):
# #     cdef int flux_size = params.cells_x * params.cells_y * params.groups
# #     flux_size *= params.angles if params.angular == True else 1
# #     dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
# #     cdef double[:] flux = dd1
# #     flux[:] = 0.0
# #     return flux

# cdef double group_convergence_scalar(double[:,:]& arr1, double[:,:]& arr2, \
#         params2d params):
#     cdef size_t cell, group
#     cdef size_t cells  = params.cells_x * params.cells_y
#     cdef double change = 0.0
#     for group in range(params.groups):
#         for cell in range(cells):
#             change += pow((arr1[cell,group] - arr2[cell,group]) \
#                             / arr1[cell,group] / (cells), 2)
#     change = sqrt(change)
#     return change

# cdef double group_convergence_angular(double[:,:,:]& arr1, double[:,:,:]& arr2, \
#         double[:]& weight, params2d params):
#     cdef size_t group, cell, angle
#     cdef double change = 0.0
#     cdef double flux_new, flux_old
#     for group in range(params.groups):
#         for cell in range(params.cells_x * params.cells_y):
#             flux_new = 0.0
#             flux_old = 0.0
#             for angle in range(params.angles):
#                 flux_new += arr1[cell,angle,group] * weight[angle]
#                 flux_old += arr2[cell,angle,group] * weight[angle]
#             change += pow((flux_new - flux_old) / flux_new \
#                             / (params.cells_x * params.cells_y), 2)
#     change = sqrt(change)
#     return change

# cdef double[:] angle_flux(params2d params, bint angular):
#     cdef int flux_size
#     if angular == True:
#         flux_size = params.cells_x * params.cells_y * params.angles
#     else:
#         flux_size = params.cells_x * params.cells_y
#     dd1 = cvarray((flux_size,), itemsize=sizeof(double), format="d")
#     cdef double[:] flux = dd1
#     flux[:] = 0.0
#     return flux

# cdef void angle_angular_to_scalar(double[:,:]& angular, double[:]& scalar, \
#         double[:]& weight, params2d params):
#     cdef size_t cell, angle
#     scalar[:] = 0.0
#     for angle in range(params.angles):
#         for cell in range(params.cells_x * params.cells_y):
#             scalar[cell] += angular[cell,angle] * weight[angle]

# cdef double angle_convergence_scalar(double[:]& arr1, double[:]& arr2, \
#         params2d params):
#     cdef size_t cell
#     cdef double change = 0.0
#     for cell in range(params.cells_x * params.cells_y):
#         change += pow((arr1[cell] - arr2[cell]) / arr1[cell] \
#                         / (params.cells_x * params.cells_y), 2)
#     change = sqrt(change)
#     return change

# cdef double angle_convergence_angular(double[:,:]& arr1, double[:,:]& arr2, \
#         double[:]& weight, params2d params):
#     cdef size_t cell, angle
#     cdef double change = 0.0
#     cdef double flux, flux_old
#     for cell in range(params.cells_x * params.cells_y):
#         flux = 0.0
#         flux_old = 0.0
#         for angle in range(params.angles):
#             flux += arr1[cell,angle] * weight[angle]
#             flux_old += arr2[cell,angle] * weight[angle]
#         change += pow((flux - flux_old) / flux \
#                         / (params.cells_x * params.cells_y), 2)
#     change = sqrt(change)
#     return change

# cdef void off_scatter_scalar(double[:,:]& flux, double[:,:]& flux_old, \
#         int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
#         params2d params, size_t group):
#     cdef size_t cell, mat, angle, og
#     source[:] = 0.0
#     for cell in range(params.cells_x * params.cells_y):
#         mat = medium_map[cell]
#         for og in range(0, group):
#             source[cell] += xs_matrix[mat,group,og] * flux[cell,og]
#         for og in range(group+1, params.groups):
#             source[cell] += xs_matrix[mat,group,og] * flux_old[cell,og]

# cdef void off_scatter_angular(double[:,:,:]& flux, double[:,:,:]& flux_old, \
#         int[:]& medium_map, double[:,:,:]& xs_matrix, double[:]& source, \
#         double[:]& weight, params2d params, size_t group):
#     cdef size_t cell, mat, angle, og
#     source[:] = 0.0
#     for cell in range(params.cells_x * params.cells_y):
#         mat = medium_map[cell]
#         for angle in range(params.angles):
#             for og in range(0, group):
#                 source[cell] += xs_matrix[mat,group,og] * weight[angle] \
#                                 * flux[cell,angle,og]
#             for og in range(group+1, params.groups):
#                 source[cell] += xs_matrix[mat,group,og] * weight[angle] \
#                                 * flux_old[cell,angle,og]

# cdef void fission_source(double[:]& power_source, double[:,:]& flux, \
#         double[:,:,:]& xs_fission, int[:]& medium_map, params2d params, \
#         double[:] keff):
#     cdef size_t cell, mat, ig, og
#     power_source[:] = 0.0
#     for cell in range(params.cells_x * params.cells_y):
#         mat = medium_map[cell]
#         for ig in range(params.groups):
#             for og in range(params.groups):
#                 power_source[ig::params.groups][cell] += (1 / keff[0]) \
#                                 * flux[cell,og] * xs_fission[mat,ig,og]

# cdef void normalize_flux(double[:,:]& flux, params2d params):
#     cdef size_t cell, group
#     cdef double keff = 0.0
#     for group in range(params.groups):
#         for cell in range(params.cells_x * params.cells_y):
#             keff += pow(flux[cell,group], 2)
#     keff = sqrt(keff)
#     for group in range(params.groups):
#         for cell in range(params.cells_x * params.cells_y):
#             flux[cell,group] /= keff

# # cdef void normalize_flux(double[:,:,:]& flux, params2d params):
# #     cdef size_t xx, yy, gg
# #     cdef double keff = 0.0
# #     for xx in range(params.cells_x):
# #         for yy in range(params.cells_y):
# #             for gg in range(params.groups):
# #                 keff += pow(flux[xx][yy][gg], 2)
# #     keff = sqrt(keff)
# #     for xx in range(params.cells_x):
# #         for yy in range(params.cells_y):
# #             for gg in range(params.groups):
# #                 flux[xx][yy][gg] /= keff

# cdef double update_keffective(double[:,:]& flux, double[:,:]& flux_old, \
#         int[:]& medium_map, double[:,:,:]& xs_fission, params2d params, \
#         double keff_old):
#     cdef size_t cell, mat, ig, og
#     cdef double rate_new = 0.0
#     cdef double rate_old = 0.0
#     for cell in range(params.cells_x * params.cells_y):
#         mat = medium_map[cell]
#         for ig in range(params.groups):
#             for og in range(params.groups):
#                 rate_new += flux[cell,og] * xs_fission[mat,ig,og]
#                 rate_old += flux_old[cell,og] * xs_fission[mat,ig,og]
#     return (rate_new * keff_old) / rate_old

# cdef void calculate_source_c(double[:,:]& flux_u, double[:,:,:]& xs_scatter_u, \
#         double[:]& source_c, int[:]& medium_map, int[:]& index_c, \
#         params2d params_u, params2d params_c):
#     cdef size_t cell, mat, ig, og
#     # Multiply flux by scatter
#     flux = flux_u.copy()
#     flux[:,:] = 0.0
#     for cell in range(params_u.cells_x * params_u.cells_y):
#         mat = medium_map[cell]
#         for og in range(params_u.groups):
#             for ig in range(params_u.groups):
#                 flux[cell,ig] += flux_u[cell,og] * xs_scatter_u[mat,ig,og]
#     # Shrink to size G hat
#     big_to_small(flux, source_c, index_c, params_u, params_c)

# cdef void calculate_source_t(double[:,:]& flux_u, double[:,:]& flux_c, \
#         double[:,:,:]& xs_scatter_u, double[:]& source_t, \
#         int[:]& medium_map, int[:]& index_u, double[:]& factor_u, \
#         params2d params_u, params2d params_c):
#     cdef size_t cell, mat, angle, group, ig, og
#     source_t[:] = 0.0
#     # Resize collided flux to size (I x G)
#     flux = small_to_big(flux_c, index_u, factor_u, params_u, params_c)
#     for cell in range(params_u.cells_x * params_u.cells_y):
#         mat = medium_map[cell]
#         for og in range(params_u.groups):
#             for ig in range(params_u.groups):
#                 source_t[ig::params_u.groups][cell] += xs_scatter_u[mat,ig,og] \
#                                     * (flux_u[cell,og] + flux[cell,og])

# cdef void calculate_source_star(double[:,:,:]& flux_last, double[:]& source_star, \
#         double[:]& source_t, double[:]& source_u, double[:]& velocity, params2d params):
#     cdef size_t cell, angle, group, start
#     cdef size_t stop = params.groups * params.angles
#     source_star[:] = 0.0
#     for group in range(params.groups):
#         for angle in range(params.angles):
#             for cell in range(params.cells_x * params.cells_y):
#                 start = group + angle * params.groups
#                 source_star[start::stop][cell] = source_t[group::params.groups][cell] \
#                         + source_u[start::stop][cell] \
#                         + flux_last[cell,angle,group] \
#                         * 1 / (velocity[group] * params.dt)
                        

# cdef void big_to_small(double[:,:]& flux_u, double[:]& flux_c, \
#         int[:]& index_c, params2d params_u, params2d params_c):
#     cdef size_t cell, group
#     flux_c[:] = 0.0
#     for cell in range(params_u.cells_x * params_u.cells_y):
#         for group in range(params_u.groups):
#             flux_c[index_c[group]::params_c.groups][cell] += flux_u[cell,group]
#             # flux_c[group::params_c.groups][cell] += flux_u[cell,group]


# cdef double[:,:] small_to_big(double[:,:]& flux_c, int[:]& index_u, \
#         double[:]& factor_u, params2d params_u, params2d params_c):
#     cdef size_t cell, group_u, group_c
#     flux_u = array_2d(params_u.cells_x * params_u.cells_y, params_u.groups)
#     for cell in range(params_c.cells_x * params_c.cells_y):
#         for group_c in range(params_c.groups):
#             for group_u in range(index_u[group_c], index_u[group_c+1]):
#                 flux_u[cell,group_u] = flux_c[cell,group_c] * factor_u[group_u]
#     return flux_u[:,:]