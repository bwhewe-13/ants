########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Functions needed for fixed source problems in three-dimensional
# neutron transport
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

from cython.view cimport array as cvarray
from libc.math cimport pow, sqrt

from ants.cytools_1d cimport _legendre_sum
from ants.cytools_1d cimport _xs_matrix_aniso as _shared_xs_matrix_aniso
from ants.cytools_shared cimport _fission_matrix as _shared_fission_matrix
from ants.cytools_shared cimport _source_columns as _shared_source_columns
from ants.cytools_shared cimport _total_velocity as _shared_total_velocity
from ants.cytools_shared cimport array_1d as _shared_array_1d
from ants.cytools_shared cimport array_2d as _shared_array_2d
from ants.cytools_shared cimport array_3d as _shared_array_3d
from ants.cytools_shared cimport array_4d as _shared_array_4d
from ants.cytools_shared cimport array_5d as _shared_array_5d
from ants.parameters cimport params


################################################################################
# Memoryview functions -- delegates to cytools_shared
################################################################################
cdef double[:] array_1d(int dim1):
    return _shared_array_1d(dim1)


cdef double[:,:] array_2d(int dim1, int dim2):
    return _shared_array_2d(dim1, dim2)


cdef double[:,:,:] array_3d(int dim1, int dim2, int dim3):
    return _shared_array_3d(dim1, dim2, dim3)


cdef double[:,:,:,:] array_4d(int dim1, int dim2, int dim3, int dim4):
    return _shared_array_4d(dim1, dim2, dim3, dim4)


cdef double[:,:,:,:,:] array_5d(int dim1, int dim2, int dim3, int dim4, \
        int dim5):
    return _shared_array_5d(dim1, dim2, dim3, dim4, dim5)


cdef double[:,:,:,:,:,:] array_6d(int dim1, int dim2, int dim3, int dim4, \
        int dim5, int dim6):
    dd6 = cvarray((dim1, dim2, dim3, dim4, dim5, dim6), itemsize=sizeof(double), \
                  format="d")
    cdef double[:,:,:,:,:,:] arr = dd6
    arr[:,:,:,:,:,:] = 0.0
    return arr

################################################################################
# Convergence functions
################################################################################
cdef double group_convergence(double[:,:,:,:]& arr1, double[:,:,:,:]& arr2, \
        params info):
    # L2 relative change of the scalar flux (cells_x, cells_y, cells_z, groups)
    cdef int ii, jj, kk, gg
    cdef double change = 0.0
    cdef int cells = info.cells_x * info.cells_y * info.cells_z
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for gg in range(info.groups):
                    if arr1[ii,jj,kk,gg] == 0.0:
                        continue
                    change += pow((arr1[ii,jj,kk,gg] - arr2[ii,jj,kk,gg]) \
                                  / arr1[ii,jj,kk,gg] / cells, 2)
    return sqrt(change)


cdef double angle_convergence(double[:,:,:]& arr1, double[:,:,:]& arr2, \
        params info) noexcept nogil:
    # L2 relative change of the one group flux (cells_x, cells_y, cells_z)
    cdef int ii, jj, kk
    cdef double change = 0.0
    cdef int cells = info.cells_x * info.cells_y * info.cells_z
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                if arr1[ii,jj,kk] == 0.0:
                    continue
                change += pow((arr1[ii,jj,kk] - arr2[ii,jj,kk]) \
                              / arr1[ii,jj,kk] / cells, 2)
    return sqrt(change)

################################################################################
# Multigroup functions
################################################################################

cdef void _xs_matrix(double[:,:,:]& mat1, double[:,:,:]& mat2, \
        double[:,:,:]& mat3, params info):
    # Initialize iterables
    cdef int ig, og, mat
    for mat in range(info.materials):
        for og in range(info.groups):
            for ig in range(info.groups):
                mat1[mat,og,ig] = (mat2[mat,og,ig] + mat3[mat,og,ig])


cdef double[:,:,:] _fission_matrix(object fission, object chi):
    return _shared_fission_matrix(fission, chi)


cdef void _dmd_subtraction(double[:,:,:,:,:]& y_minus, \
        double[:,:,:,:,:]& y_plus, double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, int snapshot, params info):
    # Initialize iterables
    cdef int ii, jj, kk, gg
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for gg in range(info.groups):
                    if (snapshot < info.dmd_snapshots - 1):
                        y_minus[ii,jj,kk,gg,snapshot] = (flux[ii,jj,kk,gg] \
                                - flux_old[ii,jj,kk,gg])
                    if (snapshot > 0):
                        y_plus[ii,jj,kk,gg,snapshot-1] = (flux[ii,jj,kk,gg] \
                                - flux_old[ii,jj,kk,gg])


cdef void _off_scatter(double[:,:,:,:]& flux, double[:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:,:]& off_scatter, params info, int group):
    # Initialize iterables
    cdef int ii, jj, kk, mat, og
    # Zero out previous values
    off_scatter[:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(0, group):
                    off_scatter[ii,jj,kk] += xs_matrix[mat,group,og] \
                                            * flux[ii,jj,kk,og]
                for og in range(group + 1, info.groups):
                    off_scatter[ii,jj,kk] += xs_matrix[mat,group,og] \
                                            * flux_old[ii,jj,kk,og]


cdef void _off_scatter_jacobi(double[:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:]& xs_matrix, \
        double[:,:,:,:]& off_scatter_all, params info, int group) noexcept nogil:
    # Jacobi variant: uses flux_old for all off-diagonal groups so that each
    # group's scattering source is independent of the sweep order.
    cdef int ii, jj, kk, mat, og
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                off_scatter_all[group,ii,jj,kk] = 0.0
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    if og != group:
                        off_scatter_all[group,ii,jj,kk] += \
                                xs_matrix[mat,group,og] * flux_old[ii,jj,kk,og]


cdef void _source_total(double[:,:,:,:,:]& source, double[:,:,:,:]& flux, \
        double[:,:,:]& xs_matrix, int[:,:,:]& medium_map, \
        double[:,:,:,:,:]& external, params info, int[:,:] q_index=None):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, kk, nn, ig, og, mat, og_q
    cdef double one_group
    # External source column of each angle and group
    cdef int[:,:] q_cols = _source_columns(external.shape[3], q_index, info)

    # Zero out previous values
    source[:,:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]

                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    one_group = 0.0
                    for ig in range(info.groups):
                        one_group += flux[ii,jj,kk,ig] * xs_matrix[mat,og,ig]

                    for nn in range(info.angles):
                        source[ii,jj,kk,nn,og] += one_group \
                                    + external[ii,jj,kk,q_cols[nn,og],og_q]


cdef int[:,:] _source_columns(int columns, int[:,:] q_index, params info):
    # External source column for each (angle, group), see cytools_shared.
    # In 3D, info.angles is the number of ordinates.
    return _shared_source_columns(columns, q_index, info.angles, info.groups)

################################################################################
# Anisotropic scattering functions
################################################################################
# Flux moments are (cells_x, cells_y, cells_z, groups, moments) and
# xs_matrix is (materials, groups, groups, moments), with the full set of
# (L + 1)**2 spherical harmonics.

cdef void _xs_matrix_aniso(double[:,:,:,:]& mat1, double[:,:,:,:]& mat2, \
        double[:,:,:]& mat3, params info):
    # Fission is isotropic, so it only adds to the k = 0 moment
    _shared_xs_matrix_aniso(mat1, mat2, mat3, info)


cdef void _off_scatter_moments(double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& flux_old, int[:,:,:]& medium_map, \
        double[:,:,:,:]& xs_matrix, double[:,:,:,:]& off_scatter, \
        params info, int group):
    # Gauss-Seidel off-group scatter moments (cells_x, cells_y, cells_z, moments)
    cdef int ii, jj, kk, mat, og, mm
    off_scatter[:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for mm in range(info.moments):
                    for og in range(0, group):
                        off_scatter[ii,jj,kk,mm] += xs_matrix[mat,group,og,mm] \
                                                    * flux[ii,jj,kk,og,mm]
                    for og in range(group + 1, info.groups):
                        off_scatter[ii,jj,kk,mm] += xs_matrix[mat,group,og,mm] \
                                                    * flux_old[ii,jj,kk,og,mm]


cdef void _off_scatter_jacobi_moments(double[:,:,:,:,:]& flux_old, \
        int[:,:,:]& medium_map, double[:,:,:,:]& xs_matrix, \
        double[:,:,:,:,:]& off_scatter_all, params info, int group) noexcept nogil:
    # Jacobi off-group scatter moments, see _off_scatter_jacobi
    cdef int ii, jj, kk, mat, og, mm
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for mm in range(info.moments):
                    off_scatter_all[group,ii,jj,kk,mm] = 0.0
                    for og in range(info.groups):
                        if og != group:
                            off_scatter_all[group,ii,jj,kk,mm] += \
                                    xs_matrix[mat,group,og,mm] \
                                    * flux_old[ii,jj,kk,og,mm]


cdef void _source_total_aniso(double[:,:,:,:,:]& source, \
        double[:,:,:,:,:]& flux, double[:,:,:,:]& xs_matrix, \
        int[:,:,:]& medium_map, double[:,:,:,:,:]& external, \
        double[:,:]& P_source, params info, int[:,:] q_index=None):
    # Create sum_k (2l + 1) Y_k (sigma_s + sigma_f)_l * phi_k + external
    cdef int ii, jj, kk, nn, ig, og, og_q, mat, mm
    cdef int[:,:] q_cols = _source_columns(external.shape[3], q_index, info)
    group_moments = array_1d(info.moments)
    source[:,:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    # Scatter moments into group og
                    for mm in range(info.moments):
                        group_moments[mm] = 0.0
                        for ig in range(info.groups):
                            group_moments[mm] += flux[ii,jj,kk,ig,mm] \
                                                 * xs_matrix[mat,og,ig,mm]
                    for nn in range(info.angles):
                        source[ii,jj,kk,nn,og] = _legendre_sum(group_moments, \
                                    P_source, nn, info) \
                                    + external[ii,jj,kk,q_cols[nn,og],og_q]


cdef void _dmd_subtraction_moments(double[:,:,:,:,:]& y_minus, \
        double[:,:,:,:,:]& y_plus, double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& flux_old, int snapshot, params info):
    # DMD differences of the flux moments, with the groups and moments
    # flattened to (cells_x, cells_y, cells_z, groups * moments, snapshots)
    cdef int ii, jj, kk, gg, mm, loc
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for gg in range(info.groups):
                    for mm in range(info.moments):
                        loc = gg * info.moments + mm
                        if (snapshot < info.dmd_snapshots - 1):
                            y_minus[ii,jj,kk,loc,snapshot] = \
                                    flux[ii,jj,kk,gg,mm] - flux_old[ii,jj,kk,gg,mm]
                        if (snapshot > 0):
                            y_plus[ii,jj,kk,loc,snapshot-1] = \
                                    flux[ii,jj,kk,gg,mm] - flux_old[ii,jj,kk,gg,mm]

################################################################################
# Boundary functions
################################################################################
# The known edges on each face are (cells_y, cells_z) for x, (cells_x,
# cells_z) for y, and (cells_x, cells_y) for z. The reflected arrays are
# (2, face, face, angles) with the lower side first.

cdef void initialize_known(double[:,:]& known, double[:,:,:]& boundary, \
        double[:,:,:,:]& reflected, double angle, int angle_idx, \
        int bc_lower, int bc_upper) noexcept nogil:
    # Incoming edges on one face from the reflector or the boundary source
    if (bc_lower == 1) and (angle > 0.0):
        known[:,:] = reflected[0,:,:,angle_idx]
    elif (bc_upper == 1) and (angle < 0.0):
        known[:,:] = reflected[1,:,:,angle_idx]
    else:
        # Pick lower / upper side
        known[:,:] = boundary[0 if angle > 0.0 else 1,:,:]


cdef void update_reflector(double[:,:]& known_x, double[:,:,:,:]& reflected_x, \
        double[:,:]& known_y, double[:,:,:,:]& reflected_y, \
        double[:,:]& known_z, double[:,:,:,:]& reflected_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        int angle, params info) noexcept nogil:
    # Initialize iterables
    cdef int opp_idx
    # Update reflected_x
    if (angle_x[angle] > 0.0) and (info.bc_x[1] == 1):
        opp_idx = _reflected_index(angle_x, angle_y, angle_z, angle, info)
        reflected_x[1,:,:,opp_idx] = known_x[:,:]
    elif (angle_x[angle] < 0.0) and (info.bc_x[0] == 1):
        opp_idx = _reflected_index(angle_x, angle_y, angle_z, angle, info)
        reflected_x[0,:,:,opp_idx] = known_x[:,:]
    # Update reflected_y
    if (angle_y[angle] > 0.0) and (info.bc_y[1] == 1):
        opp_idx = _reflected_index(angle_y, angle_x, angle_z, angle, info)
        reflected_y[1,:,:,opp_idx] = known_y[:,:]
    elif (angle_y[angle] < 0.0) and (info.bc_y[0] == 1):
        opp_idx = _reflected_index(angle_y, angle_x, angle_z, angle, info)
        reflected_y[0,:,:,opp_idx] = known_y[:,:]
    # Update reflected_z
    if (angle_z[angle] > 0.0) and (info.bc_z[1] == 1):
        opp_idx = _reflected_index(angle_z, angle_x, angle_y, angle, info)
        reflected_z[1,:,:,opp_idx] = known_z[:,:]
    elif (angle_z[angle] < 0.0) and (info.bc_z[0] == 1):
        opp_idx = _reflected_index(angle_z, angle_x, angle_y, angle, info)
        reflected_z[0,:,:,opp_idx] = known_z[:,:]


cdef int _reflected_index(double[:]& angle_opp, double[:]& angle_sim1, \
        double[:]& angle_sim2, int angle, params info) noexcept nogil:
    # Ordinate with angle_opp flipped and the other two cosines the same
    cdef int nn
    for nn in range(info.angles):
        if (angle_opp[nn] == -angle_opp[angle]) \
                and (angle_sim1[nn] == angle_sim1[angle]) \
                and (angle_sim2[nn] == angle_sim2[angle]):
            return nn
    # Returns error
    return -1

################################################################################
# Time Dependent functions
################################################################################
# The angular flux is (cells_x, cells_y, cells_z, angles, groups) at the cell
# centers. For CN and TR-BDF2 the flux is at the x, y, and z faces, (cells_x
# + 1, cells_y, cells_z, angles, groups), (cells_x, cells_y + 1, cells_z,
# angles, groups), and (cells_x, cells_y, cells_z + 1, angles, groups).

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity, \
        double constant, params info):
    cdef double[:,:] _xs_total = xs_total
    cdef double[:] _velocity = velocity
    _shared_total_velocity(_xs_total, _velocity, constant, info)


cdef void _angular_to_scalar(double[:,:,:,:,:]& angular_flux, \
        double[:,:,:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, kk, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:,:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for nn in range(info.angles):
                    for gg in range(info.groups):
                        scalar_flux[ii,jj,kk,gg] += angle_w[nn] \
                                * angular_flux[ii,jj,kk,nn,gg]


cdef double _face_center(double[:,:,:,:,:]& psi_x, double[:,:,:,:,:]& psi_y, \
        double[:,:,:,:,:]& psi_z, int ii, int jj, int kk, int nn, \
        int gg) noexcept nogil:
    # Cell center angular flux from the average of the six faces
    return (psi_x[ii,jj,kk,nn,gg] + psi_x[ii+1,jj,kk,nn,gg] \
            + psi_y[ii,jj,kk,nn,gg] + psi_y[ii,jj+1,kk,nn,gg] \
            + psi_z[ii,jj,kk,nn,gg] + psi_z[ii,jj,kk+1,nn,gg]) / 6.0


cdef void _angular_edge_to_scalar(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:]& scalar_flux, double[:]& angle_w, params info):
    # Initialize iterables
    cdef int ii, jj, kk, nn, gg
    # Zero out scalar flux term
    scalar_flux[:,:,:,:] = 0.0
    # Iterate over all spatial cells, angles, energy groups
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for nn in range(info.angles):
                    for gg in range(info.groups):
                        scalar_flux[ii,jj,kk,gg] += angle_w[nn] \
                                * _face_center(psi_x, psi_y, psi_z, ii, jj, \
                                               kk, nn, gg)


cdef void _time_source_star_bdf1(double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& q_star, double[:,:,:,:,:]& external, \
        double[:]& velocity, params info):
    # Combining the source (I x J x K x N x G) with the angular
    #     flux (I x J x K x N x G)

    # Initialize iterables
    cdef int ii, jj, kk, nn, gg, nn_q, gg_q

    # Zero out previous values
    q_star[:,:,:,:,:] = 0.0

    # Iterate over cells, angles, groups
    for nn in range(info.angles):
        nn_q = 0 if external.shape[3] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[4] == 1 else gg

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        q_star[ii,jj,kk,nn,gg] = external[ii,jj,kk,nn_q,gg_q] \
                                + flux[ii,jj,kk,nn,gg] * 1 / (velocity[gg] * info.dt)


cdef void _time_source_total_bdf1(double[:,:,:,:]& scalar, \
        double[:,:,:,:,:]& angular, double[:,:,:]& xs_matrix, \
        double[:]& velocity, double[:,:,:,:,:]& qstar, \
        double[:,:,:,:,:]& external, int[:,:,:]& medium_map, params info):
    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, kk, nn, ig, og, mat, nn_q, og_q
    cdef double one_group

    # Zero out previous values
    qstar[:,:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]

                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    one_group = 0.0
                    for ig in range(info.groups):
                        one_group += scalar[ii,jj,kk,ig] * xs_matrix[mat,og,ig]

                    for nn in range(info.angles):
                        nn_q = 0 if external.shape[3] == 1 else nn
                        qstar[ii,jj,kk,nn,og] = one_group \
                                + external[ii,jj,kk,nn_q,og_q] \
                                + angular[ii,jj,kk,nn,og] * 1 / (velocity[og] * info.dt)


cdef void _time_source_star_cn(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:]& phi, double[:,:]& xs_total, double[:,:,:]& xs_scatter, \
        double[:]& velocity, double[:,:,:,:,:]& q_star, \
        double[:,:,:,:,:]& external_prev, double[:,:,:,:,:]& external, \
        int[:,:,:]& medium_map, double[:]& delta_x, double[:]& delta_y, \
        double[:]& delta_z, double[:]& angle_x, double[:]& angle_y, \
        double[:]& angle_z, double constant, params info):

    # Combining the source (I x J x K x N x G) with the angular flux
    # external_prev is time step \ell, source is time step \ell + 1

    # Initialize iterables
    cdef int ii, jj, kk, mat, nn, og, ig, nn_q, og_q
    cdef double one_group

    # Initialize angular flux center estimates
    cdef double psi, dpsi_x, dpsi_y, dpsi_z

    # Zero out previous values
    q_star[:,:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    one_group = 0.0
                    # Add scalar term
                    for ig in range(info.groups):
                        one_group += phi[ii,jj,kk,ig] * xs_scatter[mat,og,ig]

                    for nn in range(info.angles):
                        nn_q = 0 if external.shape[3] == 1 else nn
                        # Calculate angular flux center
                        psi = _face_center(psi_x, psi_y, psi_z, ii, jj, kk, nn, og)
                        # Calculate cell flux derivative
                        dpsi_x = (psi_x[ii+1,jj,kk,nn,og] - psi_x[ii,jj,kk,nn,og]) \
                                / delta_x[ii]
                        dpsi_y = (psi_y[ii,jj+1,kk,nn,og] - psi_y[ii,jj,kk,nn,og]) \
                                / delta_y[jj]
                        dpsi_z = (psi_z[ii,jj,kk+1,nn,og] - psi_z[ii,jj,kk,nn,og]) \
                                / delta_z[kk]
                        # Add angular terms
                        q_star[ii,jj,kk,nn,og] += one_group \
                                + external[ii,jj,kk,nn_q,og_q]
                        q_star[ii,jj,kk,nn,og] += external_prev[ii,jj,kk,nn_q,og_q] \
                                - angle_x[nn] * dpsi_x - angle_y[nn] * dpsi_y \
                                - angle_z[nn] * dpsi_z + psi \
                                * (constant / (velocity[og] * info.dt) - xs_total[mat,og])


cdef void _time_source_star_bdf2(double[:,:,:,:,:]& flux_1, \
        double[:,:,:,:,:]& flux_2, double[:,:,:,:,:]& q_star, \
        double[:,:,:,:,:]& external, double[:]& velocity, params info):
    # Combining the source (I x J x K x N x G) with the angular flux
    # flux_1 is time step \ell - 1, flux_2 is time step \ell - 2
    # Initialize iterables
    cdef int ii, jj, kk, nn, gg, nn_q, gg_q

    # Zero out previous values
    q_star[:,:,:,:,:] = 0.0

    # Iterate over all cells, angles, and groups
    for nn in range(info.angles):
        nn_q = 0 if external.shape[3] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[4] == 1 else gg

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        q_star[ii,jj,kk,nn,gg] = external[ii,jj,kk,nn_q,gg_q] \
                                + flux_1[ii,jj,kk,nn,gg] * 2 / (velocity[gg] * info.dt) \
                                - flux_2[ii,jj,kk,nn,gg] * 1 / (2 * velocity[gg] * info.dt)


cdef void _time_source_total_bdf2(double[:,:,:,:]& scalar, \
        double[:,:,:,:,:]& angular_1, double[:,:,:,:,:]& angular_2, \
        double[:,:,:]& xs_matrix, double[:]& velocity, \
        double[:,:,:,:,:]& qstar, double[:,:,:,:,:]& external, \
        int[:,:,:]& medium_map, params info):

    # Create (sigma_s + sigma_f) * phi + external function
    # Initialize iterables
    cdef int ii, jj, kk, nn, ig, og, mat, nn_q, og_q
    cdef double one_group

    # Zero out previous values
    qstar[:,:,:,:,:] = 0.0

    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]

                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    one_group = 0.0
                    for ig in range(info.groups):
                        one_group += scalar[ii,jj,kk,ig] * xs_matrix[mat,og,ig]

                    for nn in range(info.angles):
                        nn_q = 0 if external.shape[3] == 1 else nn
                        qstar[ii,jj,kk,nn,og] = one_group \
                                + external[ii,jj,kk,nn_q,og_q] \
                                + angular_1[ii,jj,kk,nn,og] * 2 / (velocity[og] * info.dt) \
                                - angular_2[ii,jj,kk,nn,og] * 1 / (2 * velocity[og] * info.dt)


cdef void _time_source_star_tr_bdf2(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:,:]& flux_2, double[:,:,:,:,:]& q_star, \
        double[:,:,:,:,:]& external, double[:]& velocity, double gamma, \
        params info):
    # Combining the source (I x J x K x N x G) with the angular flux
    # psi_x is time step \ell (faces), flux_2 is time step \ell + gamma (centers)

    # Initialize iterables
    cdef int ii, jj, kk, nn, gg, nn_q, gg_q

    # Initialize angular flux center
    cdef double psi

    # Zero out previous values
    q_star[:,:,:,:,:] = 0.0

    # Iterate over all cells, angles, and groups
    for nn in range(info.angles):
        nn_q = 0 if external.shape[3] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[4] == 1 else gg

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        psi = _face_center(psi_x, psi_y, psi_z, ii, jj, kk, nn, gg)

                        q_star[ii,jj,kk,nn,gg] = external[ii,jj,kk,nn_q,gg_q] \
                                + flux_2[ii,jj,kk,nn,gg] * 1 / (gamma * (1 - gamma) \
                                    * velocity[gg] * info.dt) \
                                - psi * (1 - gamma) / (gamma * velocity[gg] * info.dt)


cdef void _time_source_star_tr_bdf2_mem(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:,:]& flux_2, double[:,:,:,:,:]& external, \
        double[:]& velocity, double gamma, params info):
    # Combining the source (I x J x K x N x G) with the angular flux
    # psi_x is time step \ell (faces), flux_2 is time step \ell + gamma (centers)
    # and is overwritten with the source

    # Initialize iterables
    cdef int ii, jj, kk, nn, gg, nn_q, gg_q
    cdef double gamma_vel_01, gamma_vel_02

    # Initialize angular flux center
    cdef double psi

    # Iterate over all cells, angles, and groups
    for nn in range(info.angles):
        nn_q = 0 if external.shape[3] == 1 else nn

        for gg in range(info.groups):
            gg_q = 0 if external.shape[4] == 1 else gg
            gamma_vel_01 = 1 / (gamma * (1 - gamma) * velocity[gg] * info.dt)
            gamma_vel_02 = (1 - gamma) / (gamma * velocity[gg] * info.dt)

            for ii in range(info.cells_x):
                for jj in range(info.cells_y):
                    for kk in range(info.cells_z):
                        psi = _face_center(psi_x, psi_y, psi_z, ii, jj, kk, nn, gg)

                        flux_2[ii,jj,kk,nn,gg] = flux_2[ii,jj,kk,nn,gg] \
                                * gamma_vel_01 + external[ii,jj,kk,nn_q,gg_q] \
                                - psi * gamma_vel_02


cdef void _time_right_side(double[:,:,:,:,:]& q_star, double[:,:,:,:]& flux, \
        double[:,:,:]& xs_scatter, int[:,:,:]& medium_map, params info):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Initialize iterables
    cdef int ii, jj, kk, nn, ig, og, mat
    cdef double one_group
    # Iterate over dimensions
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    one_group = 0.0
                    for ig in range(info.groups):
                        one_group += flux[ii,jj,kk,ig] * xs_scatter[mat,og,ig]
                    for nn in range(info.angles):
                        q_star[ii,jj,kk,nn,og] += one_group

################################################################################
# Hybrid Time Dependent functions
################################################################################
# The collided source is (cells_x, cells_y, cells_z, 1, groups_c) for
# isotropic scattering, the same for every collided ordinate.

cdef void _hybrid_source_collided(double[:,:,:,:]& flux, \
        double[:,:,:]& xs_scatter, double[:,:,:,:,:]& source_c, \
        int[:,:,:]& medium_map, int[:]& coarse_idx, params info_u):

    # Initialize iterables
    cdef int ii, jj, kk, mat, og, ig
    cdef double one_group

    # Zero out previous source
    source_c[:,:,:,:,:] = 0.0

    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info_u.groups):
                    one_group = 0.0
                    for ig in range(info_u.groups):
                        one_group += flux[ii,jj,kk,ig] * xs_scatter[mat,og,ig]
                    source_c[ii,jj,kk,0,coarse_idx[og]] += one_group


cdef void _hybrid_source_total(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& flux_c, double[:,:,:]& xs_matrix, \
        double[:,:,:,:,:]& source, int[:,:,:]& medium_map, int[:]& coarse_idx, \
        double[:]& factor_u, params info_u, params info_c):

    # Initialize iterables
    cdef int ii, jj, kk, mat, nn, ig, og
    cdef double one_group

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]

                # Combine fluxes
                for og in range(info_u.groups):
                    flux_u[ii,jj,kk,og] = flux_u[ii,jj,kk,og] \
                            + flux_c[ii,jj,kk,coarse_idx[og]] * factor_u[og]

                # Add flux-xs product to source
                for og in range(info_u.groups):
                    one_group = 0.0
                    for ig in range(info_u.groups):
                        one_group += flux_u[ii,jj,kk,ig] * xs_matrix[mat,og,ig]
                    for nn in range(info_u.angles):
                        source[ii,jj,kk,nn,og] += one_group


cdef void _vhybrid_source_c(double[:,:,:,:]& flux_u, \
        double[:,:,:]& xs_scatter, double[:,:,:,:,:]& source_c, \
        int[:,:,:]& medium_map, int[:]& edges_gidx_c, params info_u, \
        params info_c):

    # Initialize iterables
    cdef int ii, jj, kk, mat, gg, og, ig
    cdef double source

    # Zero out previous source
    source_c[:,:,:,:,:] = 0.0

    # Iterate over all spatial cells
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for gg in range(info_c.groups):
                    source = 0.0
                    for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                        for ig in range(info_u.groups):
                            source += flux_u[ii,jj,kk,ig] * xs_scatter[mat,og,ig]
                    source_c[ii,jj,kk,0,gg] += source


cdef void _variable_off_scatter(double[:,:,:,:]& flux, \
        double[:,:,:,:]& flux_old, int[:,:,:]& medium_map, \
        double[:,:,:]& xs_matrix, double[:,:,:]& off_scatter, int group, \
        double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, int out_idx2, \
        params info):
    # Collapsed off-group scatter, flux of this iteration below the group
    # (Gauss-Seidel) and of the last iteration above it

    # Initialize iterables
    cdef int gg, in_idx1, in_idx2, ii, jj, kk, mat, og, ig
    cdef double prod_tmp, delta_coarse

    # Zero out previous values
    off_scatter[:,:,:] = 0.0

    # Iterate over collided groups
    for gg in range(info.groups):
        if gg == group:
            continue

        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])

        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                for kk in range(info.cells_z):
                    mat = medium_map[ii,jj,kk]
                    prod_tmp = 0.0
                    for og in range(out_idx1, out_idx2):
                        for ig in range(in_idx1, in_idx2):
                            prod_tmp += xs_matrix[mat,og,ig] * delta_coarse \
                                        * (edges_g[ig+1] - edges_g[ig])
                    if gg < group:
                        off_scatter[ii,jj,kk] += prod_tmp * flux[ii,jj,kk,gg]
                    else:
                        off_scatter[ii,jj,kk] += prod_tmp * flux_old[ii,jj,kk,gg]


cdef void _vhybrid_source_total(double[:,:,:,:]& flux_u, \
        double[:,:,:,:]& flux_c, double[:,:,:]& xs_matrix_u, \
        double[:,:,:,:,:]& source, int[:,:,:]& medium_map, double[:]& edges_g, \
        int[:]& edges_gidx_c, params info_u, params info_c):

    # Initialize iterables
    cdef int ii, jj, kk, mat, nn, ig, og, idx1, idx2
    cdef double one_group, delta_coarse

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]

                # Combine fluxes
                for og in range(info_c.groups):
                    idx1 = edges_gidx_c[og]
                    idx2 = edges_gidx_c[og+1]
                    delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])

                    for ig in range(idx1, idx2):
                        flux_u[ii,jj,kk,ig] = flux_u[ii,jj,kk,ig] \
                                + flux_c[ii,jj,kk,og] * delta_coarse \
                                * (edges_g[ig+1] - edges_g[ig])

                # Add flux-xs product to source
                for og in range(info_u.groups):
                    one_group = 0.0
                    for ig in range(info_u.groups):
                        one_group += flux_u[ii,jj,kk,ig] * xs_matrix_u[mat,og,ig]
                    for nn in range(info_u.angles):
                        source[ii,jj,kk,nn,og] += one_group

################################################################################
# Anisotropic time dependent functions
################################################################################
# For a single moment the isotropic function is called so the results are
# unchanged.

cdef void _group_moments(double[:]& group_moments, double[:,:,:,:,:]& flux, \
        double[:,:,:,:]& xs_matrix, int mat, int ii, int jj, int kk, \
        int group, params info) noexcept nogil:
    # Scatter moments into one group: sum over groups of sigma_l * phi_k
    cdef int mm, ig
    for mm in range(info.moments):
        group_moments[mm] = 0.0
        for ig in range(info.groups):
            group_moments[mm] += flux[ii,jj,kk,ig,mm] * xs_matrix[mat,group,ig,mm]


cdef void _angular_to_moments(double[:,:,:,:,:]& angular_flux, \
        double[:,:,:,:,:]& flux, double[:,:]& P_weights, params info):
    # phi_k = sum_n w_n Y_k(Omega_n) psi_n
    cdef int ii, jj, kk, nn, gg, mm
    if info.moments == 1:
        _angular_to_scalar(angular_flux, flux[:,:,:,:,0], P_weights[0], info)
        return
    flux[:,:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for nn in range(info.angles):
                    for gg in range(info.groups):
                        for mm in range(info.moments):
                            flux[ii,jj,kk,gg,mm] += angular_flux[ii,jj,kk,nn,gg] \
                                                    * P_weights[mm,nn]


cdef void _angular_edge_to_moments(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:,:]& flux, double[:,:]& P_weights, params info):
    # Cell center moments from angular flux at cell faces
    cdef int ii, jj, kk, nn, gg, mm
    cdef double psi
    if info.moments == 1:
        _angular_edge_to_scalar(psi_x, psi_y, psi_z, flux[:,:,:,:,0], \
                                P_weights[0], info)
        return
    flux[:,:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                for nn in range(info.angles):
                    for gg in range(info.groups):
                        psi = _face_center(psi_x, psi_y, psi_z, ii, jj, kk, nn, gg)
                        for mm in range(info.moments):
                            flux[ii,jj,kk,gg,mm] += psi * P_weights[mm,nn]


cdef void _time_source_star_cn_aniso(double[:,:,:,:,:]& psi_x, \
        double[:,:,:,:,:]& psi_y, double[:,:,:,:,:]& psi_z, \
        double[:,:,:,:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:,:]& xs_scatter, double[:]& velocity, \
        double[:,:,:,:,:]& q_star, double[:,:,:,:,:]& external_prev, \
        double[:,:,:,:,:]& external, int[:,:,:]& medium_map, \
        double[:]& delta_x, double[:]& delta_y, double[:]& delta_z, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_z, \
        double[:,:]& P_source, double constant, params info):
    # Anisotropic scattering of the previous time step, see _time_source_star_cn
    cdef int ii, jj, kk, mat, nn, og, nn_q, og_q
    cdef double psi, dpsi_x, dpsi_y, dpsi_z
    if info.moments == 1:
        _time_source_star_cn(psi_x, psi_y, psi_z, phi[:,:,:,:,0], xs_total, \
                xs_scatter[:,:,:,0], velocity, q_star, external_prev, external, \
                medium_map, delta_x, delta_y, delta_z, angle_x, angle_y, \
                angle_z, constant, info)
        return
    group_moments = array_1d(info.moments)
    q_star[:,:,:,:,:] = 0.0
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    og_q = 0 if external.shape[4] == 1 else og
                    _group_moments(group_moments, phi, xs_scatter, mat, ii, jj, \
                                   kk, og, info)
                    for nn in range(info.angles):
                        nn_q = 0 if external.shape[3] == 1 else nn
                        # Calculate angular flux center
                        psi = _face_center(psi_x, psi_y, psi_z, ii, jj, kk, nn, og)
                        # Calculate cell flux derivative
                        dpsi_x = (psi_x[ii+1,jj,kk,nn,og] - psi_x[ii,jj,kk,nn,og]) \
                                / delta_x[ii]
                        dpsi_y = (psi_y[ii,jj+1,kk,nn,og] - psi_y[ii,jj,kk,nn,og]) \
                                / delta_y[jj]
                        dpsi_z = (psi_z[ii,jj,kk+1,nn,og] - psi_z[ii,jj,kk,nn,og]) \
                                / delta_z[kk]
                        q_star[ii,jj,kk,nn,og] += _legendre_sum(group_moments, \
                                                    P_source, nn, info)
                        q_star[ii,jj,kk,nn,og] += external[ii,jj,kk,nn_q,og_q] \
                                + external_prev[ii,jj,kk,nn_q,og_q] \
                                - angle_x[nn] * dpsi_x - angle_y[nn] * dpsi_y \
                                - angle_z[nn] * dpsi_z + psi \
                                * (constant / (velocity[og] * info.dt) - xs_total[mat,og])


cdef void _time_right_side_aniso(double[:,:,:,:,:]& q_star, \
        double[:,:,:,:,:]& flux, double[:,:,:,:]& xs_scatter, \
        int[:,:,:]& medium_map, double[:,:]& P_source, params info):
    # Add the anisotropic (sigma_s + sigma_f) * phi source to q_star
    cdef int ii, jj, kk, nn, og, mat
    if info.moments == 1:
        _time_right_side(q_star, flux[:,:,:,:,0], xs_scatter[:,:,:,0], \
                         medium_map, info)
        return
    group_moments = array_1d(info.moments)
    for ii in range(info.cells_x):
        for jj in range(info.cells_y):
            for kk in range(info.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info.groups):
                    _group_moments(group_moments, flux, xs_scatter, mat, ii, jj, \
                                   kk, og, info)
                    for nn in range(info.angles):
                        q_star[ii,jj,kk,nn,og] += _legendre_sum(group_moments, \
                                                    P_source, nn, info)


################################################################################
# Anisotropic hybrid time dependent functions
################################################################################
# The collided source is (cells_x, cells_y, cells_z, angles_c, groups_c).

cdef void _hybrid_source_collided_aniso(double[:,:,:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:,:,:]& source_c, \
        int[:,:,:]& medium_map, int[:]& coarse_idx, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x J x K x N' x G')
    cdef int ii, jj, kk, mat, nn, og
    if info_u.moments == 1:
        _hybrid_source_collided(flux_u[:,:,:,:,0], xs_scatter[:,:,:,0], \
                                source_c, medium_map, coarse_idx, info_u)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info_u.groups):
                    _group_moments(group_moments, flux_u, xs_scatter, mat, ii, \
                                   jj, kk, og, info_u)
                    for nn in range(info_c.angles):
                        source_c[ii,jj,kk,nn,coarse_idx[og]] += _legendre_sum( \
                                        group_moments, P_source_c, nn, info_u)


cdef void _hybrid_source_total_aniso(double[:,:,:,:,:]& flux_u, \
        double[:,:,:,:,:]& flux_c, double[:,:,:,:]& xs_matrix, \
        double[:,:,:,:,:]& source, int[:,:,:]& medium_map, int[:]& coarse_idx, \
        double[:]& factor_u, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, jj, kk, mat, nn, og, mm
    if info_u.moments == 1:
        _hybrid_source_total(flux_u[:,:,:,:,0], flux_c[:,:,:,:,0], \
                    xs_matrix[:,:,:,0], source, medium_map, coarse_idx, \
                    factor_u, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info_u.groups):
                    for mm in range(info_u.moments):
                        flux_u[ii,jj,kk,og,mm] = flux_u[ii,jj,kk,og,mm] \
                                + flux_c[ii,jj,kk,coarse_idx[og],mm] * factor_u[og]
                for og in range(info_u.groups):
                    _group_moments(group_moments, flux_u, xs_matrix, mat, ii, \
                                   jj, kk, og, info_u)
                    for nn in range(info_u.angles):
                        source[ii,jj,kk,nn,og] += _legendre_sum(group_moments, \
                                                        P_source_u, nn, info_u)


cdef void _vhybrid_source_c_aniso(double[:,:,:,:,:]& flux_u, \
        double[:,:,:,:]& xs_scatter, double[:,:,:,:,:]& source_c, \
        int[:,:,:]& medium_map, int[:]& edges_gidx_c, double[:,:]& P_source_c, \
        params info_u, params info_c):
    # Collided source on the collided ordinates (I x J x K x N' x G')
    cdef int ii, jj, kk, mat, nn, gg, og
    if info_u.moments == 1:
        _vhybrid_source_c(flux_u[:,:,:,:,0], xs_scatter[:,:,:,0], source_c, \
                          medium_map, edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    source_c[:,:,:,:,:] = 0.0
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for gg in range(info_c.groups):
                    for og in range(edges_gidx_c[gg], edges_gidx_c[gg+1]):
                        _group_moments(group_moments, flux_u, xs_scatter, mat, \
                                       ii, jj, kk, og, info_u)
                        for nn in range(info_c.angles):
                            source_c[ii,jj,kk,nn,gg] += _legendre_sum( \
                                        group_moments, P_source_c, nn, info_u)


cdef void _coarsen_moments(double[:,:,:,:,:]& flux_u, \
        double[:,:,:,:,:]& flux_c, int[:]& edges_gidx_c, params info_c):
    # Sum the fine group moments in each coarse group
    cdef int ii, jj, kk, og, ig, mm
    flux_c[:,:,:,:,:] = 0.0
    for ii in range(info_c.cells_x):
        for jj in range(info_c.cells_y):
            for kk in range(info_c.cells_z):
                for og in range(info_c.groups):
                    for ig in range(edges_gidx_c[og], edges_gidx_c[og+1]):
                        for mm in range(info_c.moments):
                            flux_c[ii,jj,kk,og,mm] += flux_u[ii,jj,kk,ig,mm]


cdef void _variable_off_scatter_aniso(double[:,:,:,:,:]& flux, \
        double[:,:,:,:,:]& flux_old, int[:,:,:]& medium_map, \
        double[:,:,:,:]& xs_matrix, double[:,:,:,:]& off_scatter, int group, \
        double[:]& edges_g, int[:]& edges_gidx_c, int out_idx1, int out_idx2, \
        params info):
    # Collapsed off-group scatter moments, see _variable_off_scatter
    cdef int gg, in_idx1, in_idx2, ii, jj, kk, mat, og, ig, mm
    cdef double prod_tmp, delta_coarse
    off_scatter[:,:,:,:] = 0.0
    for gg in range(info.groups):
        if gg == group:
            continue
        in_idx1 = edges_gidx_c[gg]
        in_idx2 = edges_gidx_c[gg + 1]
        delta_coarse = 1.0 / (edges_g[in_idx2] - edges_g[in_idx1])
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                for kk in range(info.cells_z):
                    mat = medium_map[ii,jj,kk]
                    for mm in range(info.moments):
                        prod_tmp = 0.0
                        for og in range(out_idx1, out_idx2):
                            for ig in range(in_idx1, in_idx2):
                                prod_tmp += xs_matrix[mat,og,ig,mm] * delta_coarse \
                                            * (edges_g[ig+1] - edges_g[ig])
                        if gg < group:
                            off_scatter[ii,jj,kk,mm] += prod_tmp \
                                                        * flux[ii,jj,kk,gg,mm]
                        else:
                            off_scatter[ii,jj,kk,mm] += prod_tmp \
                                                        * flux_old[ii,jj,kk,gg,mm]


cdef void _vhybrid_source_total_aniso(double[:,:,:,:,:]& flux_u, \
        double[:,:,:,:,:]& flux_c, double[:,:,:,:]& xs_matrix_u, \
        double[:,:,:,:,:]& source, int[:,:,:]& medium_map, double[:]& edges_g, \
        int[:]& edges_gidx_c, double[:,:]& P_source_u, params info_u, \
        params info_c):
    # Add the collided moments to the uncollided and update the source
    cdef int ii, jj, kk, mat, nn, ig, og, mm, idx1, idx2
    cdef double delta_coarse
    if info_u.moments == 1:
        _vhybrid_source_total(flux_u[:,:,:,:,0], flux_c[:,:,:,:,0], \
                xs_matrix_u[:,:,:,0], source, medium_map, edges_g, \
                edges_gidx_c, info_u, info_c)
        return
    group_moments = array_1d(info_u.moments)
    for ii in range(info_u.cells_x):
        for jj in range(info_u.cells_y):
            for kk in range(info_u.cells_z):
                mat = medium_map[ii,jj,kk]
                for og in range(info_c.groups):
                    idx1 = edges_gidx_c[og]
                    idx2 = edges_gidx_c[og+1]
                    delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])
                    for ig in range(idx1, idx2):
                        for mm in range(info_u.moments):
                            flux_u[ii,jj,kk,ig,mm] = flux_u[ii,jj,kk,ig,mm] \
                                    + flux_c[ii,jj,kk,og,mm] * delta_coarse \
                                    * (edges_g[ig+1] - edges_g[ig])
                for og in range(info_u.groups):
                    _group_moments(group_moments, flux_u, xs_matrix_u, mat, ii, \
                                   jj, kk, og, info_u)
                    for nn in range(info_u.angles):
                        source[ii,jj,kk,nn,og] += _legendre_sum(group_moments, \
                                                        P_source_u, nn, info_u)
