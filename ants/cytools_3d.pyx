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
