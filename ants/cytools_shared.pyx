################################################################################
#                            ___    _   _____________
#                           /   |  / | / /_  __/ ___/
#                          / /| | /  |/ / / /  \__ \
#                         / ___ |/ /|  / / /  ___/ /
#                        /_/  |_/_/ |_/ /_/  /____/
#
# Shared fused-type implementations for functions that are identical between
# 1D and 2D neutron transport, differing only in spatial array dimensionality.
# cytools_1d.pyx and cytools_2d.pyx delegate to these implementations.
#
################################################################################

# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=True
# distutils: language = c++

from libc.math cimport sqrt, pow

from ants.parameters cimport params

################################################################################
# Convergence functions
################################################################################

cdef double group_convergence(scalar_flux_nd arr1, scalar_flux_nd arr2,
                               params info):
    """L2 relative convergence of the scalar flux over spatial cells and groups.

    Dispatches at compile time to the 1D (cells_x, groups) or 2D
    (cells_x, cells_y, groups) implementation based on array rank.
    """
    cdef int ii, jj, gg
    cdef double change = 0.0
    cdef int cells

    if scalar_flux_nd is double[:,:]:
        # 1D: scalar flux is (cells_x, groups)
        cells = info.cells_x
        for gg in range(info.groups):
            for ii in range(info.cells_x):
                if arr1[ii, gg] == 0.0:
                    continue
                change += pow((arr1[ii, gg] - arr2[ii, gg]) / arr1[ii, gg] / cells, 2)
    else:
        # 2D: scalar flux is (cells_x, cells_y, groups)
        cells = info.cells_x * info.cells_y
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                for gg in range(info.groups):
                    if arr1[ii, jj, gg] == 0.0:
                        continue
                    change += pow(
                        (arr1[ii, jj, gg] - arr2[ii, jj, gg]) / arr1[ii, jj, gg] / cells,
                        2,
                    )

    return sqrt(change)


cdef double angle_convergence(spatial_nd arr1, spatial_nd arr2, params info):
    """L2 relative convergence of a spatial array over the ordinate iteration.

    Dispatches at compile time to the 1D (cells_x,) or 2D (cells_x, cells_y)
    implementation based on array rank.
    """
    cdef int ii, jj
    cdef double change = 0.0
    cdef int cells

    if spatial_nd is double[:]:
        # 1D: array is (cells_x,)
        cells = info.cells_x
        for ii in range(info.cells_x):
            if arr1[ii] == 0.0:
                continue
            change += pow((arr1[ii] - arr2[ii]) / arr1[ii] / cells, 2)
    else:
        # 2D: array is (cells_x, cells_y)
        cells = info.cells_x * info.cells_y
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                if arr1[ii, jj] == 0.0:
                    continue
                change += pow(
                    (arr1[ii, jj] - arr2[ii, jj]) / arr1[ii, jj] / cells,
                    2,
                )

    return sqrt(change)

################################################################################
# Criticality functions
################################################################################

cdef void _normalize_flux(scalar_flux_nd flux, params info):
    """Normalize flux in-place to unit L2 norm for power iteration.

    Dispatches at compile time to 1D or 2D based on flux array rank.
    """
    cdef int ii, jj, gg
    cdef double norm = 0.0

    if scalar_flux_nd is double[:,:]:
        # 1D: flux is (cells_x, groups)
        for gg in range(info.groups):
            for ii in range(info.cells_x):
                norm += pow(flux[ii, gg], 2)
        norm = sqrt(norm)
        for gg in range(info.groups):
            for ii in range(info.cells_x):
                flux[ii, gg] /= norm
    else:
        # 2D: flux is (cells_x, cells_y, groups)
        for gg in range(info.groups):
            for jj in range(info.cells_y):
                for ii in range(info.cells_x):
                    norm += flux[ii, jj, gg] * flux[ii, jj, gg]
        norm = sqrt(norm)
        for gg in range(info.groups):
            for jj in range(info.cells_y):
                for ii in range(info.cells_x):
                    flux[ii, jj, gg] /= norm


cdef double _update_keffective(scalar_flux_nd flux_new, scalar_flux_nd flux_old,
                                double[:,:,:] xs_fission,
                                medium_map_nd medium_map,
                                params info, double keff):
    """Compute updated k-effective from ratio of new/old fission reaction rates.

    Dispatches at compile time to 1D or 2D based on flux array rank.
    xs_fission is always 3D (materials, groups, groups) in both cases.
    """
    cdef int ii, jj, mat, ig, og
    cdef double rate_new = 0.0
    cdef double rate_old = 0.0

    if scalar_flux_nd is double[:,:] and medium_map_nd is int[:]:
        # 1D: flux is (cells_x, groups), medium_map is (cells_x,)
        for ii in range(info.cells_x):
            mat = medium_map[ii]
            for og in range(info.groups):
                for ig in range(info.groups):
                    rate_new += flux_new[ii, ig] * xs_fission[mat, og, ig]
                    rate_old += flux_old[ii, ig] * xs_fission[mat, og, ig]
    elif scalar_flux_nd is double[:,:,:] and medium_map_nd is int[:,:]:
        # 2D: flux is (cells_x, cells_y, groups), medium_map is (cells_x, cells_y)
        for ii in range(info.cells_x):
            for jj in range(info.cells_y):
                mat = medium_map[ii, jj]
                for og in range(info.groups):
                    for ig in range(info.groups):
                        rate_new += flux_new[ii, jj, ig] * xs_fission[mat, og, ig]
                        rate_old += flux_old[ii, jj, ig] * xs_fission[mat, og, ig]

    return (rate_new * keff) / rate_old

################################################################################
# Time Dependent functions
################################################################################

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity,
                           double constant, params info):
    """Add 1/(v*dt) term to total cross section in-place.

    Signature is identical between 1D and 2D (xs_total is always (materials,
    groups)), so no fused dispatch is needed.
    """
    cdef int mm, gg
    for gg in range(info.groups):
        for mm in range(info.materials):
            xs_total[mm, gg] += constant / (velocity[gg] * info.dt)
