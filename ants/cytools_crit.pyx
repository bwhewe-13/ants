########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Functions needed for criticality problems in one-dimensional neutron 
# transport 
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.cytools cimport params1d

from libc.math cimport sqrt, pow

cdef void fission_source(double[:] power_source, double[:] flux, \
                    double[:,:,:] xs_fission, int[:] medium_map, \
                    params1d params, double[:] keff):
    power_source[:] = 0.0
    cdef size_t cell, mat, ig, og
    for cell in range(params.cells):
        mat = medium_map[cell]
        for ig in range(params.groups):
            for og in range(params.groups):
                power_source[ig::params.groups][cell] += (1 / keff[0]) \
                                    * flux[og::params.groups][cell] \
                                    * xs_fission[mat][ig][og]

cdef void normalize_flux(double[:]& flux, params1d params):
    cdef size_t cell
    cdef double keff = 0.0
    for cell in range(params.cells * params.groups):
        keff += pow(flux[cell], 2)
    keff = sqrt(keff)
    for cell in range(params.cells * params.groups):
        flux[cell] /= keff

cdef double update_keffective(double[:] flux, double[:] flux_old, \
                            int[:] medium_map, double[:,:,:] xs_fission, \
                            params1d params, double keff_old):
    cdef double frate = 0.0
    cdef double frate_old = 0.0
    cdef size_t cell, mat, ig, og
    for cell in range(params.cells):
        mat = medium_map[cell]
        for ig in range(params.groups):
            for og in range(params.groups):
                frate += flux[og::params.groups][cell] * xs_fission[mat][ig][og]
                frate_old += flux_old[og::params.groups][cell] * xs_fission[mat][ig][og]
    return (frate * keff_old) / frate_old