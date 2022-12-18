########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Power iteration for one-dimensional multigroup neutron transport 
# criticality problems.
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants cimport source_iteration as si
from ants cimport cytools as tools
from ants.cytools cimport params1d
from ants cimport cytools_crit as ctools
from ants.constants import MAX_ITERATIONS, OUTER_TOLERANCE

import numpy as np

cdef double[:] multigroup(double[:] flux_guess, double[:,:] xs_total, \
                        double[:,:,:] xs_scatter, double[:,:,:] xs_fission, 
                        double[:] power_source, int[:] medium_map, \
                        double[:] cell_width, double[:] mu, \
                        double[:] angle_w, params1d params, double[:] keff):
    # Initialize flux
    flux = tools.group_flux(params, False)
    flux = flux_guess.copy()
    flux_old = flux_guess.copy()
    # Vacuum boundaries
    # cdef double boundary
    # bounary[:] = 0.0
    cdef double[:] boundary = np.zeros((2))
    # Set convergence limits
    cdef bint converged = False
    cdef size_t count = 1
    cdef double change = 0.0
    while not (converged):
        ctools.fission_source(power_source, flux_old, xs_fission, \
                            medium_map, params, keff)
        flux = si.multigroup(flux, xs_total, xs_scatter, power_source, \
                            boundary, medium_map, cell_width, mu, angle_w, \
                            params, False)
        keff[0] = ctools.update_keffective(flux, flux_old, medium_map, \
                                    xs_fission, params, keff[0])
        ctools.normalize_flux(flux, params)
        change = tools.group_convergence(flux, flux_old, angle_w, params, False)
        print("Power Iteration {}\n{}\nChange {} Keff {}".format(count, \
                "="*35, change, keff[0]))
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
        count += 1
        flux_old[:] = flux[:]
    return flux[:]


# cdef double[:] adjoint(double[:] flux_guess, double[:,:] xs_total, \
#                         double[:,:,:] xs_scatter, double[:,:,:] xs_fission, 
#                         double[:] power_source, int[:] medium_map, \
#                         double[:] cell_width, double[:] mu, \
#                         double[:] angle_w, params1d params, double[:] keff):
#     # Initialize flux
#     flux = tools.group_flux(params, False)
#     flux = flux_guess.copy()
#     flux_old = flux_guess.copy()
#     # Vacuum boundaries
#     # cdef double boundary
#     # bounary[:] = 0.0
#     cdef double[:] boundary = np.zeros((2))
#     # Set convergence limits
#     cdef bint converged = False
#     cdef size_t count = 1
#     cdef double change = 0.0
#     while not (converged):
#         ctools.fission_source(power_source, flux_old, xs_fission, \
#                             medium_map, params, keff)
#         flux = si.multigroup(flux, xs_total, xs_scatter, power_source, \
#                             boundary, medium_map, cell_width, mu, angle_w, \
#                             params, False)
#         keff[0] = ctools.update_keffective(flux, flux_old, medium_map, \
#                                     xs_fission, params, keff[0])
#         ctools.normalize_flux(flux, params)
#         change = tools.group_convergence(flux, flux_old, angle_w, params, False)
#         print("Power Iteration {}\n{}\nChange {} Keff {}".format(count, \
#                 "="*35, change, keff[0]))
#         converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
#         count += 1
#         flux_old[:] = flux[:]
#     return flux[:]