########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Time Dependent Source Multigroup Neutron Transport Problems
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

from ants cimport source_iteration_2d as si
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d

import numpy as np
from tqdm import tqdm

cdef double[:,:,:,:] multigroup_bdf1(double[:,:,:]& flux_guess, \
                        double[:,:]& xs_total_v, double[:,:,:]& xs_scatter, \
                        double[:]& velocity, double[:]& source, \
                        double[:]& boundary_x, double[:]& boundary_y, \
                        int[:]& medium_map, \
                        double[:]& delta_x, double[:]& delta_y, \
                        double[:]& angle_x, double[:]& angle_y, \
                        double[:]& angle_w, params2d params):
    cdef size_t step
    # Combine last time step and source term
    source_star = tools.array_1d_ijng(params)
    # Initialize fluxes
    flux_last = flux_guess.copy()
    flux_times = tools.array_4d_tijng(params)
    # for step in range(params.steps):
    for step in tqdm(range(params.steps)):
        tools.combine_source_flux(flux_last, source_star, \
                                    source, velocity, params)
        flux_times[step] = si.multigroup_angular(flux_last, xs_total_v, \
                        xs_scatter, source_star, boundary_x, boundary_y, \
                        medium_map, delta_x, delta_y, angle_x, angle_y, \
                        angle_w, params)
        boundary_x[:] = 0.0
        flux_last[:,:,:] = flux_times[step,:,:,:]
        # print("Multigroup", step, np.sum(flux_last))
    return flux_times[:,:,:,:]

