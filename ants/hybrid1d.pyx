########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Hybrid Multigroup Neutron Transport Problems
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

from ants cimport time_dependent_1d as td
from ants cimport cytools_1d as tools
from ants.cytools_1d cimport params1d
from ants.utils import dimensions
from ants import transport

import numpy as np

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')

def backward_euler(double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, \
            double[:,:,:] xs_fission_u, double[:] velocity_u, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, double[:] energy_edges, int[:] idx_edges, \
            dict params_dict_u, dict params_dict_c):
    # Create angles and weights
    angle_xu, angle_wu = transport.calculate_x_angles(params_dict_u)
    angle_xc, angle_wc = transport.calculate_x_angles(params_dict_c)
    ####################################################################
    # UNCOLLIDED PORTION
    ####################################################################
    # Uncollided dictionary to params1d
    params_u = tools._to_params1d(params_dict_u)
    # Combine fission and scattering for uncollided
    xs_matrix_u = memoryview(np.zeros((params_u.materials, \
                            params_u.groups, params_u.groups)))
    tools.combine_self_scattering(xs_matrix_u, xs_scatter_u, xs_fission_u, params_u)
    # Combine total and 1/vdt for uncollided
    xs_total_vu = memoryview(np.zeros((params_u.materials, params_u.groups)))
    tools.combine_total_velocity(xs_total_vu, xs_total_u, velocity_u, params_u)
    ####################################################################
    # COLLIDED PORTION
    ####################################################################
    # Collided dictionary to params1d
    params_c = tools._to_params1d(params_dict_c)
    # Create collided cross sections and velocity    
    xs_total_c = dimensions.xs_vector_coarsen(xs_total_u, energy_edges, idx_edges)
    xs_scatter_c = dimensions.xs_matrix_coarsen(xs_scatter_u, energy_edges, idx_edges)
    xs_fission_c = dimensions.xs_matrix_coarsen(xs_fission_u, energy_edges, idx_edges)
    velocity_c = dimensions.velocity_mean_coarsen(velocity_u, idx_edges)
    # Combine fission and scattering for collided
    xs_matrix_c = memoryview(np.zeros((params_c.materials, \
                            params_c.groups, params_c.groups)))
    tools.combine_self_scattering(xs_matrix_c, xs_scatter_c, xs_fission_c, params_c)
    # Combine total and 1/vdt for collided
    xs_total_vc = memoryview(np.zeros((params_c.materials, params_c.groups)))
    tools.combine_total_velocity(xs_total_vc, xs_total_c, velocity_c, params_c)
    ####################################################################
    # Indexing Parameters
    index_c = dimensions.calculate_collided_index(params_u.groups, idx_edges)
    delta_u, delta_c = dimensions.energy_bin_widths(energy_edges, idx_edges)
    factor_u = dimensions.calculate_hybrid_factor(params_u.groups, params_c.groups, \
                        delta_u, delta_c, idx_edges)
    index_u = dimensions.calculate_uncollided_index(params_c.groups, idx_edges)
    ####################################################################
    # print(type(xs_scatter_c), type(xs_scatter_u))
    # print(type(xs_fission_c), type(xs_fission_u))
    # print(type(xs_matrix_c), type(xs_matrix_u))
    # print(xs_matrix_c.shape, np.asarray(xs_scatter_c[0]))
    # return 0
    ####################################################################
    flux = td.hybrid_bdf1(xs_total_vu, xs_total_vc, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, source, boundary, medium_map, \
                delta_x, angle_xu, angle_wu, angle_xc, angle_wc, \
                index_u, index_c, factor_u, params_u, params_c)
    # print("after function")
    return np.asarray(flux)

