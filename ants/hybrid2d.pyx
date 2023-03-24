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

import ants
from ants cimport time_dependent_2d as td
from ants cimport cytools_2d as tools
from ants.cytools_2d cimport params2d
from ants.utils import dimensions

import numpy as np

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')

def backward_euler(double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, \
        double[:,:,:] xs_fission_u, double[:] velocity_u, double[:] external, \
        double[:] boundary_x, double[:] boundary_y, int[:] medium_map, \
        double[:] delta_x, double[:] delta_y, double[:] energy_edges, \
        int[:] idx_edges, dict params_dict_u, dict params_dict_c):
    # Create angles and weights
    angle_xu, angle_yu, angle_wu = ants._angle_xy(params_dict_u)
    angle_xc, angle_yc, angle_wc = ants._angle_xy(params_dict_c)
    ####################################################################
    # UNCOLLIDED PORTION
    ####################################################################
    # Uncollided dictionary to params2d
    params_u = tools._to_params2d(params_dict_u)
    # Combine fission and scattering for uncollided
    # xs_matrix_u = memoryview(np.zeros((params_u.materials, \
    #                         params_u.groups, params_u.groups)))
    # xs_matrix_u = tools.array_3d_mgg(params_u)
    xs_matrix_u = tools.array_3d(params_u.materials, params_u.groups, \
                                 params_u.groups)
    tools.combine_self_scattering(xs_matrix_u, xs_scatter_u, xs_fission_u, params_u)
    # Combine total and 1/vdt for uncollided
    xs_total_vu = memoryview(np.zeros((params_u.materials, params_u.groups)))
    tools.combine_total_velocity(xs_total_vu, xs_total_u, velocity_u, params_u)
    ####################################################################
    # COLLIDED PORTION
    ####################################################################
    # Collided dictionary to params2d
    params_c = tools._to_params2d(params_dict_c)
    # Create collided cross sections and velocity
    xs_total_c = dimensions.xs_vector_coarsen(xs_total_u, energy_edges, idx_edges)
    xs_scatter_c = dimensions.xs_matrix_coarsen(xs_scatter_u, energy_edges, idx_edges)
    xs_fission_c = dimensions.xs_matrix_coarsen(xs_fission_u, energy_edges, idx_edges)
    velocity_c = dimensions.velocity_mean_coarsen(velocity_u, idx_edges)
    # Combine fission and scattering for collided
    # xs_matrix_c = tools.array_3d_mgg(params_c)
    xs_matrix_c = tools.array_3d(params_c.materials, params_c.groups, \
                                 params_c.groups)
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
    flux = td.hybrid_bdf1(xs_total_vu, xs_total_vc, xs_matrix_u, xs_matrix_c, \
                          velocity_u, velocity_c, external, boundary_x, \
                          boundary_y, medium_map, delta_x, delta_y, angle_xu, \
                          angle_yu, angle_wu, angle_xc, angle_yc, angle_wc, \
                          index_u, index_c, factor_u, params_u, params_c)
    flux = np.asarray(flux).reshape(params_u.steps, params_u.cells_x, \
                    params_u.cells_y, params_u.angles, params_u.groups)
    return flux
