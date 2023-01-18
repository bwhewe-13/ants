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
from ants.utils import functions

import numpy as np

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')

def backward_euler(double[:,:] xs_total_u, double[:,:,:] xs_scatter_u, \
            double[:,:,:] xs_fission_u, double[:] energy_edges_u, \
            double[:] source, double[:] boundary, int[:] medium_map, \
            double[:] delta_x, dict params_dict_u, dict params_dict_c, \
            int[:] index_edges_c):
    # UNCOLLIDED PORTION
    ####################################################################
    # Uncollided dictionary to params1d
    params_u = tools._to_params1d(params_dict_u)
    # Combine fission and scattering for uncollided
    xs_matrix_u = memoryview(np.zeros((params_u.materials, \
                            params_u.groups, params_u.groups)))
    tools.combine_self_scattering(xs_matrix_u, xs_scatter_u, xs_fission_u, params_u)
    # Create uncollided velocity
    velocity_u = functions.velocity_calc(params_u.groups, np.asarray(energy_edges_u), index_edges_c)
    # Combine total and 1/vdt for uncollided
    xs_total_vu = memoryview(np.zeros((params_u.materials, params_u.groups)))
    tools.combine_total_velocity(xs_total_vu, xs_total_u, velocity_u, params_u)
    # Create uncollided angles
    angle_xu, angle_wu = functions.angles_x_calc(params_u.angles, params_u.bc)
    ####################################################################
    # COLLIDED PORTION
    ####################################################################
    # Collided dictionary to params1d
    params_c = tools._to_params1d(params_dict_c)
    # Create collided cross sections and velocity    
    xs_total_c = functions.xs_total_coarsen(xs_total_u, energy_edges_u, index_edges_c)
    xs_scatter_c = functions.xs_scatter_coarsen(xs_scatter_u, energy_edges_u, index_edges_c)
    xs_fission_c = functions.xs_scatter_coarsen(xs_fission_u, energy_edges_u, index_edges_c)
    velocity_c = functions.velocity_calc(params_c.groups, energy_edges_u, index_edges_c)
    # Combine fission and scattering for collided
    xs_matrix_c = memoryview(np.zeros((params_c.materials, \
                            params_c.groups, params_c.groups)))
    tools.combine_self_scattering(xs_matrix_c, xs_scatter_c, xs_fission_c, params_c)
    # Combine total and 1/vdt for collided
    xs_total_vc = memoryview(np.zeros((params_c.materials, params_c.groups)))
    tools.combine_total_velocity(xs_total_vc, xs_total_c, velocity_c, params_c)
    # Create collided angles
    angle_xc, angle_wc = functions.angles_x_calc(params_c.angles, params_c.bc)
    ####################################################################
    # Indexing Parameters
    index_c = functions.index_collided_calc(params_u.groups, index_edges_c)
    delta_u, delta_c = functions.energy_widths(energy_edges_u, index_edges_c)
    factor_u = functions.hybrid_factor_calc(params_u.groups, params_c.groups, \
                        delta_u, delta_c, index_edges_c)
    # factor_u = 
    index_u = functions.index_uncollided_calc(params_c.groups, index_edges_c)
    ####################################################################
    # print(np.asarray(delta_u), np.asarray(delta_u))
    # print(np.asarray(factor_u), np.asarray(index_c), np.asarray(index_u))
    # print("total star", np.asarray(xs_total_vu), np.asarray(xs_total_vc))
    # print("scatter", np.asarray(xs_scatter_u), np.asarray(xs_scatter_c))
    # print("matrix", np.asarray(xs_matrix_u), np.asarray(xs_matrix_c))
    flux = td.hybrid_bdf1(xs_total_vu, xs_total_vc, xs_matrix_u, xs_matrix_c, \
                velocity_u, velocity_c, source, boundary, medium_map, \
                delta_x, angle_xu, angle_wu, angle_xc, angle_wc, \
                index_u, index_c, factor_u, params_u, params_c)
    return np.asarray(flux)

