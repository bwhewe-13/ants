########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# One-Dimensional Fixed Source Multigroup Neutron Transport Problems
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants cimport source_iteration as si
from ants cimport cytools as tools
from ants.cytools cimport params1d

from cython.view cimport array as cvarray
import numpy as np

def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
            double[:,:,:] xs_fission, double[:] source, double[:] boundary, \
            int[:] medium_map, double[:] cell_width, double[:] mu, \
            double[:] angle_w, dict params_dict, bint angular=False):
    # Covert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Combine fission and scattering
    xs_matrix = memoryview(np.zeros((params.materials, params.groups, \
                            params.groups)))
    tools.combine_self_scattering(xs_matrix, xs_scatter, xs_fission)
    # Initialize flux guess
    flux_old = tools.group_flux(params, angular)
    # Run source iteration multigroup
    flux = si.multigroup(flux_old, xs_total, xs_matrix, source, boundary, \
                        medium_map, cell_width, mu, angle_w, params, \
                        angular)
    # Expand to correct dimensions (I x N x G) or (I x G)
    if angular == True:
        return np.asarray(flux).reshape(params.cells, params.angles, params.groups)
    else:
        return np.asarray(flux).reshape(params.cells, params.groups)


def adjoint(double[:,:] xs_total, double[:,:,:] xs_scatter, \
            double[:,:,:] xs_fission, double[:] source, double[:] boundary, \
            int[:] medium_map, double[:] cell_width, double[:] mu, \
            double[:] angle_w, dict params_dict, bint angular=False):
    # Covert dictionary to type params1d
    params = tools._to_params1d(params_dict)
    # Combine fission and scattering
    xs_matrix = memoryview(np.zeros((params.materials, params.groups, \
                            params.groups)))
    tools.combine_self_scattering(xs_matrix, xs_scatter, xs_fission)
    # Initialize flux guess
    flux_old = tools.group_flux(params, angular)
    # Run source iteration multigroup
    flux = si.multigroup(flux_old, xs_total, xs_matrix, source, boundary, \
                        medium_map, cell_width, mu, angle_w, params, \
                        angular)
    # Expand to correct dimensions (I x N x G) or (I x G)
    if angular == True:
        return np.asarray(flux).reshape(params.cells, params.angles, params.groups)
    else:
        return np.asarray(flux).reshape(params.cells, params.groups)
