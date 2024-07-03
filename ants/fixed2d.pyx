########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two-Dimensional Fixed Source Multigroup Neutron Transport Problems
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

import numpy as np

from ants cimport multi_group_2d as mg
from ants cimport cytools_2d as tools
from ants cimport parameters


def source_iteration(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        dict params_dict):
    
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed2d_source_iteration(info, xs_total.shape[0])
    
    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    
    # Solve for cell center first
    info.edges = 0
    
    # Initialize flux_old to zeros
    flux_old = tools.array_3d(info.cells_x, info.cells_y, info.groups)

    # Run source iteration
    flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    
    # Return scalar flux cell centers
    if (info.angular == False) and (params_dict.get("edges", 0) == 0):
        return np.asarray(flux)
    
    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, params_dict)


def dynamic_mode_decomp(double[:,:] xs_total, double[:,:,:] xs_scatter, \
        double[:,:,:] xs_fission, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        dict params_dict):
    
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    parameters._check_fixed2d_source_iteration(info, xs_total.shape[0])
    
    # Add fission matrix to scattering
    xs_matrix = tools.array_3d(info.materials, info.groups, info.groups)
    tools._xs_matrix(xs_matrix, xs_scatter, xs_fission, info)
    
    # Solve for cell center first
    info.edges = 0
    
    # Initialize flux_old to zeros
    flux_old = tools.array_3d(info.cells_x, info.cells_y, info.groups)
    
    # Run source iteration
    flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                            boundary_x, boundary_y, medium_map, delta_x, \
                            delta_y, angle_x, angle_y, angle_w, info)
    
    # Return scalar flux cell centers
    if (info.angular == False) and (params_dict.get("edges", 0) == 0):
        return np.asarray(flux)
    
    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
                    boundary_x, boundary_y, medium_map, delta_x, delta_y, \
                    angle_x, angle_y, angle_w, params_dict)


def known_source_calculation(double[:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        dict params_dict):
    # This is for solving for angular flux or cell interfaces
    
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    
    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, \
                            info.angles * info.angles, info.groups)
    tools._source_total(source, flux, xs_matrix, medium_map, external, info)
    
    # Solve for angular flux at cell centers
    if (info.angular == True) and (info.edges == 0):
        angular_flux = mg._known_source_angular(xs_total, source, boundary_x, \
                                boundary_y, medium_map, delta_x, delta_y, \
                                angle_x, angle_y, angle_w, info)
        return np.asarray(angular_flux)
    
    # Solve for angular flux cell interfaces
    elif (info.angular == True) and (info.edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, \
                                     info.angles * info.angles, info.groups)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, \
                                     info.angles * info.angles, info.groups)
        mg._interface_angular(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x), np.asarray(flux_edge_y)
    
    # Solve for scalar flux cell interfaces
    elif (info.angular == False) and (info.edges == 1):
        flux_edge_x = tools.array_4d(info.cells_x + 1, info.cells_y, info.groups, 1)
        flux_edge_y = tools.array_4d(info.cells_x, info.cells_y + 1, info.groups, 1)
        mg._interface_scalar(flux_edge_x, flux_edge_y, xs_total, source, \
                             boundary_x, boundary_y, medium_map, delta_x, \
                             delta_y, angle_x, angle_y, angle_w, info)
        return np.asarray(flux_edge_x[...,0]), np.asarray(flux_edge_y[...,0])
    
    return -1
    

def known_source_single(double[:,:,:] flux, double[:,:] xs_total, \
        double[:,:,:] xs_matrix, double[:,:,:,:] external, \
        double[:,:,:,:] boundary_x, double[:,:,:,:] boundary_y, \
        int[:,:] medium_map, double[:] delta_x, double[:] delta_y, \
        double[:] angle_x, double[:] angle_y, double[:] angle_w, \
        int group, dict params_dict):
    # This is for solving for angular flux or cell interfaces
    
    # Covert dictionary to type params
    info = parameters._to_params(params_dict)
    
    # Create (sigma_s + sigma_f) * phi + external function
    source = tools.array_4d(info.cells_x, info.cells_y, external.shape[2], 1)
    if external.shape[2] == 1:
        tools._source_total_single(source, flux, xs_matrix, medium_map, \
                                    external, group, info)
    else:
        tools._source_total_nsingle(source, flux, xs_matrix, medium_map, \
                                    external, group, info)
    
    # Solve for angular flux at cell centers
    angular_flux = mg._known_source_single(xs_total, source, boundary_x, \
                            boundary_y, medium_map, delta_x, delta_y, \
                            angle_x, angle_y, angle_w, group, info)
    return np.asarray(angular_flux)
