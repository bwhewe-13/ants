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
# cython: infertypes=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c++
# distutils: extra_compile_args = -O3 -march=native -ffast-math

import numpy as np
from tqdm import tqdm

from libc.math cimport sqrt

from ants cimport cytools_1d as tools
from ants cimport multi_group_1d as mg
from ants cimport parameters
from ants.parameters cimport params

from ants.datatypes import TemporalDiscretization, create_params
from ants.quadrature import legendre_data
from ants.timed1d import _scatter_moments
from ants.utils.sources import TimeSource

# Uncollided is fine grid (N x G)
# Collided is coarse grid (N' x G')
# Scattering is carried as Legendre moments, see timed1d

def time_dependent(materials_u, materials_c, sources, geometry, quadrature_u, \
        quadrature_c, solver, time_data, hybrid_data):
    # Unpack Python DataTypes to Cython memoryviews
    cdef double[:,:] xs_total_u = materials_u.total
    cdef double[:,:,:,:] xs_scatter_u = _scatter_moments(materials_u.scatter)
    cdef double[:,:,:] xs_fission_u = tools._fission_matrix(materials_u.fission, materials_u.chi)
    cdef double[:] velocity_u = materials_u.velocity
    cdef double[:,:] xs_total_c = materials_c.total
    cdef double[:,:,:,:] xs_scatter_c = _scatter_moments(materials_c.scatter)
    cdef double[:,:,:] xs_fission_c = tools._fission_matrix(materials_c.fission, materials_c.chi)
    cdef double[:] velocity_c = materials_c.velocity
    cdef double[:,:,:] initial_flux = sources.initial_flux
    cdef int[:] medium_map = geometry.medium_map
    cdef double[:] delta_x = geometry.delta_x
    cdef double[:] angle_xu = quadrature_u.angle_x
    cdef double[:] angle_wu = quadrature_u.angle_w
    cdef double[:] angle_xc = quadrature_c.angle_x
    cdef double[:] angle_wc = quadrature_c.angle_w
    cdef int[:] fine_idx = hybrid_data.fine_idx
    cdef int[:] coarse_idx = hybrid_data.coarse_idx
    cdef double[:] factor = hybrid_data.factor

    # Convert uncollided dictionary to type params
    params_u = create_params(materials_u, quadrature_u, geometry, solver, time_data)
    info_u = parameters._to_params(params_u)

    # Convert collided dictionary to type params
    params_c = create_params(materials_c, quadrature_c, geometry, solver, time_data)
    info_c = parameters._to_params(params_c)
    parameters._check_timed1d(info_c, 0, xs_total_c.shape[0])
    assert info_u.moments == info_c.moments, \
        "Uncollided and collided scattering need the same Legendre moments"

    # Legendre polynomials for the scattering source and flux moments
    P_source_u_np, P_weights_u_np = legendre_data(quadrature_u, info_u.moments)
    cdef double[:,:] P_source_u = P_source_u_np
    cdef double[:,:] P_weights_u = P_weights_u_np
    P_source_c_np, P_weights_c_np = legendre_data(quadrature_c, info_c.moments)
    cdef double[:,:] P_source_c = P_source_c_np
    cdef double[:,:] P_weights_c = P_weights_c_np

    # Combine fission and scattering - Uncollided groups
    xs_matrix_u = tools.array_4d(info_u.materials, info_u.groups, \
                                 info_u.groups, info_u.moments)
    tools._xs_matrix_aniso(xs_matrix_u, xs_scatter_u, xs_fission_u, info_u)

    # Combine fission and scattering - Collided groups
    xs_matrix_c = tools.array_4d(info_c.materials, info_c.groups, \
                                 info_c.groups, info_c.moments)
    tools._xs_matrix_aniso(xs_matrix_c, xs_scatter_c, xs_fission_c, info_c)

    # Sources of each time step, the boundary is expanded to all angles
    external_u = TimeSource(sources.external)
    boundary_xu = TimeSource(sources.boundary_x, quadrature_u.angle_x, \
                    (2, info_u.angles, info_u.groups), "boundary_x")

    # Optionally create a memmap file to stream per-step output
    flux_file = None
    if time_data.save_to_file is not None:
        flux_file = np.lib.format.open_memmap(
            time_data.save_to_file, mode='w+', dtype=np.float64,
            shape=(info_u.steps, info_u.cells_x, info_u.groups)
        )

    if params_u.time_disc == TemporalDiscretization.BDF1:
        # Run backward Euler method
        parameters._check_bdf_timed1d(info_u, initial_flux.shape[0], \
                    external_u.steps, boundary_xu.steps, xs_total_u.shape[0])
        flux_final = backward_euler(initial_flux.copy(), xs_total_u, xs_total_c, \
                            xs_matrix_u, xs_matrix_c, velocity_u, velocity_c, \
                            external_u, boundary_xu, medium_map, delta_x, \
                            angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, \
                            coarse_idx, factor, P_source_u, P_weights_u, \
                            P_source_c, P_weights_c, flux_file, info_u, info_c)
    elif params_u.time_disc == TemporalDiscretization.CN:
        # Run Crank Nicolson method
        parameters._check_cn_timed1d(info_u, initial_flux.shape[0], \
                external_u.steps, boundary_xu.steps, xs_total_u.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        flux_final = crank_nicolson(initial_flux.copy(), xs_total_u, xs_total_c, \
                            xs_matrix_u, xs_matrix_c, velocity_u, velocity_c, \
                            external_u, boundary_xu, medium_map, delta_x, \
                            angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, \
                            coarse_idx, factor, P_source_u, P_weights_u, \
                            P_source_c, P_weights_c, flux_file, info_u, info_c, \
                            info_edge)
    elif params_u.time_disc == TemporalDiscretization.BDF2:
        # Run BDF2 method
        parameters._check_bdf_timed1d(info_u, initial_flux.shape[0], \
                external_u.steps, boundary_xu.steps, xs_total_u.shape[0])

        flux_final = bdf2(initial_flux.copy(), xs_total_u, xs_total_c, xs_matrix_u, \
                    xs_matrix_c, velocity_u, velocity_c, external_u, boundary_xu, \
                    medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                    fine_idx, coarse_idx, factor, P_source_u, P_weights_u, \
                    P_source_c, P_weights_c, flux_file, info_u, info_c)
    elif params_u.time_disc == TemporalDiscretization.TR_BDF2:
        # Run TR-BDF2 method
        parameters._check_tr_bdf_timed1d(info_u, initial_flux.shape[0], \
                external_u.steps, boundary_xu.steps, xs_total_u.shape[0])

        # Create params with edges for CN method
        info_edge = parameters._to_params(params_u)
        info_edge.flux_at_edges = 1

        # Run TR-BDF2
        flux_final = tr_bdf2(initial_flux.copy(), xs_total_u, xs_total_c, xs_matrix_u, \
                    xs_matrix_c, velocity_u, velocity_c, external_u, boundary_xu, \
                    medium_map, delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                    fine_idx, coarse_idx, factor, P_source_u, P_weights_u, \
                    P_source_c, P_weights_c, flux_file, info_u, info_c, info_edge)

    return np.asarray(flux_final)


cdef double[:,:,:] _collided_source(params info_c):
    # Isotropic collided source is the same for every collided ordinate
    cdef int angles = 1 if info_c.moments == 1 else info_c.angles
    return tools.array_3d(info_c.cells_x, angles, info_c.groups)


cdef double[:,:] backward_euler(double[:,:,:]& flux_last, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:,:]& xs_scatter_u, double[:,:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, \
        object external_u, object boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        object flux_file, params info_u, params info_c):

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize flux moments
    flux_u = tools.array_3d(info_u.cells_x, info_u.groups, info_u.moments)
    tools._angular_to_moments(flux_last, flux_u, P_weights_u, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.groups, info_c.moments)

    # Initialize collided source and boundary
    source_c = _collided_source(info_c)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Buffer for per-step scalar flux output
    flux_out = tools.array_2d(info_u.cells_x, info_u.groups)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF1*    ", ascii=True):
        # Boundary source of the time step for all angles
        bc_full = boundary_xu(step)

        # Update q_star as external + 1/(v*dt) * psi
        tools._time_source_star_bdf1(flux_last, q_star, external_u(step), \
                                     velocity_u, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      bc_full, boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, P_source_u, P_weights_u, \
                      P_source_c, P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    bc_full, medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)

        # Step 5: Compute scalar flux output, optionally stream to file
        tools._angular_to_scalar(flux_last, flux_out, angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = np.asarray(flux_out)

    return flux_out


cdef double[:,:] crank_nicolson(double[:,:,:]& flux_last, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:,:]& xs_scatter_u, double[:,:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, \
        object external_u, object boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        object flux_file, params info_u, params info_c, params info_edge):

    # Initialize time step
    cdef int step

    # Create sigma_t + 2 / (v * dt) - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 2.0, info_u)

    # Create sigma_t + 2 / (v * dt) - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 2.0, info_c)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize flux moments
    flux_u = tools.array_3d(info_u.cells_x, info_u.groups, info_u.moments)
    tools._angular_edge_to_moments(flux_last, flux_u, P_weights_u, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.groups, info_c.moments)

    # Initialize collided source and boundary
    source_c = _collided_source(info_c)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="CN*      ", ascii=True):
        # Boundary source of the time step for all angles
        bc_full = boundary_xu(step)

        # Update q_star
        tools._time_source_star_cn_aniso(flux_last, flux_u, xs_total_u, \
                    xs_scatter_u, velocity_u, q_star, external_u(step), \
                    external_u(step + 1), medium_map, delta_x, angle_xu, \
                    P_source_u, 2.0, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      bc_full, boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, P_source_u, P_weights_u, \
                      P_source_c, P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    bc_full, medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)

        # Step 5: Update and repeat; optionally stream to file
        tools._angular_edge_to_moments(flux_last, flux_u, P_weights_u, info_u)
        if flux_file is not None:
            flux_file[step] = np.asarray(flux_u[:,:,0])

    return flux_u[:,:,0]


cdef double[:,:] bdf2(double[:,:,:]& flux_last_1, double[:,:]& xs_total_u, \
        double[:,:]& xs_total_c, double[:,:,:,:]& xs_scatter_u, \
        double[:,:,:,:]& xs_scatter_c, double[:]& velocity_u, double[:]& velocity_c, \
        object external_u, object boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        object flux_file, params info_u, params info_c):

    # Initialize time step
    cdef int step

    # Create sigma_t + 1 / (v * dt) - For BDF1 Step - Uncollided
    xs_total_vu = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu, velocity_u, 1.0, info_u)

    # Create sigma_t + 1 / (v * dt) - For BDF1 Step - Collided
    xs_total_vc = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc, velocity_c, 1.0, info_c)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_2 = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize uncollided flux moments
    flux_u = tools.array_3d(info_u.cells_x, info_u.groups, info_u.moments)
    tools._angular_to_moments(flux_last_1, flux_u, P_weights_u, info_u)
    flux_c = tools.array_3d(info_c.cells_x, info_c.groups, info_c.moments)

    # Initialize collided source and boundary
    source_c = _collided_source(info_c)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Buffer for per-step scalar flux output
    flux_out = tools.array_2d(info_u.cells_x, info_u.groups)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="BDF2*    ", ascii=True):
        # Boundary source of the time step for all angles
        bc_full = boundary_xu(step)

        # Update q_star
        if step == 0:
            # Run BDF1 on first time step
            tools._time_source_star_bdf1(flux_last_1, q_star, \
                                external_u(step), velocity_u, info_u)
        else:
            # Run BDF2 on rest of time steps
            tools._time_source_star_bdf2(flux_last_1, flux_last_2, q_star, \
                        external_u(step), velocity_u, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_vu, xs_total_vc, \
                      xs_scatter_u, xs_scatter_c, q_star, source_c, \
                      bc_full, boundary_xc, medium_map, \
                      delta_x, angle_xu, angle_xc, angle_wu, angle_wc, \
                      fine_idx, coarse_idx, factor, P_source_u, P_weights_u, \
                      P_source_c, P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last_2[:,:,:] = flux_last_1[:,:,:]
        flux_last_1[:,:,:] = mg._known_source_angular(xs_total_vu, q_star, \
                                    bc_full, medium_map, \
                                    delta_x, angle_xu, angle_wu, info_u)

        # Step 5: Compute scalar flux output, optionally stream to file
        tools._angular_to_scalar(flux_last_1, flux_out, angle_wu, info_u)
        if flux_file is not None:
            flux_file[step] = np.asarray(flux_out)

        # Create sigma_t + 3 / (2 * v * dt) (For BDF2 time steps)
        if step == 0:
            xs_total_vu[:,:] = xs_total_u[:,:]
            tools._total_velocity(xs_total_vu, velocity_u, 1.5, info_u)
            xs_total_vc[:,:] = xs_total_c[:,:]
            tools._total_velocity(xs_total_vc, velocity_c, 1.5, info_c)

    return flux_out


cdef double[:,:] tr_bdf2(double[:,:,:]& flux_last_ell, \
        double[:,:]& xs_total_u, double[:,:]& xs_total_c, \
        double[:,:,:,:]& xs_scatter_u, double[:,:,:,:]& xs_scatter_c, \
        double[:]& velocity_u, double[:]& velocity_c, \
        object external_u, object boundary_xu, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        object flux_file, params info_u, params info_c, params info_edge):

    # Initialize time step
    cdef int step

    # Initialize gamma
    cdef double gamma = 0.5

    # Create sigma_t + 2 / (gamma * v * dt) - CN Step
    xs_total_vu_cn = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_cn[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_cn, velocity_u, 2.0 / gamma, info_u)

    xs_total_vc_cn = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc_cn[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc_cn, velocity_c, 2.0 / gamma, info_c)

    # Create sigma_t + (2 - gamma) / ((1 - gamma) * v * dt) - BDF2 Step
    xs_total_vu_bdf2 = tools.array_2d(info_u.materials, info_u.groups)
    xs_total_vu_bdf2[:,:] = xs_total_u[:,:]
    tools._total_velocity(xs_total_vu_bdf2, velocity_u, \
                            (2.0 - gamma) / (1.0 - gamma), info_u)

    xs_total_vc_bdf2 = tools.array_2d(info_c.materials, info_c.groups)
    xs_total_vc_bdf2[:,:] = xs_total_c[:,:]
    tools._total_velocity(xs_total_vc_bdf2, velocity_c, \
                            (2.0 - gamma) / (1.0 - gamma), info_c)

    # Combine last time step and uncollided source term
    q_star = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize angular flux for previous time steps
    flux_last_gamma = tools.array_3d(info_u.cells_x, info_u.angles, info_u.groups)

    # Initialize flux moments - Uncollided
    flux_u = tools.array_3d(info_u.cells_x, info_u.groups, info_u.moments)
    tools._angular_edge_to_moments(flux_last_ell, flux_u, P_weights_u, info_u)

    flux_c = tools.array_3d(info_c.cells_x, info_c.groups, info_c.moments)
    source_c = _collided_source(info_c)
    boundary_xc = tools.array_3d(2, 1, 1)

    # Iterate over time steps
    for step in tqdm(range(info_u.steps), desc="TR-BDF2* ", ascii=True):


        # Boundary sources of the time step for all angles
        bc_full = boundary_xu(step * 2)
        bca_full = boundary_xu(step * 2 + 1)

        ########################################################################
        # Crank Nicolson
        ########################################################################
        # Update q_star
        tools._time_source_star_cn_aniso(flux_last_ell, flux_u, xs_total_u, \
                    xs_scatter_u, velocity_u, q_star, external_u(step * 2), \
                    external_u(step * 2 + 1), medium_map, delta_x, angle_xu, \
                    P_source_u, 2.0 / gamma, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_vu_cn, \
                xs_total_vc_cn, xs_scatter_u, xs_scatter_c, q_star, \
                source_c, bc_full, boundary_xc, medium_map, \
                delta_x, angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, \
                coarse_idx, factor, P_source_u, P_weights_u, P_source_c, \
                P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last_gamma = mg._known_source_angular(xs_total_vu_cn, q_star, \
                                        bc_full, medium_map, \
                                        delta_x, angle_xu, angle_wu, info_u)

        ########################################################################
        # BDF2
        ########################################################################
        # Update q_star
        tools._time_source_star_tr_bdf2(flux_last_ell, flux_last_gamma, \
                q_star, external_u(step * 2 + 2), velocity_u, gamma, info_u)

        # Run Hybrid Method
        hybrid_method(flux_u, flux_c, xs_total_vu_bdf2, \
                xs_total_vc_bdf2, xs_scatter_u, xs_scatter_c, q_star, \
                source_c, bca_full, boundary_xc, medium_map, \
                delta_x, angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, \
                coarse_idx, factor, P_source_u, P_weights_u, P_source_c, \
                P_weights_c, info_u, info_c)

        # Solve for angular flux of time step
        flux_last_ell[:,:,:] = mg._known_source_angular(xs_total_vu_bdf2, \
                                    q_star, bca_full, medium_map, \
                                    delta_x, angle_xu, angle_wu, info_edge)

        # Step 5: Update and repeat; optionally stream to file
        tools._angular_edge_to_moments(flux_last_ell, flux_u, P_weights_u, info_u)
        if flux_file is not None:
            flux_file[step] = np.asarray(flux_u[:,:,0])

    return flux_u[:,:,0]


cdef void hybrid_method(double[:,:,:]& flux_u, double[:,:,:]& flux_c, \
        double[:,:]& xs_total_vu, double[:,:]& xs_total_vc, \
        double[:,:,:,:]& xs_scatter_u, double[:,:,:,:]& xs_scatter_c, \
        double[:,:,:]& q_star, double[:,:,:]& source_c, \
        double[:,:,:]& boundary_xu, double[:,:,:]& boundary_xc, \
        int[:]& medium_map, double[:]& delta_x, double[:]& angle_xu, \
        double[:]& angle_xc, double[:]& angle_wu, double[:]& angle_wc, \
        int[:]& fine_idx, int[:]& coarse_idx, double[:]& factor, \
        double[:,:]& P_source_u, double[:,:]& P_weights_u, \
        double[:,:]& P_source_c, double[:,:]& P_weights_c, \
        params info_u, params info_c):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G x L)
    flux_u[:,:,:] = mg._known_source_moments(xs_total_vu, q_star, boundary_xu, \
                        medium_map, delta_x, angle_xu, angle_wu, P_weights_u, \
                        info_u)

    # Step 2: Compute collided source (I x N' x G')
    tools._hybrid_source_collided_aniso(flux_u, xs_scatter_u, source_c, \
                        medium_map, coarse_idx, P_source_c, info_u, info_c)

    # Step 3: Solve Collided Equation (I x G' x L)
    flux_c[:,:,:] = mg.multi_group_moments(flux_c, xs_total_vc, xs_scatter_c, \
                        source_c, boundary_xc, medium_map, delta_x, angle_xc, \
                        angle_wc, P_source_c, P_weights_c, info_c)

    # Step 4: Create a new source and solve for angular flux
    tools._hybrid_source_total_aniso(flux_u, flux_c, xs_scatter_u, q_star, \
                        medium_map, coarse_idx, factor, P_source_u, info_u, info_c)
