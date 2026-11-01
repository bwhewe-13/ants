########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Mesh storage and time dependent functions for two-dimensional
# polygon meshes.
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

from ants.cytools_shared cimport array_4d
from ants.parameters cimport params

from ants.utils.pytools import polygon_opposite_faces, polygon_sweep_setup

########################################################################
# Polygon Mesh
#
# The polygon cells are stored as a (cells x 1) medium_map so the
# multigroup and source functions in cytools_2d are reused. The mesh
# connectivity and the sweep ordering for one angle set are built by
# each solver and passed with delta_x and delta_y (None for rectangular
# cells). Hybrid methods keep one for each angle set.
#
# Boundary sources have shape (1, F_b, N**2, G) ordered as
# mesh.boundary_faces. Edge fluxes are split into face fluxes with shape
# (F, 1, N**2, G) in place of psi_x and cell center fluxes with shape
# (cells, 1, N**2, G) in place of psi_y.
########################################################################

cdef class PolygonSweep:

    def __init__(self, geometry, angle_x, angle_y):
        mesh = geometry.mesh
        assert mesh is not None, "POLYGON2D geometry needs a PolygonMesh"
        bc_sides = geometry.bc_sides
        if bc_sides is None:
            bc_sides = [0] * mesh.sides
        assert len(bc_sides) == mesh.sides, "Need a boundary condition for each side"
        self.geometry = geometry

        # Mesh connectivity
        self.cell_faces = np.ascontiguousarray(mesh.cell_faces, dtype=np.int32)
        self.cell_opposite, self.cell_width = polygon_opposite_faces(mesh)
        self.cell_area = np.ascontiguousarray(mesh.cell_area, dtype=np.float64)
        self.face_cells = np.ascontiguousarray(mesh.face_cells, dtype=np.int32)
        self.face_normal = np.ascontiguousarray(mesh.face_normal, dtype=np.float64)
        self.face_length = np.ascontiguousarray(mesh.face_length, dtype=np.float64)
        self.boundary_faces = np.ascontiguousarray(mesh.boundary_faces, dtype=np.int32)
        self.face_side = np.ascontiguousarray(mesh.face_side, dtype=np.int32)

        # Boundary conditions for each side
        self.side_reflect = np.array(bc_sides, dtype=np.int32)
        self.reflective = np.any(np.asarray(self.side_reflect) == 1)

        # Sweep ordering and reflected angles
        self.sweep_order, self.reflect_map = polygon_sweep_setup(mesh, \
                                                angle_x, angle_y, bc_sides)

    def with_angles(self, angle_x, angle_y):
        # Same mesh with the sweep ordering for another angle set
        return PolygonSweep(self.geometry, angle_x, angle_y)


########################################################################
# Time Dependent functions
########################################################################

cdef void _time_source_star_cn_polygon(double[:,:,:,:]& psi_face, \
        double[:,:,:,:]& psi_center, double[:,:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external_prev, double[:,:,:,:]& external, \
        int[:,:]& medium_map, PolygonSweep mesh, double[:]& angle_x, \
        double[:]& angle_y, double constant, params info):
    # Same as _time_source_star_cn with the streaming term from faces
    # external_prev is time step \ell, source is time step \ell + 1

    # Initialize iterables
    cdef int cc, kk, ff, mat, nn, og, ig, nn_q, og_q
    cdef double one_group, stream, dot

    # Zero out previous values
    q_star[:,:,:,:] = 0.0

    for cc in range(info.cells_x):
        mat = medium_map[cc,0]
        for og in range(info.groups):
            og_q = 0 if external.shape[3] == 1 else og
            one_group = 0.0
            # Add scalar term
            for ig in range(info.groups):
                one_group += phi[cc,0,ig] * xs_scatter[mat,og,ig]

            for nn in range(info.angles * info.angles):
                nn_q = 0 if external.shape[2] == 1 else nn
                # Calculate streaming term
                stream = 0.0
                for kk in range(mesh.cell_faces.shape[1]):
                    ff = mesh.cell_faces[cc,kk]
                    if ff < 0:
                        break
                    dot = (angle_x[nn] * mesh.face_normal[ff,0] + angle_y[nn] \
                            * mesh.face_normal[ff,1]) * mesh.face_length[ff]
                    if mesh.face_cells[ff,0] != cc:
                        dot = -dot
                    stream += dot * psi_face[ff,0,nn,og]
                # Add angular terms
                q_star[cc,0,nn,og] += one_group + external[cc,0,nn_q,og_q]
                q_star[cc,0,nn,og] += external_prev[cc,0,nn_q,og_q] \
                        - stream / mesh.cell_area[cc] + psi_center[cc,0,nn,og] \
                        * (constant / (velocity[og] * info.dt) - xs_total[mat,og])


cdef double[:,:,:,:] _expand_boundary_polygon(double[:,:,:,:]& half_bc, \
        PolygonSweep mesh, double[:]& angle_x, double[:]& angle_y, params info):
    # Expand a half-angle (or broadcast) boundary array into a full-angle
    # boundary array with shape (1, F_b, angles**2, groups).
    #
    # half_bc angle layout:
    #   half_bc[0, bb, kk, gg] = boundary face bb, kk-th incoming angle
    #                            (Omega . n < 0), in traversal order
    cdef int N2 = info.angles * info.angles
    cdef int bb, ff, nn, gg, kk
    cdef double dot
    cdef bint bc_face  = (half_bc.shape[1] > 1)
    cdef bint bc_angle = (half_bc.shape[2] > 1)
    cdef bint bc_group = (half_bc.shape[3] > 1)
    full_bc = array_4d(1, mesh.boundary_faces.shape[0], N2, info.groups)
    for bb in range(mesh.boundary_faces.shape[0]):
        ff = mesh.boundary_faces[bb]
        kk = 0
        for nn in range(N2):
            dot = angle_x[nn] * mesh.face_normal[ff,0] \
                    + angle_y[nn] * mesh.face_normal[ff,1]
            if dot >= 0.0:
                continue
            for gg in range(info.groups):
                full_bc[0, bb, nn, gg] = half_bc[0, bb if bc_face else 0, \
                            kk if bc_angle else 0, gg if bc_group else 0]
            kk += 1
    return full_bc
