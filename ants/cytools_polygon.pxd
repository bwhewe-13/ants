########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Header file for cytools_polygon.pyx
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

from ants.parameters cimport params


cdef class PolygonSweep:
    # Geometry with the PolygonMesh and boundary conditions
    cdef object geometry

    # Mesh connectivity
    cdef int[:,:] cell_faces
    cdef int[:,:] cell_opposite
    cdef double[:,:] cell_width
    cdef double[:] cell_area
    cdef int[:,:] face_cells
    cdef double[:,:] face_normal
    cdef double[:] face_length
    cdef int[:] boundary_faces
    cdef int[:] face_side

    # Boundary conditions for each side
    cdef int[:] side_reflect
    cdef bint reflective

    # Sweep ordering and reflected angles
    cdef int[:,:] sweep_order
    cdef int[:,:] reflect_map


cdef void _time_source_star_cn_polygon(double[:,:,:,:]& psi_face, \
        double[:,:,:,:]& psi_center, double[:,:,:]& phi, double[:,:]& xs_total, \
        double[:,:,:]& xs_scatter, double[:]& velocity, double[:,:,:,:]& q_star, \
        double[:,:,:,:]& external_prev, double[:,:,:,:]& external, \
        int[:,:]& medium_map, PolygonSweep mesh, double[:]& angle_x, \
        double[:]& angle_y, double constant, params info)

cdef double[:,:,:,:] _expand_boundary_polygon(double[:,:,:,:]& half_bc, \
        PolygonSweep mesh, double[:]& angle_x, double[:]& angle_y, params info)
