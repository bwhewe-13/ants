########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Spatial sweeps for two-dimensional polygon meshes (triangles,
# hexagons, etc.) with step, diamond difference, and step characteristic
# differencing. The mesh is a cytools_polygon.PolygonSweep.
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

from libc.math cimport fabs, tanh

from cython.parallel import prange, threadid

from ants cimport cytools_2d as tools
from ants.cytools_polygon cimport PolygonSweep
from ants.parameters cimport params

########################################################################
# Iterative Sweep - Polygon Geometry
#
# Same structure as square_ordinates: prange over angles with per-thread
# scalar flux buffers. Each thread keeps one face flux array of length
# F. The boundary face fluxes for each angle are kept in known_bc_work
# so the reflectors are updated sequentially after the prange.
########################################################################

cdef void polygon_ordinates(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:,:]& external, double[:,:,:]& boundary, \
        int[:,:]& medium_map, PolygonSweep mesh, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):

    cdef int nn, ii, qq, bc, tid
    cdef int N2 = info.angles * info.angles

    # Unpack mesh for the nogil sweeps
    cdef int[:,:] cell_faces = mesh.cell_faces
    cdef int[:,:] cell_opposite = mesh.cell_opposite
    cdef double[:,:] cell_width = mesh.cell_width
    cdef double[:] cell_area = mesh.cell_area
    cdef int[:,:] face_cells = mesh.face_cells
    cdef double[:,:] face_normal = mesh.face_normal
    cdef double[:] face_length = mesh.face_length
    cdef int[:] boundary_faces = mesh.boundary_faces
    cdef int[:,:] sweep_order = mesh.sweep_order

    # Per-thread scalar flux and face flux buffers
    thread_flux = tools.array_3d(info.num_threads, info.cells_x, 1)
    thread_face = tools.array_2d(info.num_threads, face_length.shape[0])

    # Boundary face fluxes (incoming before sweep, outgoing after)
    known_bc_work = tools.array_2d(N2, boundary_faces.shape[0])
    reflected = tools.array_2d(boundary_faces.shape[0], N2)

    cdef bint converged = False
    cdef int count = 1
    cdef double change = 0.0

    while not converged:

        flux[:, :] = 0.0
        thread_flux[:, :, :] = 0.0

        # Initialize boundary face fluxes (sequential - reads reflected)
        for nn in range(N2):
            bc = 0 if boundary.shape[2] == 1 else nn
            initialize_known_polygon(known_bc_work[nn, :], boundary[0, :, bc], \
                                     reflected, mesh, angle_x[nn], angle_y[nn], nn)

        # Parallel angular sweep
        for nn in prange(N2, nogil=True, schedule="static", num_threads=info.num_threads):
            qq = 0 if external.shape[2] == 1 else nn
            tid = threadid()
            polygon_sweep(thread_flux[tid, :, :], flux_old, xs_total, xs_scatter, \
                    off_scatter, external[:, :, qq], thread_face[tid, :], \
                    known_bc_work[nn, :], medium_map, sweep_order[nn, :], \
                    cell_faces, cell_opposite, cell_width, cell_area, face_cells, \
                    face_normal, face_length, boundary_faces, angle_x[nn], \
                    angle_y[nn], angle_w[nn], info)

        # Sequential reduction into scalar flux
        for nn in range(info.num_threads):
            for ii in range(info.cells_x):
                flux[ii, 0] += thread_flux[nn, ii, 0]

        # Update reflectors from exit faces left in known_bc_work
        for nn in range(N2):
            update_reflector_polygon(known_bc_work[nn, :], reflected, mesh, nn)

        change = tools.angle_convergence(flux, flux_old, info)
        converged = (change < info.tol_angular) or (count >= info.max_iter_angular)
        count += 1
        flux_old[:, :] = flux[:, :]


cdef void polygon_sweep(double[:,:]& flux, double[:,:]& flux_old, \
        double[:]& xs_total, double[:]& xs_scatter, double[:,:]& off_scatter, \
        double[:,:]& external, double[:]& known_face, double[:]& known_bc, \
        int[:,:]& medium_map, int[:]& sweep_order, int[:,:]& cell_faces, \
        int[:,:]& cell_opposite, double[:,:]& cell_width, double[:]& cell_area, \
        int[:,:]& face_cells, double[:,:]& face_normal, double[:]& face_length, \
        int[:]& boundary_faces, double angle_x, double angle_y, \
        double angle_w, params info) noexcept nogil:
    # Single-angle sweep into a private flux array.  Thread-safe.
    # Outgoing faces use psi_out = (2 * center - (1 - W) * psi_in) / (1 + W)
    # with W = 1 (step), 0 (diamond), or the step characteristic weight.
    # Centrally symmetric cells pair each outgoing face with the opposite
    # incoming face, other cells use the mean of the incoming faces.

    # Initialize iterables
    cdef int ii, cc, kk, ff, bb, mat, opp
    cdef double center, numer, denom, dot, width_in, psi_in, weight

    # Incoming boundary faces
    for bb in range(boundary_faces.shape[0]):
        known_face[boundary_faces[bb]] = known_bc[bb]

    # Iterate over cells in upwind order
    for ii in range(info.cells_x):
        cc = sweep_order[ii]
        mat = medium_map[cc, 0]

        numer = (xs_scatter[mat] * flux_old[cc, 0] + external[cc, 0] \
                    + off_scatter[cc, 0]) * cell_area[cc]
        denom = xs_total[mat] * cell_area[cc]

        # Projected width and flux of incoming faces
        width_in = 0.0
        psi_in = 0.0
        for kk in range(cell_faces.shape[1]):
            ff = cell_faces[cc, kk]
            if ff < 0:
                break
            dot = outward_dot(face_normal, face_cells, ff, cc, angle_x, angle_y)
            if dot >= 0.0:
                continue
            if cell_opposite[cc, 0] >= 0:
                # Paired faces - optical thickness across the cell
                weight = 2.0 / (1.0 + closure_weight((xs_total[mat] - info.xs_time) \
                            * cell_width[cc, kk] / -dot, info.spatial))
                numer -= weight * dot * face_length[ff] * known_face[ff]
                denom -= weight * dot * face_length[ff]
            width_in -= dot * face_length[ff]
            psi_in -= dot * face_length[ff] * known_face[ff]

        # Unpaired faces - mean chord length across the cell
        if cell_opposite[cc, 0] < 0:
            weight = closure_weight((xs_total[mat] - info.xs_time) \
                            * cell_area[cc] / width_in, info.spatial)
            numer += 2.0 / (1.0 + weight) * psi_in
            denom += 2.0 / (1.0 + weight) * width_in
            psi_in = psi_in / width_in

        # Calculate flux center
        center = numer / denom

        # Update flux with cell centers
        flux[cc, 0] += angle_w * center

        # Update known flux on outgoing faces
        for kk in range(cell_faces.shape[1]):
            ff = cell_faces[cc, kk]
            if ff < 0:
                break
            dot = outward_dot(face_normal, face_cells, ff, cc, angle_x, angle_y)
            if dot < 0.0:
                continue
            if dot == 0.0:
                known_face[ff] = center
            elif cell_opposite[cc, 0] >= 0:
                opp = cell_faces[cc, cell_opposite[cc, kk]]
                weight = closure_weight((xs_total[mat] - info.xs_time) \
                            * cell_width[cc, kk] / dot, info.spatial)
                known_face[ff] = (2.0 * center - (1.0 - weight) \
                            * known_face[opp]) / (1.0 + weight)
            else:
                known_face[ff] = (2.0 * center - (1.0 - weight) * psi_in) \
                            / (1.0 + weight)

    # Outgoing boundary faces
    for bb in range(boundary_faces.shape[0]):
        known_bc[bb] = known_face[boundary_faces[bb]]


cdef inline double outward_dot(double[:,:]& face_normal, int[:,:]& face_cells, \
        int ff, int cc, double angle_x, double angle_y) noexcept nogil:
    # Omega . n with n pointing out of cell cc
    cdef double dot = angle_x * face_normal[ff, 0] + angle_y * face_normal[ff, 1]
    if face_cells[ff, 0] != cc:
        return -dot
    return dot


cdef inline double closure_weight(double tau, int spatial) noexcept nogil:
    # Step (1), diamond difference (2), step characteristic (3)
    if spatial == 2:
        return 0.0
    elif spatial == 3:
        # Series for thin cells, coth(tau / 2) - 2 / tau ~ tau / 6
        if tau < 1e-4:
            return tau / 6.0
        return 1.0 / tanh(0.5 * tau) - 2.0 / tau
    return 1.0


cdef void initialize_known_polygon(double[:]& known_bc, double[:]& boundary, \
        double[:,:]& reflected, PolygonSweep mesh, double angle_x, \
        double angle_y, int angle):
    # Initialize iterables
    cdef int bb, ff
    cdef double dot
    for bb in range(mesh.boundary_faces.shape[0]):
        ff = mesh.boundary_faces[bb]
        dot = angle_x * mesh.face_normal[ff, 0] + angle_y * mesh.face_normal[ff, 1]
        # Update with reflected array
        if (dot < 0.0) and (mesh.side_reflect[mesh.face_side[ff]] == 1):
            known_bc[bb] = reflected[bb, angle]
        else:
            known_bc[bb] = boundary[0 if boundary.shape[0] == 1 else bb]


cdef void update_reflector_polygon(double[:]& known_bc, double[:,:]& reflected, \
        PolygonSweep mesh, int angle):
    # Initialize iterables
    cdef int bb, opp_idx
    for bb in range(mesh.boundary_faces.shape[0]):
        opp_idx = mesh.reflect_map[mesh.face_side[mesh.boundary_faces[bb]], angle]
        if opp_idx >= 0:
            reflected[bb, opp_idx] = known_bc[bb]


cdef double reflector_convergence(double[:,:]& reflected, \
        double[:,:]& reflected_old):
    # Maximum change in the reflected face fluxes relative to the largest
    cdef int bb, nn
    cdef double change = 0.0
    cdef double scale = 0.0
    for bb in range(reflected.shape[0]):
        for nn in range(reflected.shape[1]):
            change = max(change, fabs(reflected[bb, nn] - reflected_old[bb, nn]))
            scale = max(scale, fabs(reflected[bb, nn]))
    if scale == 0.0:
        return 0.0
    return change / scale


########################################################################
# Known Source Spatial Sweeps
#
# Single-pass sweeps over all angles. Reflected sides depend on angles
# swept later, so the sweep is repeated until the reflectors converge.
########################################################################

cdef void _known_polygon(double[:,:,:]& flux, double[:]& xs_total, \
        double[:,:]& zero_2d, double[:,:,:]& source, double[:,:,:]& boundary, \
        int[:,:]& medium_map, PolygonSweep mesh, double[:]& angle_x, \
        double[:]& angle_y, double[:]& angle_w, params info):
    _known_polygon_sweep(flux, flux, xs_total, zero_2d, source, boundary, \
                medium_map, mesh, angle_x, angle_y, angle_w, False, info)


cdef void _known_polygon_interface(double[:,:,:]& flux_face, \
        double[:,:,:]& flux_center, double[:]& xs_total, double[:,:,:]& source, \
        double[:,:,:]& boundary, int[:,:]& medium_map, PolygonSweep mesh, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        params info):
    # flux_center is (cells x 1 x N) for time dependent problems,
    # otherwise a dummy array
    cdef int xdim = flux_face.shape[2]
    zero_2d = tools.array_2d(info.cells_x, 1)
    if (flux_center.shape[0] == info.cells_x) and (flux_center.shape[2] == xdim):
        _known_polygon_sweep(flux_center, flux_face, xs_total, zero_2d, source, \
                    boundary, medium_map, mesh, angle_x, angle_y, angle_w, \
                    True, info)
        return
    flux = tools.array_3d(info.cells_x, 1, xdim)
    _known_polygon_sweep(flux, flux_face, xs_total, zero_2d, source, boundary, \
                medium_map, mesh, angle_x, angle_y, angle_w, True, info)


cdef void _known_polygon_sweep(double[:,:,:]& flux, double[:,:,:]& flux_edge, \
        double[:]& xs_total, double[:,:]& zero_2d, double[:,:,:]& source, \
        double[:,:,:]& boundary, int[:,:]& medium_map, PolygonSweep mesh, \
        double[:]& angle_x, double[:]& angle_y, double[:]& angle_w, \
        bint edges, params info):

    # Initialize indices etc
    cdef int nn, qq, bc, ff, count
    cdef int N2 = info.angles * info.angles
    cdef double weight

    # Add dummy dimension to run both (C x 1 x N) and (C x 1 x 1) fluxes
    cdef int xdim = flux.shape[2]

    # Face and reflector arrays
    known_face = tools.array_1d(mesh.face_length.shape[0])
    known_bc = tools.array_1d(mesh.boundary_faces.shape[0])
    reflected = tools.array_2d(mesh.boundary_faces.shape[0], N2)
    reflected_old = tools.array_2d(mesh.boundary_faces.shape[0], N2)

    # Add zero placeholder
    zero_1d = tools.array_1d(info.materials)

    for count in range(info.max_iter_angular):

        flux[:,:,:] = 0.0
        if edges:
            flux_edge[:,:,:] = 0.0
        reflected_old[:,:] = reflected[:,:]

        # Iterate over angles
        for nn in range(N2):

            # Determine dimensions of external and boundary sources
            qq = 0 if source.shape[2] == 1 else nn
            bc = 0 if boundary.shape[2] == 1 else nn

            # Initialize boundary faces
            initialize_known_polygon(known_bc, boundary[0,:,bc], reflected, \
                                     mesh, angle_x[nn], angle_y[nn], nn)

            weight = angle_w[nn] if xdim == 1 else 1.0
            polygon_sweep(flux[:,:,0 if xdim == 1 else nn], zero_2d, xs_total, \
                    zero_1d, zero_2d, source[:,:,qq], known_face, known_bc, \
                    medium_map, mesh.sweep_order[nn], mesh.cell_faces, \
                    mesh.cell_opposite, mesh.cell_width, mesh.cell_area, \
                    mesh.face_cells, mesh.face_normal, mesh.face_length, \
                    mesh.boundary_faces, angle_x[nn], angle_y[nn], weight, info)

            # Save face fluxes
            if edges:
                for ff in range(mesh.face_length.shape[0]):
                    flux_edge[ff, 0, 0 if xdim == 1 else nn] += weight * known_face[ff]

            # Save known_bc into reflected
            update_reflector_polygon(known_bc, reflected, mesh, nn)

        # Reflectors converged
        if not mesh.reflective:
            break
        if reflector_convergence(reflected, reflected_old) < info.tol_angular:
            break
