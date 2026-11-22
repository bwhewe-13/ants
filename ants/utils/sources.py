########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Converting dense and compact sources for the solvers
#
########################################################################

"""Convert external and boundary sources to the arrays the solvers use.

Sources are either dense arrays (dimensions of size 1 are broadcast) or
:class:`ants.datatypes.CompactSource`. The steady-state sweeps read a compact
external source through its index, so it is never expanded. Boundary sources
are surface arrays and are expanded to dense arrays. Time-dependent sources
are expanded one time step at a time.
"""

import numpy as np

from ants.datatypes import CompactSource


def _broadcast_index(index, angles, groups):
    # Full (angles, groups) index from one with broadcast dimensions
    try:
        index = np.broadcast_to(index, (angles, groups))
    except ValueError:
        raise ValueError(
            f"Compact source index has shape {index.shape}, "
            f"expected ({angles}, {groups}) or with dimensions "
            "of size 1"
        ) from None
    # Copy since the broadcast view is read-only
    return np.array(index, dtype=np.int32)


def _gather(values, index):
    # Dense array (*spatial, N, G) from the slabs, keeping dimensions of
    # size 1 where the index does not change
    if (index.shape[0] > 1) and (index == index[:1]).all():
        index = index[:1]
    if (index.shape[1] > 1) and (index == index[:, :1]).all():
        index = index[:, :1]
    return np.take(values, index, axis=-1)


def steady_external(external, spatial, angles, groups):
    """External source and slab index for the steady-state solvers.

    Parameters
    ----------
    external : numpy.ndarray or CompactSource
        Dense ``(*spatial, N, G)`` source or a compact source.
    spatial : tuple of int
        Spatial shape, ``(I,)`` in 1D and ``(I, J)`` in 2D.
    angles : int
        Number of angles (``N`` in 1D and ``N**2`` in 2D).
    groups : int
        Number of energy groups.

    Returns
    -------
    external : numpy.ndarray
        The dense source, or the compact slabs with shape
        ``(*spatial, S, 1)``.
    index : numpy.ndarray or None
        Slab of each (angle, group), shape ``(angles, groups)``, or None for
        a dense source.
    """
    if not isinstance(external, CompactSource):
        return external, None
    if external.steps > 1:
        raise ValueError(
            "Steady-state problems need a compact source without a time dependence"
        )
    values, index = external.step(0)
    if values.shape[:-1] != tuple(spatial):
        raise ValueError(
            f"Compact external source values have spatial "
            f"shape {values.shape[:-1]}, expected {tuple(spatial)}"
        )
    return values[..., None], _broadcast_index(index, angles, groups)


def check_steady_external(external, spatial, angles, groups):
    """Check a dense external source is ``(*spatial, angles, groups)``.

    The angle and group dimensions can have size 1. The sweeps do not check
    bounds, so other sizes would be read out of bounds.
    """
    shape = np.shape(external)
    full = (*spatial, angles, groups)
    broadcast = all(size in (1, n) for size, n in zip(shape[-2:], full[-2:]))
    if (len(shape) != len(full)) or (shape[:-2] != full[:-2]) or not broadcast:
        raise ValueError(f"external has shape {shape}, expected {full}")


def check_steady_boundary(boundary, spatial, angles, groups, name, sides=2):
    """Check a dense boundary is ``(sides, *spatial, angles, groups)``.

    Dimensions after the first can have size 1. The sweeps do not check
    bounds, so other sizes would be read out of bounds. Polygon meshes have
    one side with every boundary face, ``(1, F_b, angles, groups)``.
    """
    shape = np.shape(boundary)
    full = (sides, *spatial, angles, groups)
    broadcast = all(size in (1, n) for size, n in zip(shape[1:], full[1:]))
    if (len(shape) != len(full)) or (shape[0] != sides) or not broadcast:
        raise ValueError(f"{name} has shape {shape}, expected {full}")


def steady_boundary(boundary, spatial, angles, groups, name, sides=2):
    """Dense boundary source for the steady-state solvers.

    Parameters
    ----------
    boundary : numpy.ndarray or CompactSource
        Dense boundary source or a compact source.
    spatial, angles, groups, name, sides
        Expected shape and name, see :func:`check_steady_boundary`.

    Returns
    -------
    numpy.ndarray
        Dense boundary source. Angle and group dimensions that do not change
        are kept with size 1.
    """
    if isinstance(boundary, CompactSource):
        if boundary.steps > 1:
            raise ValueError(
                "Steady-state problems need a compact source without a time "
                "dependence"
            )
        boundary = _gather(*boundary.step(0))
    check_steady_boundary(boundary, spatial, angles, groups, name, sides)
    return boundary


def polygon_incoming(mesh, angle_x, angle_y):
    """Incoming angles of each boundary face of a polygon mesh.

    Parameters
    ----------
    mesh : PolygonMesh
        Polygon mesh with outward face normals.
    angle_x, angle_y : numpy.ndarray
        Direction cosines of the ordinates.

    Returns
    -------
    numpy.ndarray
        Boolean array ``(1, F_b, N**2)`` that is True where
        ``Omega . n < 0`` on boundary face ``F_b``.
    """
    normal = mesh.face_normal[mesh.boundary_faces]
    dot = normal[:, 0, None] * angle_x[None] + normal[:, 1, None] * angle_y[None]
    return (dot < 0.0)[None]


def dense_source(source):
    """Dense array of a dense or compact source."""
    if isinstance(source, CompactSource):
        return source.to_dense()
    return source


class TimeSource:
    """Time-dependent source expanded one time step at a time.

    Parameters
    ----------
    source : numpy.ndarray or CompactSource
        Dense source with a leading time axis of size 1 (constant) or the
        number of steps, or a compact source.
    angle : numpy.ndarray, optional
        Direction cosines normal to the boundary. If given, the source is a
        boundary source. A dense boundary only has the incoming angles (or
        size 1) and each step is expanded to all angles, shape
        ``full_shape``, with zeros for the outgoing angles.
    full_shape : tuple of int, optional
        Shape of one expanded boundary step, ``(2, N, G)`` in 1D and
        ``(2, J, N**2, G)`` or ``(2, I, N**2, G)`` in 2D.
    name : str, optional
        Name used in error messages.
    incoming : numpy.ndarray, optional
        Boolean array ``(sides, *spatial, N)`` of the incoming angles at each
        boundary location, used in place of ``angle`` when the incoming
        angles change along the boundary (polygon meshes, see
        :func:`polygon_incoming`).
    """

    def __init__(
        self, source, angle=None, full_shape=None, name="source", incoming=None
    ):
        self.compact = isinstance(source, CompactSource)
        self.source = source if self.compact else np.asarray(source)
        self.angle = None if angle is None else np.asarray(angle)
        self.incoming = None if incoming is None else np.asarray(incoming, bool)
        self.full_shape = full_shape
        if (self.incoming is not None) and not self.compact:
            half_angles = int(self.incoming.sum(axis=-1).max())
            assert self.source.shape[-2] in (1, half_angles), (
                f"{name} angle dimension must be 1 (broadcast) or "
                f"{half_angles} (incoming angles only), got "
                f"{self.source.shape[-2]}"
            )
        elif (self.angle is not None) and not self.compact:
            half_angles = int(np.sum(self.angle > 0))
            assert self.source.shape[-2] in (1, half_angles), (
                f"{name} angle dimension must be 1 (broadcast) or "
                f"{half_angles} (incoming angles only), got "
                f"{self.source.shape[-2]}"
            )

    @property
    def steps(self):
        """Number of time steps stored (1 for a constant source)."""
        if self.compact:
            return self.source.steps
        return self.source.shape[0]

    def __call__(self, step):
        """Return the dense source of a time step."""
        step = 0 if self.steps == 1 else step
        boundary = (self.angle is not None) or (self.incoming is not None)
        if not self.compact:
            dense = self.source[step]
            if not boundary:
                return dense
            return self._expand(dense, half=True)
        values, index = self.source.step(step)
        if not boundary:
            return _gather(values, index)
        index = _broadcast_index(index, *self.full_shape[-2:])
        return self._expand(np.take(values, index, axis=-1), half=False)

    def _expand(self, dense, half):
        # Full angle boundary (2, *spatial, N, G) with the incoming angles of
        # each side and zeros for the outgoing angles. A half angle array
        # holds the incoming angles of each side in order.
        if self.incoming is not None:
            return self._expand_incoming(dense, half)
        full = np.zeros(self.full_shape)
        for side, incoming in enumerate((self.angle > 0, self.angle < 0)):
            angles = np.flatnonzero(incoming)
            if not half:
                cols = angles
            elif dense.shape[-2] == 1:
                cols = np.zeros(angles.size, dtype=int)
            else:
                cols = np.arange(angles.size)
            full[side][..., angles, :] = dense[side][..., cols, :]
        return full

    def _expand_incoming(self, dense, half):
        # Same as _expand where the incoming angles change along the
        # boundary, the k-th column of a half angle array is the k-th
        # incoming angle of that location
        mask = np.broadcast_to(self.incoming, self.full_shape[:-1])
        if not half:
            values = np.broadcast_to(dense, self.full_shape)
        else:
            if dense.shape[-2] == 1:
                cols = np.zeros(mask.shape, dtype=int)
            else:
                cols = np.where(mask, np.cumsum(mask, axis=-1) - 1, 0)
            shape = (*mask.shape[:-1], dense.shape[-2], self.full_shape[-1])
            values = np.take_along_axis(
                np.broadcast_to(dense, shape), cols[..., None], axis=-2
            )
        return np.where(mask[..., None], values, 0.0)


def polygon_time_boundaries(boundary_x, boundary_y, mesh, angle_x, angle_y, groups):
    """Time-dependent boundary sources of a polygon mesh.

    ``boundary_x`` is ``(T, 1, F_b, N, G)`` with the incoming angles
    (``Omega . n < 0``) of each boundary face and is expanded to
    ``(1, F_b, N**2, G)``. ``boundary_y`` is a placeholder.
    """
    full_shape = (1, mesh.boundary_faces.size, angle_x.size, groups)
    incoming = polygon_incoming(mesh, angle_x, angle_y)
    boundary_x = TimeSource(boundary_x, None, full_shape, "boundary_x", incoming)
    return boundary_x, TimeSource(boundary_y)
