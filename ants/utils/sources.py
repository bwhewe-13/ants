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


def steady_boundary(boundary):
    """Dense boundary source for the steady-state solvers.

    Parameters
    ----------
    boundary : numpy.ndarray or CompactSource
        Dense boundary source or a compact source.

    Returns
    -------
    numpy.ndarray
        Dense boundary source. Angle and group dimensions that do not change
        are kept with size 1.
    """
    if not isinstance(boundary, CompactSource):
        return boundary
    if boundary.steps > 1:
        raise ValueError(
            "Steady-state problems need a compact source without a time dependence"
        )
    return _gather(*boundary.step(0))


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
    """

    def __init__(self, source, angle=None, full_shape=None, name="source"):
        self.compact = isinstance(source, CompactSource)
        self.source = source if self.compact else np.asarray(source)
        self.angle = None if angle is None else np.asarray(angle)
        self.full_shape = full_shape
        if (self.angle is not None) and not self.compact:
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
        if not self.compact:
            dense = self.source[step]
            if self.angle is None:
                return dense
            return self._expand(dense, half=True)
        values, index = self.source.step(step)
        if self.angle is None:
            return _gather(values, index)
        index = _broadcast_index(index, *self.full_shape[-2:])
        return self._expand(np.take(values, index, axis=-1), half=False)

    def _expand(self, dense, half):
        # Full angle boundary (2, *spatial, N, G) with the incoming angles of
        # each side and zeros for the outgoing angles. A half angle array
        # holds the incoming angles of each side in order.
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
