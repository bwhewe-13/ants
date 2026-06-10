########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
########################################################################

"""Public package interface for ANTS.

This module exposes the most commonly used datatypes and helper functions at
the package level while loading submodules lazily to reduce import overhead and
avoid unnecessary import-time coupling.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

_MODULE_EXPORTS = (
    "boundary1d",
    "boundary2d",
    "external1d",
    "external2d",
)

_DATATYPE_EXPORTS = (
    "GeometryData",
    "HybridData",
    "MaterialData",
    "QuadratureData",
    "SolverData",
    "TimeDependentData",
)

_QUADRATURE_EXPORTS = (
    "_angular_x",
    "_angular_xy",
    "_angular_xyz",
    "angular_x",
    "angular_xy",
    "angular_xyz",
    "artificial_scatter_matrix",
)

_MAIN_EXPORTS = (
    "_energy_grid",
    "energy_grid",
    "energy_velocity",
    "gamma_time_steps",
    "spatial1d",
    "spatial2d",
    "weight_matrix2d",
    "weight_spatial2d",
)

_MATERIAL_EXPORTS = ("materials",)

__all__ = [
    "__version__",
    *_MODULE_EXPORTS,
    *_DATATYPE_EXPORTS,
    *_QUADRATURE_EXPORTS,
    *_MAIN_EXPORTS,
    *_MATERIAL_EXPORTS,
]


def __getattr__(name):
    """Lazily resolve package-level exports from their source modules."""

    if name in _MODULE_EXPORTS:
        value = import_module(f".{name}", __name__)
    elif name in _DATATYPE_EXPORTS:
        value = getattr(import_module(".datatypes", __name__), name)
    elif name in _QUADRATURE_EXPORTS:
        value = getattr(import_module(".quadrature", __name__), name)
    elif name in _MAIN_EXPORTS:
        value = getattr(import_module(".main", __name__), name)
    elif name in _MATERIAL_EXPORTS:
        value = getattr(import_module(".materials", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__():
    """Return dynamic module attributes for auto-completion/introspection."""

    return sorted(set(globals()) | set(__all__))


try:
    __version__ = version("ants")
except PackageNotFoundError:
    __version__ = "0.2.0"
