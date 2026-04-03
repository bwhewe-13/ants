########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Custom datatypes to reduce argument count in solver functions
#
########################################################################

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CrossSections:
    """Bundle of material cross-section arrays.

    Attributes
    ----------
    total : ndarray, shape (materials, groups)
    scatter : ndarray, shape (materials, groups, groups)
    fission : ndarray, shape (materials, groups, groups)
    """
    total: np.ndarray
    scatter: np.ndarray
    fission: np.ndarray


@dataclass
class QuadratureData:
    """Angular quadrature angles and weights.

    Attributes
    ----------
    angle_x : ndarray, shape (angles,)
    angle_w : ndarray, shape (angles,)
    angle_y : ndarray, shape (angles,), optional -- 2D only
    """
    angle_x: np.ndarray
    angle_w: np.ndarray
    angle_y: Optional[np.ndarray] = None


@dataclass
class SpatialGrid:
    """Spatial cell widths.

    Attributes
    ----------
    delta_x : ndarray, shape (cells_x,)
    delta_y : ndarray, shape (cells_y,), optional -- 2D only
    """
    delta_x: np.ndarray
    delta_y: Optional[np.ndarray] = None


@dataclass
class HybridMapping:
    """Coarse-to-fine group mapping for hybrid solvers.

    Attributes
    ----------
    fine_idx : ndarray of int, shape (groups_fine,)
    coarse_idx : ndarray of int, shape (groups_fine,)
    factor : ndarray, shape (groups_fine,)
    """
    fine_idx: np.ndarray
    coarse_idx: np.ndarray
    factor: np.ndarray
