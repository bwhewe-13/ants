########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Kobayashi 3D benchmarks for simple geometries with void regions
#
########################################################################

"""Kobayashi three-dimensional benchmarks with void regions.

The problems and reference fluxes are from K. Kobayashi, N. Sugimura, and
Y. Nagaya, "3-D Radiation Transport Benchmark Problems and Results for
Simple Geometries with Void Regions", NEA/NSC/DOC(2000)4, OECD Nuclear
Energy Agency (2000), also published in Progress in Nuclear Energy 39,
119-144 (2001).

Each problem is a one-group source in the corner of a shield, with a void
region next to the source: a cube (Problem 1), a straight duct (Problem 2),
and a dog leg duct (Problem 3). The x = 0, y = 0, and z = 0 faces are
reflecting and the outer faces are vacuum. Case i is a pure absorber and
case ii has a scattering cross section that is half of the total. The
reference fluxes are the exact values for case i and Monte Carlo (GMVP)
values for case ii.
"""

import numpy as np

import ants
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SourceData,
    SpatialDiscretization,
)

# Outer (x, y, z) lengths of each problem (cm)
LENGTHS = {1: (100.0, 100.0, 100.0), 2: (60.0, 100.0, 60.0), 3: (60.0, 100.0, 60.0)}

# Source region in the corner of every problem, [(x1, y1, z1), dx, dy, dz]
SOURCE = [[(0.0, 0.0, 0.0), 10.0, 10.0, 10.0]]

# Void regions, which are placed on the shield before the source region
VOID = {
    1: [[(0.0, 0.0, 0.0), 50.0, 50.0, 50.0]],
    2: [[(0.0, 10.0, 0.0), 10.0, 90.0, 10.0]],
    3: [
        [(0.0, 10.0, 0.0), 10.0, 50.0, 10.0],
        [(10.0, 50.0, 0.0), 30.0, 10.0, 10.0],
        [(30.0, 50.0, 10.0), 10.0, 10.0, 30.0],
        [(30.0, 60.0, 30.0), 10.0, 40.0, 10.0],
    ],
}

# Total cross sections of the source (0), void (1), and shield (2) regions,
# the scattering ratio of case ii, and the source strength (n / cm^3 s)
XS_TOTAL = np.array([0.1, 1e-4, 0.1])
SCATTER_RATIO = 0.5
SOURCE_STRENGTH = 1.0

# Reference total flux at each point (cm^-2 s^-1). The columns are the
# (x, y, z) location (cm), the exact flux for case i, and the GMVP flux and
# its 1 sigma fractional standard deviation (%) for case ii, from Tables 2-4
# of NEA/NSC/DOC(2000)4.
REFERENCE = {
    "1A": """
           5    5    5   5.95659e+00   8.29260e+00   0.021
           5   15    5   1.37185e+00   1.87028e+00   0.005
           5   25    5   5.00871e-01   7.13986e-01   0.003
           5   35    5   2.52429e-01   3.84685e-01   0.004
           5   45    5   1.50260e-01   2.53984e-01   0.006
           5   55    5   5.95286e-02   1.37220e-01   0.073
           5   65    5   1.53283e-02   4.65913e-02   0.117
           5   75    5   4.17689e-03   1.58766e-02   0.197
           5   85    5   1.18533e-03   5.47036e-03   0.343
           5   95    5   3.46846e-04   1.85082e-03   0.619
    """,
    "1B": """
           5    5    5   5.95659e+00   8.29260e+00   0.021
          15   15   15   4.70754e-01   6.63233e-01   0.004
          25   25   25   1.69968e-01   2.68828e-01   0.003
          35   35   35   8.68334e-02   1.56683e-01   0.005
          45   45   45   5.25132e-02   1.04405e-01   0.011
          55   55   55   1.33378e-02   3.02145e-02   0.061
          65   65   65   1.45867e-03   4.06555e-03   0.074
          75   75   75   1.75364e-04   5.86124e-04   0.116
          85   85   85   2.24607e-05   8.66059e-05   0.198
          95   95   95   3.01032e-06   1.12892e-05   0.383
    """,
    "1C": """
           5   55    5   5.95286e-02   1.37220e-01   0.073
          15   55    5   5.50247e-02   1.27890e-01   0.076
          25   55    5   4.80754e-02   1.13582e-01   0.080
          35   55    5   3.96765e-02   9.59578e-02   0.088
          45   55    5   3.16366e-02   7.82701e-02   0.094
          55   55    5   2.35303e-02   5.67030e-02   0.111
          65   55    5   5.83721e-03   1.88631e-02   0.189
          75   55    5   1.56731e-03   6.46624e-03   0.314
          85   55    5   4.53113e-04   2.28099e-03   0.529
          95   55    5   1.37079e-04   7.93924e-04   0.890
    """,
    "2A": """
           5    5    5   5.95659e+00   8.61696e+00   0.063
           5   15    5   1.37185e+00   2.16123e+00   0.015
           5   25    5   5.00871e-01   8.93437e-01   0.011
           5   35    5   2.52429e-01   4.77452e-01   0.012
           5   45    5   1.50260e-01   2.88719e-01   0.013
           5   55    5   9.91726e-02   1.88959e-01   0.014
           5   65    5   7.01791e-02   1.31026e-01   0.016
           5   75    5   5.22062e-02   9.49890e-02   0.017
           5   85    5   4.03188e-02   7.12403e-02   0.019
           5   95    5   3.20574e-02   5.44807e-02   0.019
    """,
    "2B": """
           5   95    5   3.20574e-02   5.44807e-02   0.019
          15   95    5   1.70541e-03   6.58233e-03   0.244
          25   95    5   1.40557e-04   1.28002e-03   0.336
          35   95    5   3.27058e-05   4.13414e-04   0.363
          45   95    5   1.08505e-05   1.55548e-04   0.454
          55   95    5   4.14132e-06   6.02771e-05   0.599
    """,
    "3A": """
           5    5    5   5.95659e+00   8.61578e+00   0.044
           5   15    5   1.37185e+00   2.16130e+00   0.010
           5   25    5   5.00871e-01   8.93784e-01   0.008
           5   35    5   2.52429e-01   4.78052e-01   0.008
           5   45    5   1.50260e-01   2.89424e-01   0.009
           5   55    5   9.91726e-02   1.92698e-01   0.010
           5   65    5   4.22623e-02   1.04982e-01   0.077
           5   75    5   1.14703e-02   3.37544e-02   0.107
           5   85    5   3.24662e-03   1.08158e-02   0.163
           5   95    5   9.48324e-04   3.39632e-03   0.275
    """,
    "3B": """
           5   55    5   9.91726e-02   1.92698e-01   0.010
          15   55    5   2.45041e-02   6.72147e-02   0.019
          25   55    5   4.54477e-03   2.21799e-02   0.028
          35   55    5   1.42960e-03   9.90646e-03   0.033
          45   55    5   2.64846e-04   3.39066e-03   0.195
          55   55    5   9.14210e-05   1.05629e-03   0.327
    """,
    "3C": """
           5   95   35   3.27058e-05   3.44804e-04   0.793
          15   95   35   2.68415e-05   2.91825e-04   0.659
          25   95   35   1.70019e-05   2.05793e-04   0.529
          35   95   35   3.37981e-05   2.62086e-04   0.075
          45   95   35   6.04893e-06   1.05367e-04   0.402
          55   95   35   3.36460e-06   4.44962e-05   0.440
    """,
}


def build_problem(
    problem,
    scattering=False,
    cell_width=2.0,
    angles=8,
    space_disc=SpatialDiscretization.STEP_CHARACTERISTIC,
    solver=None,
):
    """Create the ANTS inputs for a Kobayashi benchmark problem.

    Parameters
    ----------
    problem : int
        Problem number (1, 2, or 3).
    scattering : bool, optional
        If True, case ii (50% scattering), otherwise case i (pure absorber).
    cell_width : float, optional
        Width of the cubic spatial cells (cm). Every 10 cm must hold an odd
        number of cells so the reference points are at cell centers.
    angles : int, optional
        Number of angles in each direction for ``ants.angular_xyz``.
    space_disc : SpatialDiscretization, optional
        Spatial discretization. The default is step characteristic, since
        diamond difference gives negative fluxes in the shield.
    solver : SolverData, optional
        Solver options. The default is ``SolverData()``.

    Returns
    -------
    materials : MaterialData
        Source (0), void (1), and shield (2) cross sections.
    sources : SourceData
        Unit source in the source region and vacuum boundaries.
    geometry : GeometryData
        Medium map and cell widths.
    quadrature : QuadratureData
        Angles ordered for the reflecting faces.
    solver : SolverData
        Solver options.
    """
    cells_per_10 = 10.0 / cell_width
    if (abs(cells_per_10 - round(cells_per_10)) > 1e-9) or (
        round(cells_per_10) % 2 == 0
    ):
        raise ValueError(
            f"cell_width={cell_width} must split 10 cm into an odd number of "
            "cells, such as 10 / 3, 2, or 10 / 9"
        )
    edges = [
        np.linspace(0, length, int(round(length / cell_width)) + 1)
        for length in LENGTHS[problem]
    ]
    cells = tuple(edge.size - 1 for edge in edges)

    # Shield everywhere, with the void and the source placed on top
    medium_map = np.full(cells, 2, dtype=np.int32)
    medium_map = ants.spatial3d(medium_map, 1, VOID[problem], *edges)
    medium_map = ants.spatial3d(medium_map, 0, SOURCE, *edges)

    ratio = SCATTER_RATIO if scattering else 0.0
    materials = MaterialData(
        total=XS_TOTAL[:, None].copy(),
        scatter=ratio * XS_TOTAL[:, None, None],
        fission=np.zeros((3, 1, 1)),
    )

    external = np.zeros(cells + (1, 1))
    external[medium_map == 0] = SOURCE_STRENGTH
    sources = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1, 1)),
        boundary_z=np.zeros((2, 1, 1, 1, 1)),
    )

    # Reflecting at x = 0, y = 0, and z = 0
    bc = [1, 0]
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.diff(edges[0]),
        delta_y=np.diff(edges[1]),
        delta_z=np.diff(edges[2]),
        bc_x=bc,
        bc_y=bc,
        bc_z=bc,
        geometry=Geometry.SLAB3D,
        space_disc=space_disc,
    )
    quadrature = ants.angular_xyz(angles, bc_x=bc, bc_y=bc, bc_z=bc)
    solver = SolverData() if solver is None else solver
    return materials, sources, geometry, quadrature, solver


def reference(case, scattering=False):
    """Reference total flux of a Kobayashi benchmark case.

    Parameters
    ----------
    case : str
        Set of points, ``"1A"``, ``"1B"``, ``"1C"``, ``"2A"``, ``"2B"``,
        ``"3A"``, ``"3B"``, or ``"3C"``. The first character is the problem.
    scattering : bool, optional
        If True, case ii (50% scattering), otherwise case i (pure absorber).

    Returns
    -------
    points : ndarray, shape (points, 3)
        (x, y, z) location of each point (cm).
    flux : ndarray, shape (points,)
        Reference total flux (cm^-2 s^-1).
    uncertainty : ndarray, shape (points,)
        Fractional standard deviation of the Monte Carlo flux for case ii,
        and zeros for the exact case i values.
    """
    table = np.array(REFERENCE[case].split(), dtype=float).reshape(-1, 6)
    points = table[:, :3]
    if scattering:
        return points, table[:, 4], 0.01 * table[:, 5]
    return points, table[:, 3], np.zeros(points.shape[0])


def flux_at(flux, points, geometry):
    """Scalar flux of the cells holding each point.

    Parameters
    ----------
    flux : ndarray, shape (cells_x, cells_y, cells_z) or (..., groups)
        Scalar flux from ``ants.fixed3d.fixed_source``.
    points : ndarray, shape (points, 3)
        (x, y, z) location of each point (cm).
    geometry : GeometryData
        Geometry used for the flux.

    Returns
    -------
    ndarray, shape (points,) or (points, groups)
        Flux of the cell holding each point. A point on an edge between
        two cells takes the cell above it, except on the upper boundary.
    """
    points = np.asarray(points, dtype=float)
    index = []
    for axis, delta in enumerate(
        (geometry.delta_x, geometry.delta_y, geometry.delta_z)
    ):
        edges = np.concatenate(([0.0], np.cumsum(delta)))
        if np.any((points[:, axis] < 0.0) | (points[:, axis] > edges[-1])):
            raise ValueError(f"points must lie inside the domain along axis {axis}")
        cell = np.searchsorted(edges, points[:, axis], side="right") - 1
        index.append(np.minimum(cell, delta.size - 1))
    return flux[tuple(index)]
