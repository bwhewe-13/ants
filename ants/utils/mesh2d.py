########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Utility functions for generating 2D geometry weight matrices and
# medium maps for common neutron transport benchmark layouts.
#
# Layouts:
#   cylinder-1mat   - single fissile cylinder in void
#   cylinder-2mat   - fuel core + reflector annulus in void
#   double-chevron  - 4-triangle chevron pattern in moderator
#   c5g7            - 7-material C5G7 MOX benchmark (medium map only)
#
# Utilities:
#   resize_weight_matrix - expand or shrink a weight matrix
#
# Polygon meshes (Geometry.POLYGON2D):
#   polygon_mesh          - build a PolygonMesh from vertices and cells
#   rectangle_mesh        - rectangular cells as polygons
#   triangle_mesh         - rectangles split into triangles
#   triangle_mesh_delaunay - Delaunay triangulation of a point set
#   hexagon_mesh          - hexagonal lattice clipped to a convex boundary
#   polygon_medium_map    - material map from cell centroids
#   polygon_weight_matrix - material weights from MC sampling
#
# CLI:
#   python -m ants.utils.mesh2d <layout> --cells-x N --cells-y N [opts]
#   python -m ants.utils.mesh2d --resize input.npy --cells-x N --cells-y N [opts]
#
########################################################################

import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom

from ants.datatypes import PolygonMesh
from ants.main import (
    _inside_circle,
    _inside_hexagon,
    _inside_rectangle,
    _inside_triangle,
    spatial2d,
    weight_matrix2d,
)

# Rounding used to merge vertices shared between cells
_VERTEX_PRECISION = 10


def cylinder_1mat(
    cells_x,
    cells_y,
    length_x=None,
    length_y=None,
    radius=None,
    N_particles=None,
):
    """Weight matrix for a single fissile cylinder embedded in void.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    length_x : float, optional
        Domain length in x [cm]. Default is ``2 * radius``.
    length_y : float, optional
        Domain length in y [cm]. Default is ``2 * radius``.
    radius : float, optional
        Cylinder radius [cm]. Default is 4.279960.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * 50_000``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 2)
        Material weight fractions. Material 0 = fuel cylinder,
        material 1 = void background.
    """
    if radius is None:
        radius = 4.279960
    if length_x is None:
        length_x = 2.0 * radius
    if length_y is None:
        length_y = 2.0 * radius
    assert length_x == length_y, "cylinder layouts require a square domain"
    if N_particles is None:
        N_particles = cells_x * 50_000

    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    cx = length_x / 2.0
    cy = length_y / 2.0

    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=2,
        N_particles=N_particles,
        circles=[[(cx, cy), (0.0, radius)]],
        circle_index=[0],
    )


def cylinder_2mat(
    cells_x,
    cells_y,
    length_x=None,
    length_y=None,
    r_inner=None,
    r_outer=None,
    N_particles=None,
):
    """Weight matrix for a fuel core + reflector annulus embedded in void.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    length_x : float, optional
        Domain length in x [cm]. Default is ``2 * r_outer``.
    length_y : float, optional
        Domain length in y [cm]. Default is ``2 * r_outer``.
    r_inner : float, optional
        Inner (fuel) cylinder radius [cm]. Default is 3.0.
    r_outer : float, optional
        Outer (reflector) annulus radius [cm]. Default is 4.5.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * 50_000``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 3)
        Material weight fractions. Material 0 = inner fuel, 1 = reflector annulus,
        2 = void background.
    """
    if r_inner is None:
        r_inner = 3.0
    if r_outer is None:
        r_outer = 4.5
    if length_x is None:
        length_x = 2.0 * r_outer
    if length_y is None:
        length_y = 2.0 * r_outer
    assert length_x == length_y, "cylinder layouts require a square domain"
    if N_particles is None:
        N_particles = cells_x * 50_000

    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    cx = length_x / 2.0
    cy = length_y / 2.0

    # circles[-1] must be the outer annulus so _quarter_symmetry uses r_outer
    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=3,
        N_particles=N_particles,
        circles=[
            [(cx, cy), (0.0, r_inner)],
            [(cx, cy), (r_inner, r_outer)],
        ],
        circle_index=[0, 1],
    )


def double_chevron(cells_x, cells_y, N_particles=None):
    """Weight matrix for the 9 x 9 cm double-chevron benchmark geometry.

    Four triangular uranium fuel regions in an HDPE moderator background,
    matching the geometry encoded in examples/weight_matrix_2d_chevron.npy.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * cells_y * 40``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 2)
        Material weight fractions. Material 0 = uranium fuel,
        material 1 = HDPE background.
    """
    if N_particles is None:
        N_particles = cells_x * cells_y * 40

    length_x = 9.0
    length_y = 9.0

    edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 10)
    edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 10)

    triangles = np.array(
        [
            [(0.1, 1.0), (0.1, 3.9), (5.9, 1.0)],
            [(6.0, 1.0), (8.9, 1.0), (8.9, 3.9)],
            [(0.1, 4.9), (0.1, 7.8), (5.9, 4.9)],
            [(6.0, 4.9), (8.9, 4.9), (8.9, 7.8)],
        ]
    )
    rectangles = [
        [(0, 0), 0.1, 9.0],
        [(0, 8.9), 9.0, 0.1],
        [(8.9, 0), 0.1, 9.0],
    ]

    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=2,
        N_particles=N_particles,
        triangles=triangles,
        triangle_index=[0, 0, 0, 0],
        rectangles=rectangles,
        rectangle_index=[0, 0, 0],
    )


def c5g7(cells_x=153, cells_y=153):
    """Medium map for the 2D C5G7 MOX fuel assembly benchmark.

    Based on M.A. Smith et al. NEA/NSC/DOC(2003)16, Appendix A.
    Returns an integer medium map (not a weight matrix) because the C5G7
    geometry consists entirely of rectangular regions with no curved boundaries.

    Parameters
    ----------
    cells_x : int, optional
        Number of spatial cells in x. Default is 153 (= 51 * 3).
    cells_y : int, optional
        Number of spatial cells in y. Default is 153 (= 51 * 3).

    Returns
    -------
    medium_map : ndarray of int, shape (cells_x, cells_y)
        Material index per cell. 0=moderator, 1=UO2, 2=MOX43, 3=MOX70, 4=MOX87,
        5=guide tube, 6=fission chamber.

    Raises
    ------
    ValueError
        If cells do not align with the 1.26 cm pin pitch.
    """
    pin = 1.26
    length_x = 64.26
    length_y = 64.26

    # Validate pin-pitch alignment
    pins_per_length = length_x / pin
    if abs(round(pins_per_length) - pins_per_length) > 1e-9:
        raise ValueError(
            f"length_x={length_x} is not a multiple of pin pitch {pin}. "
            f"Use cells_x that is a multiple of {round(pins_per_length)}."
        )
    cells_per_pin = cells_x / pins_per_length
    if abs(round(cells_per_pin) - cells_per_pin) > 1e-9:
        raise ValueError(
            f"cells_x={cells_x} does not divide evenly into"
            f" {pins_per_length:.0f} pins. "
            f"Try cells_x = 51*k (e.g. 51, 102, 153)."
        )

    edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 12)
    edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 12)

    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)

    # UO2 assemblies (material 1)
    uo2_assemblies = [
        [(34 * pin, 0), 17 * pin, 17 * pin],
        [(17 * pin, 17 * pin), 17 * pin, 17 * pin],
    ]
    medium_map = spatial2d(medium_map, 1, uo2_assemblies, edges_x, edges_y)

    # MOX 4.3% assemblies (material 2)
    mox43_assemblies = [
        [(17 * pin, 0), 17 * pin, 17 * pin],
        [(34 * pin, 17 * pin), 17 * pin, 17 * pin],
    ]
    medium_map = spatial2d(medium_map, 2, mox43_assemblies, edges_x, edges_y)

    # MOX 7.0% assemblies (material 3)
    mox70_assemblies = [
        [(17 * pin + pin, pin), 15 * pin, 15 * pin],
        [(34 * pin + pin, 17 * pin + pin), 15 * pin, 15 * pin],
    ]
    medium_map = spatial2d(medium_map, 3, mox70_assemblies, edges_x, edges_y)

    # MOX 8.7% assemblies (material 4)
    mox87_assemblies = [
        [(17 * pin + 4 * pin, 4 * pin), 9 * pin, 9 * pin],
        [(17 * pin + 5 * pin, 3 * pin), 7 * pin, pin],
        [(17 * pin + 3 * pin, 5 * pin), pin, 7 * pin],
        [(17 * pin + 13 * pin, 5 * pin), pin, 7 * pin],
        [(17 * pin + 5 * pin, 13 * pin), 7 * pin, pin],
        [(34 * pin + 4 * pin, 17 * pin + 4 * pin), 9 * pin, 9 * pin],
        [(34 * pin + 5 * pin, 17 * pin + 3 * pin), 7 * pin, pin],
        [(34 * pin + 3 * pin, 17 * pin + 5 * pin), pin, 7 * pin],
        [(34 * pin + 13 * pin, 17 * pin + 5 * pin), pin, 7 * pin],
        [(34 * pin + 5 * pin, 17 * pin + 13 * pin), 7 * pin, pin],
    ]
    medium_map = spatial2d(medium_map, 4, mox87_assemblies, edges_x, edges_y)

    # Guide tubes (material 5)
    gt_idx = [
        (5, 2),
        (8, 2),
        (11, 2),
        (3, 3),
        (13, 3),
        (2, 5),
        (5, 5),
        (8, 5),
        (11, 5),
        (14, 5),
        (2, 8),
        (5, 8),
        (11, 8),
        (14, 8),
        (2, 11),
        (5, 11),
        (8, 11),
        (11, 11),
        (14, 11),
        (3, 13),
        (13, 13),
        (5, 14),
        (8, 14),
        (11, 14),
    ]
    gt_assemblies = []
    for x, y in gt_idx:
        gt_assemblies.append([(34 * pin + x * pin, y * pin), pin, pin])
        gt_assemblies.append([(17 * pin + x * pin, 17 * pin + y * pin), pin, pin])
        gt_assemblies.append([(17 * pin + x * pin, y * pin), pin, pin])
        gt_assemblies.append([(34 * pin + x * pin, 17 * pin + y * pin), pin, pin])
    medium_map = spatial2d(medium_map, 5, gt_assemblies, edges_x, edges_y)

    # Fission chambers (material 6)
    fc_assemblies = [
        [(34 * pin + 8 * pin, 8 * pin), pin, pin],
        [(17 * pin + 8 * pin, 8 * pin), pin, pin],
        [(34 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
        [(17 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
    ]
    medium_map = spatial2d(medium_map, 6, fc_assemblies, edges_x, edges_y)

    return medium_map


def resize_weight_matrix(weight_matrix, new_cells_x, new_cells_y):
    """Expand or shrink a weight matrix (or medium map) to a new grid size.

    Parameters
    ----------
    weight_matrix : ndarray
        Float array of shape (cells_x, cells_y, materials), or integer array
        of shape (cells_x, cells_y).
    new_cells_x : int
        Target number of cells in x.
    new_cells_y : int
        Target number of cells in y.

    Returns
    -------
    resized : ndarray
        Array with shape (new_cells_x, new_cells_y, materials) for float inputs,
        or (new_cells_x, new_cells_y) for integer medium maps. Float outputs are
        renormalized so weights sum to 1 along axis 2.
    """
    zoom_x = new_cells_x / weight_matrix.shape[0]
    zoom_y = new_cells_y / weight_matrix.shape[1]

    if weight_matrix.ndim == 2 and np.issubdtype(weight_matrix.dtype, np.integer):
        # Integer medium map: nearest-neighbour zoom
        resized = _ndimage_zoom(weight_matrix.astype(float), (zoom_x, zoom_y), order=0)
        return resized.astype(np.int32)

    # Float weight matrix: per-material bilinear zoom then renormalize
    n_materials = weight_matrix.shape[2]
    result = np.zeros((new_cells_x, new_cells_y, n_materials))
    for m in range(n_materials):
        result[..., m] = _ndimage_zoom(
            weight_matrix[..., m], (zoom_x, zoom_y), order=1, prefilter=False
        )
    # Clip negatives introduced by bilinear interpolation at boundaries
    result = np.clip(result, 0.0, None)
    sums = result.sum(axis=2, keepdims=True)
    sums[sums == 0.0] = 1.0
    return result / sums


def polygon_mesh(vertices, cells, side_normal=None):
    """Build a polygon mesh from vertex coordinates and cell vertex lists.

    Parameters
    ----------
    vertices : array_like, shape (V, 2)
        Vertex coordinates [cm].
    cells : list of list of int
        Vertex indices of each convex cell. Clockwise cells are reordered
        to counter-clockwise.
    side_normal : array_like, shape (S, 2), optional
        Outward unit normal of each boundary side. Boundary faces are
        assigned to the side with a matching normal. Default is the unique
        boundary face normals, ordered by angle.

    Returns
    -------
    mesh : PolygonMesh
        Mesh connectivity and geometry.
    """
    vertices = np.asarray(vertices, dtype=float)
    cells_n = len(cells)
    max_faces = max(len(cell) for cell in cells)

    cell_vertices = -np.ones((cells_n, max_faces), dtype=np.int32)
    cell_faces = -np.ones((cells_n, max_faces), dtype=np.int32)
    cell_area = np.zeros((cells_n,))
    cell_centroid = np.zeros((cells_n, 2))

    face_index = {}
    face_cells = []
    face_vertices = []

    for cc, cell in enumerate(cells):
        cell = list(cell)
        points = vertices[cell]
        # Shoelace formula
        cross = (
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
        area = 0.5 * np.sum(cross)
        # Make counter-clockwise
        if area < 0.0:
            cell = cell[::-1]
            points = points[::-1]
            cross = -cross[::-1]
            area = -area
        assert area > 0.0, f"Cell {cc} has zero area"
        _check_convex(points, cc)
        cell_area[cc] = area
        cell_centroid[cc] = np.sum(
            (points + np.roll(points, -1, axis=0)) * cross[:, None], axis=0
        ) / (6.0 * area)
        cell_vertices[cc, : len(cell)] = cell
        # Connect faces
        for kk in range(len(cell)):
            v1 = cell[kk]
            v2 = cell[(kk + 1) % len(cell)]
            key = (min(v1, v2), max(v1, v2))
            if key in face_index:
                ff = face_index[key]
                assert face_cells[ff][1] == -1, f"Face {key} has more than two cells"
                face_cells[ff][1] = cc
            else:
                ff = len(face_cells)
                face_index[key] = ff
                face_cells.append([cc, -1])
                face_vertices.append((v1, v2))
            cell_faces[cc, kk] = ff

    face_cells = np.array(face_cells, dtype=np.int32)
    face_vertices = np.array(face_vertices)

    # Outward normal of the owner cell (counter-clockwise edges)
    edge = vertices[face_vertices[:, 1]] - vertices[face_vertices[:, 0]]
    face_length = np.sqrt(np.sum(edge**2, axis=1))
    face_normal = np.stack((edge[:, 1], -edge[:, 0]), axis=1) / face_length[:, None]
    face_center = 0.5 * (vertices[face_vertices[:, 0]] + vertices[face_vertices[:, 1]])

    # Assign boundary faces to sides
    boundary = np.argwhere(face_cells[:, 1] == -1).flatten()
    if side_normal is None:
        side_normal = np.unique(np.round(face_normal[boundary], 8), axis=0)
        theta = np.arctan2(side_normal[:, 1], side_normal[:, 0])
        side_normal = side_normal[np.argsort(theta)]
    side_normal = np.asarray(side_normal, dtype=float)
    side_normal = side_normal / np.sqrt(np.sum(side_normal**2, axis=1))[:, None]

    face_side = -np.ones((face_cells.shape[0],), dtype=np.int32)
    for ff in boundary:
        match = np.argwhere(side_normal @ face_normal[ff] > 1.0 - 1e-8).flatten()
        if len(match) == 0:
            raise ValueError(
                f"Boundary face {ff} at {face_center[ff]} does not match a side"
            )
        face_side[ff] = match[0]

    return PolygonMesh(
        vertices=vertices,
        cell_vertices=cell_vertices,
        cell_faces=cell_faces,
        face_cells=face_cells,
        face_normal=face_normal,
        face_length=face_length,
        face_center=face_center,
        cell_area=cell_area,
        cell_centroid=cell_centroid,
        face_side=face_side,
        side_normal=side_normal,
        boundary_faces=boundary.astype(np.int32),
    )


def _check_convex(points, cell):
    # Cross product of consecutive edges (counter-clockwise)
    edge = np.roll(points, -1, axis=0) - points
    turn = edge[:, 0] * np.roll(edge[:, 1], -1) - edge[:, 1] * np.roll(edge[:, 0], -1)
    scale = np.max(np.sum(edge**2, axis=1))
    if np.any(turn < -1e-12 * scale):
        raise ValueError(f"Cell {cell} is not convex")


def rectangle_mesh(edges_x, edges_y):
    """Polygon mesh of rectangular cells.

    Cell ``ii * cells_y + jj`` is the rectangle ``(ii, jj)`` so the flux
    can be reshaped to ``(cells_x, cells_y, ...)``. Sides are ordered
    [x(0), x(X), y(0), y(Y)] to match ``bc_x + bc_y``.

    Parameters
    ----------
    edges_x : ndarray, shape (cells_x + 1,)
        Cell edge locations in x [cm].
    edges_y : ndarray, shape (cells_y + 1,)
        Cell edge locations in y [cm].

    Returns
    -------
    mesh : PolygonMesh
        Mesh with ``cells_x * cells_y`` quadrilateral cells.
    """
    vertices = _grid_vertices(edges_x, edges_y)
    cells_x = edges_x.size - 1
    cells_y = edges_y.size - 1
    cells = []
    for ii in range(cells_x):
        for jj in range(cells_y):
            v00, v10, v11, v01 = _grid_corners(ii, jj, cells_y)
            cells.append([v00, v10, v11, v01])
    return polygon_mesh(vertices, cells, side_normal=_RECTANGLE_SIDES)


def triangle_mesh(edges_x, edges_y, pattern="banded"):
    """Polygon mesh of rectangles split into triangles.

    Parameters
    ----------
    edges_x : ndarray, shape (cells_x + 1,)
        Rectangle edge locations in x [cm].
    edges_y : ndarray, shape (cells_y + 1,)
        Rectangle edge locations in y [cm].
    pattern : str, optional
        "banded" (default) splits every rectangle along the same diagonal,
        "union_jack" alternates the diagonal, and "crossed" splits each
        rectangle into four triangles about its center.

    Returns
    -------
    mesh : PolygonMesh
        Triangular mesh with sides ordered [x(0), x(X), y(0), y(Y)].
    """
    vertices = _grid_vertices(edges_x, edges_y)
    cells_x = edges_x.size - 1
    cells_y = edges_y.size - 1
    cells = []
    for ii in range(cells_x):
        for jj in range(cells_y):
            v00, v10, v11, v01 = _grid_corners(ii, jj, cells_y)
            if pattern == "banded" or (pattern == "union_jack" and (ii + jj) % 2 == 0):
                cells.append([v00, v10, v11])
                cells.append([v00, v11, v01])
            elif pattern == "union_jack":
                cells.append([v00, v10, v01])
                cells.append([v10, v11, v01])
            elif pattern == "crossed":
                vc = vertices.shape[0]
                center = [
                    0.5 * (edges_x[ii] + edges_x[ii + 1]),
                    0.5 * (edges_y[jj] + edges_y[jj + 1]),
                ]
                vertices = np.vstack((vertices, center))
                cells.append([v00, v10, vc])
                cells.append([v10, v11, vc])
                cells.append([v11, v01, vc])
                cells.append([v01, v00, vc])
            else:
                raise ValueError(f"Unknown triangle pattern {pattern}")
    return polygon_mesh(vertices, cells, side_normal=_RECTANGLE_SIDES)


def triangle_mesh_delaunay(points, side_normal=None):
    """Delaunay triangulation of a point set.

    Parameters
    ----------
    points : array_like, shape (V, 2)
        Vertex coordinates [cm]. The convex hull is the domain boundary.
    side_normal : array_like, shape (S, 2), optional
        Outward unit normal of each boundary side, see :func:`polygon_mesh`.

    Returns
    -------
    mesh : PolygonMesh
        Triangular mesh.
    """
    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=float)
    triangulation = Delaunay(points)
    return polygon_mesh(points, triangulation.simplices, side_normal=side_normal)


def hexagon_boundary(center, pitch, orientation="flat"):
    """Vertices of a hexagonal domain.

    Parameters
    ----------
    center : tuple of float
        Center of the hexagon [cm].
    pitch : float
        Flat-to-flat width [cm].
    orientation : str, optional
        "flat" (default, flat top) or "pointy".

    Returns
    -------
    boundary : ndarray, shape (6, 2)
        Counter-clockwise vertices.
    """
    radius = pitch / np.sqrt(3)
    offset = 0.0 if orientation == "flat" else np.pi / 6
    theta = offset + np.pi / 3 * np.arange(6)
    return np.stack(
        (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)),
        axis=1,
    )


def hexagon_mesh(pitch, boundary, orientation="flat", origin=(0.0, 0.0), clip=True):
    """Hexagonal lattice clipped to a convex boundary.

    Hexagons crossing the boundary are clipped to convex cut cells. The
    boundary edges become the mesh sides, in the order given. With
    ``clip=False`` only whole hexagons inside the boundary are kept, which
    gives a jagged boundary with one side per hexagon face direction.

    Parameters
    ----------
    pitch : float
        Flat-to-flat width of each hexagon [cm].
    boundary : array_like, shape (S, 2)
        Counter-clockwise vertices of the convex domain [cm].
    orientation : str, optional
        "flat" (default, flat top) or "pointy" hexagons.
    origin : tuple of float, optional
        Center of one lattice hexagon [cm].
    clip : bool, optional
        If True (default), clip hexagons to the boundary. Otherwise keep
        the hexagons with every vertex inside the boundary.

    Returns
    -------
    mesh : PolygonMesh
        Hexagonal mesh.
    """
    boundary = np.asarray(boundary, dtype=float)
    radius = pitch / np.sqrt(3)
    hexagon = hexagon_boundary((0.0, 0.0), pitch, orientation)

    # Lattice centers covering the boundary
    if orientation == "flat":
        step_q, step_r = 1.5 * radius, pitch
    else:
        step_q, step_r = pitch, 1.5 * radius
    low = np.min(boundary, axis=0) - np.array(origin) - 2 * radius
    high = np.max(boundary, axis=0) - np.array(origin) + 2 * radius

    vertex_index = {}
    vertices = []
    cells = []

    if orientation == "flat":
        range_q = range(
            int(np.floor(low[0] / step_q)), int(np.ceil(high[0] / step_q)) + 1
        )
        range_r = range(
            int(np.floor(low[1] / step_r)) - 1, int(np.ceil(high[1] / step_r)) + 1
        )
    else:
        range_q = range(
            int(np.floor(low[1] / step_r)), int(np.ceil(high[1] / step_r)) + 1
        )
        range_r = range(
            int(np.floor(low[0] / step_q)) - 1, int(np.ceil(high[0] / step_q)) + 1
        )

    for qq in range_q:
        for rr in range_r:
            if orientation == "flat":
                center = (qq * step_q, (rr + 0.5 * (qq % 2)) * step_r)
            else:
                center = ((rr + 0.5 * (qq % 2)) * step_q, qq * step_r)
            points = hexagon + np.array(center) + np.array(origin)
            clipped = _clip_polygon(points, boundary)
            # Keep only whole hexagons without clipping
            if not clip and not np.isclose(
                _polygon_area(clipped) if len(clipped) > 2 else 0.0,
                _polygon_area(points),
            ):
                continue
            points = clipped
            if len(points) < 3 or _polygon_area(points) < 1e-10 * pitch**2:
                continue
            cell = []
            for point in np.round(points, _VERTEX_PRECISION):
                key = (point[0], point[1])
                if key not in vertex_index:
                    vertex_index[key] = len(vertices)
                    vertices.append(point)
                if vertex_index[key] not in cell:
                    cell.append(vertex_index[key])
            cells.append(cell)

    if not clip:
        return polygon_mesh(np.array(vertices), cells)
    edge = np.roll(boundary, -1, axis=0) - boundary
    side_normal = np.stack((edge[:, 1], -edge[:, 0]), axis=1)
    return polygon_mesh(np.array(vertices), cells, side_normal=side_normal)


def _clip_polygon(points, boundary):
    # Sutherland-Hodgman clipping against a convex counter-clockwise boundary
    output = list(points)
    for kk in range(boundary.shape[0]):
        a = boundary[kk]
        b = boundary[(kk + 1) % boundary.shape[0]]
        normal = np.array([b[1] - a[1], a[0] - b[0]])
        inside = [np.dot(normal, p - a) <= 1e-12 for p in output]
        clipped = []
        for ii in range(len(output)):
            p1 = output[ii]
            p2 = output[(ii + 1) % len(output)]
            if inside[ii]:
                clipped.append(p1)
            if inside[ii] != inside[(ii + 1) % len(output)]:
                d1 = np.dot(normal, p1 - a)
                d2 = np.dot(normal, p2 - a)
                clipped.append(p1 + d1 / (d1 - d2) * (p2 - p1))
        output = clipped
        if len(output) == 0:
            break
    return np.array(output)


def _polygon_area(points):
    return 0.5 * np.fabs(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1))
        - np.sum(np.roll(points[:, 0], -1) * points[:, 1])
    )


def _grid_vertices(edges_x, edges_y):
    xx, yy = np.meshgrid(edges_x, edges_y, indexing="ij")
    return np.stack((xx.flatten(), yy.flatten()), axis=1)


def _grid_corners(ii, jj, cells_y):
    v00 = ii * (cells_y + 1) + jj
    v10 = (ii + 1) * (cells_y + 1) + jj
    return v00, v10, v10 + 1, v00 + 1


# Sides of rectangular domains [x(0), x(X), y(0), y(Y)]
_RECTANGLE_SIDES = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


def polygon_medium_map(mesh, default=0, **kwargs):
    """Material map for a polygon mesh evaluated at cell centroids.

    Parameters
    ----------
    mesh : PolygonMesh
        Polygon mesh.
    default : int, optional
        Material of cells outside every shape. Default is 0.

    Keyword Arguments
    -----------------
    rectangles, triangles, circles, hexagons : list
        Shapes as in :func:`ants.main.weight_matrix2d`; hexagons are
        ``[(x1, y1), pitch]`` or ``[(x1, y1), pitch, orientation]``.
    rectangle_index, triangle_index, circle_index, hexagon_index : list of int
        Material of each shape.

    Returns
    -------
    medium_map : ndarray of int32, shape (cells, 1)
        Material index for each cell.
    """
    medium_map = np.full((mesh.cells, 1), default, dtype=np.int32)
    materials = 1 + max(
        max(kwargs[key]) for key in kwargs.keys() if key.endswith("_index")
    )
    tally = np.zeros((materials))
    for cc, (x, y) in enumerate(mesh.cell_centroid):
        tally *= 0.0
        _inside_shapes(x, y, tally, kwargs)
        if np.sum(tally) > 0.0:
            medium_map[cc, 0] = np.argmax(tally)
    return medium_map


def polygon_weight_matrix(mesh, materials, N_particles=None, **kwargs):
    """Material weight fractions for a polygon mesh using MC sampling.

    Points outside every shape are tallied to the last material. The
    result can be passed to :func:`ants.main.weight_spatial2d`.

    Parameters
    ----------
    mesh : PolygonMesh
        Polygon mesh.
    materials : int
        Number of materials.
    N_particles : int, optional
        Samples per cell. Default is 1_000.

    Keyword Arguments
    -----------------
    See :func:`polygon_medium_map`.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells, 1, materials)
        Material weight fractions.
    """
    if N_particles is None:
        N_particles = 1_000
    weight_matrix = np.zeros((mesh.cells, 1, materials))
    tally = np.zeros((materials))
    for cc in range(mesh.cells):
        cell = mesh.cell_vertices[cc][mesh.cell_vertices[cc] >= 0]
        for x, y in _sample_polygon(mesh.vertices[cell], N_particles):
            tally *= 0.0
            _inside_shapes(x, y, tally, kwargs)
            tally[tally > 0.0] = 1.0
            if np.sum(tally) == 0:
                tally[-1] = 1
            weight_matrix[cc, 0] += tally
    return weight_matrix / np.sum(weight_matrix, axis=2)[..., None]


def _inside_shapes(x, y, tally, kwargs):
    if "triangles" in kwargs.keys() and np.sum(tally) == 0.0:
        _inside_triangle(kwargs["triangles"], x, y, tally, kwargs["triangle_index"])
    if "rectangles" in kwargs.keys() and np.sum(tally) == 0.0:
        _inside_rectangle(kwargs["rectangles"], x, y, tally, kwargs["rectangle_index"])
    if "circles" in kwargs.keys() and np.sum(tally) == 0.0:
        _inside_circle(kwargs["circles"], x, y, tally, kwargs["circle_index"])
    if "hexagons" in kwargs.keys() and np.sum(tally) == 0.0:
        _inside_hexagon(kwargs["hexagons"], x, y, tally, kwargs["hexagon_index"])


def _sample_polygon(points, N_particles):
    # Uniform samples from a triangle fan about the first vertex
    v0 = points[0]
    edge1 = points[1:-1] - v0
    edge2 = points[2:] - v0
    area = 0.5 * np.fabs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    fan = np.random.choice(area.size, size=N_particles, p=area / np.sum(area))
    r1 = np.sqrt(np.random.uniform(size=N_particles))
    r2 = np.random.uniform(size=N_particles)
    return v0 + (r1 * (1 - r2))[:, None] * edge1[fan] + (r1 * r2)[:, None] * edge2[fan]


def _plot_polygon(mesh, values):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polygons = [mesh.vertices[cell[cell >= 0]] for cell in mesh.cell_vertices]
    fig, ax = plt.subplots(figsize=(6, 6))
    collection = PolyCollection(
        polygons, array=np.asarray(values).flatten(), edgecolors="k", linewidths=0.2
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.colorbar(collection, ax=ax)
    plt.tight_layout()
    plt.show()


def _plot_result(arr):
    import matplotlib.pyplot as plt

    if arr.ndim == 3:
        n_mat = arr.shape[2]
        fig, axes = plt.subplots(1, n_mat, figsize=(4 * n_mat, 4))
        if n_mat == 1:
            axes = [axes]
        for m, ax in enumerate(axes):
            im = ax.imshow(arr[:, :, m].T, origin="lower", vmin=0, vmax=1)
            ax.set_title(f"Material {m}")
            plt.colorbar(im, ax=ax)
    else:
        n_mat = int(arr.max()) + 1
        cmap = plt.get_cmap("tab10", n_mat)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=n_mat - 0.5)
        cbar = plt.colorbar(im, ax=ax, ticks=range(n_mat))
        cbar.set_label("Material index")
        ax.set_title("Medium map")

    plt.tight_layout()
    plt.show()


_LAYOUTS = ("cylinder-1mat", "cylinder-2mat", "double-chevron", "c5g7")

_DISPATCH = {
    "cylinder-1mat": cylinder_1mat,
    "cylinder-2mat": cylinder_2mat,
    "double-chevron": double_chevron,
    "c5g7": c5g7,
}

_LAYOUT_KWARGS = {
    "cylinder-1mat": ("radius", "length_x", "length_y", "n_particles"),
    "cylinder-2mat": ("r_inner", "r_outer", "length_x", "length_y", "n_particles"),
    "double-chevron": ("n_particles"),
    "c5g7": (),
}


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ants.utils.mesh2d",
        description="Generate or resize 2D neutron transport geometry layouts.",
    )
    parser.add_argument(
        "layout",
        nargs="?",
        choices=_LAYOUTS,
        help="Geometry layout to generate",
    )
    parser.add_argument(
        "--resize",
        metavar="INPUT.NPY",
        help=(
            "Path to existing weight matrix .npy file to resize"
            " (mutually exclusive with layout)"
        ),
    )
    parser.add_argument(
        "--cells-x", type=int, required=True, help="Number of cells in x"
    )
    parser.add_argument(
        "--cells-y", type=int, required=True, help="Number of cells in y"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show matplotlib plot of result"
    )
    parser.add_argument("--save", metavar="OUTPUT.NPY", help="Save result to .npy file")
    # Layout-specific overrides
    parser.add_argument(
        "--radius", type=float, help="[cylinder_1mat] cylinder radius [cm]"
    )
    parser.add_argument(
        "--r-inner",
        type=float,
        dest="r_inner",
        help="[cylinder-2mat] inner radius [cm]",
    )
    parser.add_argument(
        "--r-outer",
        type=float,
        dest="r_outer",
        help="[cylinder-2mat] outer radius [cm]",
    )
    parser.add_argument(
        "--length-x", type=float, dest="length_x", help="Domain length in x [cm]"
    )
    parser.add_argument(
        "--length-y", type=float, dest="length_y", help="Domain length in y [cm]"
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        dest="n_particles",
        help="[MC layouts] number of MC samples",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.layout is None and args.resize is None:
        parser.error("specify a layout name or --resize INPUT.NPY")
    if args.layout is not None and args.resize is not None:
        parser.error("layout and --resize are mutually exclusive")

    if args.resize is not None:
        data = np.load(args.resize)
        result = resize_weight_matrix(data, args.cells_x, args.cells_y)
    else:
        fn = _DISPATCH[args.layout]
        allowed = _LAYOUT_KWARGS[args.layout]
        kwargs = {k: v for k in allowed if (v := getattr(args, k, None)) is not None}
        result = fn(args.cells_x, args.cells_y, **kwargs)

    print(f"Result shape: {result.shape}")

    if args.save:
        np.save(args.save, result)
        print(f"Saved to {args.save}")

    if args.plot:
        _plot_result(result)
