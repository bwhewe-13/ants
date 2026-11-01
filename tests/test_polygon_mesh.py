########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for two-dimensional polygon meshes (triangles, hexagons).
# Includes mesh connectivity, sweep ordering, and curve fits.
#
########################################################################

import numpy as np
import pytest

import ants
from ants.utils import mesh2d
from ants.utils.interp_polygon import PolygonInterpolation, polygon_quadrature
from ants.utils.pytools import polygon_opposite_faces, polygon_sweep_setup

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)]


def _meshes():
    edges_x = np.linspace(0, 1, 6)
    edges_y = np.linspace(0, 2, 5)
    hexagon = mesh2d.hexagon_boundary((0.0, 0.0), 2.0, "pointy")
    return {
        "rectangle": (mesh2d.rectangle_mesh(edges_x, edges_y), 2.0),
        "banded": (mesh2d.triangle_mesh(edges_x, edges_y), 2.0),
        "union_jack": (mesh2d.triangle_mesh(edges_x, edges_y, "union_jack"), 2.0),
        "crossed": (mesh2d.triangle_mesh(edges_x, edges_y, "crossed"), 2.0),
        "hex_flat": (mesh2d.hexagon_mesh(0.3, SQUARE), 2.0),
        "hex_pointy": (mesh2d.hexagon_mesh(0.3, SQUARE, orientation="pointy"), 2.0),
        "hex_domain": (mesh2d.hexagon_mesh(0.4, hexagon), 2.0 * np.sqrt(3)),
    }


MESHES = _meshes()


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.parametrize("name", MESHES.keys())
def test_mesh_geometry(name):
    mesh, area = MESHES[name]
    assert np.isclose(np.sum(mesh.cell_area), area)
    for cc in range(mesh.cells):
        faces = mesh.cell_faces[cc][mesh.cell_faces[cc] >= 0]
        sign = np.where(mesh.face_cells[faces, 0] == cc, 1.0, -1.0)
        normal = sign[:, None] * mesh.face_normal[faces]
        # Closed cell
        closed = np.sum(normal * mesh.face_length[faces, None], axis=0)
        assert np.allclose(closed, 0.0)
        # Outward normals
        outward = mesh.face_center[faces] - mesh.cell_centroid[cc]
        assert np.all(np.sum(normal * outward, axis=1) > 0.0)
    # Boundary faces are on a side
    boundary = mesh.face_cells[:, 1] == -1
    assert np.all(mesh.face_side[boundary] >= 0)
    assert np.all(mesh.face_side[~boundary] == -1)


@pytest.mark.polygon2d
def test_side_order():
    mesh, _ = MESHES["banded"]
    sides = mesh.face_side[mesh.boundary_faces]
    centers = mesh.face_center[mesh.boundary_faces]
    # [x(0), x(X), y(0), y(Y)]
    assert np.allclose(centers[sides == 0, 0], 0.0)
    assert np.allclose(centers[sides == 1, 0], 1.0)
    assert np.allclose(centers[sides == 2, 1], 0.0)
    assert np.allclose(centers[sides == 3, 1], 2.0)
    mesh, _ = MESHES["hex_domain"]
    assert mesh.sides == 6
    assert np.all(np.bincount(mesh.face_side[mesh.boundary_faces]) > 0)


@pytest.mark.polygon2d
def test_not_convex():
    vertices = [(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)]
    with pytest.raises(ValueError):
        mesh2d.polygon_mesh(vertices, [[0, 1, 2, 3, 4]])


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.parametrize("name", ["union_jack", "crossed", "hex_domain"])
def test_sweep_order(name):
    mesh, _ = MESHES[name]
    quadrature = ants.angular_xy(6)
    order, _ = polygon_sweep_setup(
        mesh, quadrature.angle_x, quadrature.angle_y, [0] * mesh.sides
    )
    interior = mesh.face_cells[:, 1] >= 0
    for nn in range(quadrature.angle_x.size):
        position = np.empty((mesh.cells,), dtype=int)
        position[order[nn]] = np.arange(mesh.cells)
        dot = (
            quadrature.angle_x[nn] * mesh.face_normal[:, 0]
            + quadrature.angle_y[nn] * mesh.face_normal[:, 1]
        )
        upwind = np.where(dot > 0, mesh.face_cells[:, 0], mesh.face_cells[:, 1])
        downwind = np.where(dot > 0, mesh.face_cells[:, 1], mesh.face_cells[:, 0])
        check = interior & (dot != 0.0)
        assert np.all(position[upwind[check]] < position[downwind[check]])


@pytest.mark.polygon2d
def test_reflected_angles():
    mesh, _ = MESHES["hex_domain"]
    # Six angles are closed under reflection about the hexagon sides
    quadrature = ants.angular_xy(6)
    _, reflect = polygon_sweep_setup(
        mesh, quadrature.angle_x, quadrature.angle_y, [1] * mesh.sides
    )
    assert np.sum(reflect >= 0) == mesh.sides * quadrature.angle_x.size // 2
    # Four angles are not
    quadrature = ants.angular_xy(4)
    with pytest.raises(ValueError):
        polygon_sweep_setup(
            mesh, quadrature.angle_x, quadrature.angle_y, [1] * mesh.sides
        )


@pytest.mark.polygon2d
def test_medium_map():
    mesh, _ = MESHES["hex_domain"]
    medium_map = mesh2d.polygon_medium_map(
        mesh, default=1, hexagons=[[(0.0, 0.0), 1.0]], hexagon_index=[0]
    )
    assert medium_map.shape == (mesh.cells, 1)
    inside = np.zeros((2,))
    for cc, (x, y) in enumerate(mesh.cell_centroid):
        inside *= 0.0
        ants.main._inside_hexagon([[(0.0, 0.0), 1.0]], x, y, inside, [0])
        assert medium_map[cc, 0] == (0 if inside[0] else 1)
    weight_matrix = mesh2d.polygon_weight_matrix(
        mesh, 2, N_particles=200, hexagons=[[(0.0, 0.0), 1.0]], hexagon_index=[0]
    )
    assert weight_matrix.shape == (mesh.cells, 1, 2)
    assert np.allclose(np.sum(weight_matrix, axis=2), 1.0)


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.math
@pytest.mark.parametrize("name", ["crossed", "hex_domain"])
def test_curve_fit_quintic(name):
    mesh, _ = MESHES[name]

    def func(x, y):
        return 1 + x**3 * y - 2 * y**5 + x * y**2

    def func_dx(x, y):
        return 3 * x**2 * y + y**2

    points, weights = polygon_quadrature(mesh, 5)
    assert np.allclose(np.sum(weights, axis=1), mesh.cell_area)
    average = np.sum(func(*points.T).T * weights, axis=1) / mesh.cell_area
    medium_map = np.zeros((mesh.cells, 1), dtype=np.int32)
    approx = PolygonInterpolation(average, mesh, medium_map, degree=5)

    int_psi, int_dx, _ = approx.integrate_centers()
    assert np.allclose(int_psi, average * mesh.cell_area)
    assert np.allclose(int_dx, np.sum(func_dx(*points.T).T * weights, axis=1))
    assert np.allclose(approx.interpolate_centers(), func(*mesh.cell_centroid.T))
    faces = mesh.boundary_faces
    assert np.allclose(
        approx.interpolate_faces(faces), func(*mesh.face_center[faces].T)
    )


@pytest.mark.polygon2d
def test_opposite_faces():
    # Rectangles and whole hexagons are paired, triangles are not
    for name, paired in [("rectangle", True), ("crossed", False)]:
        mesh, _ = MESHES[name]
        opposite, _ = polygon_opposite_faces(mesh)
        assert np.all((opposite[:, 0] >= 0) == paired)
    mesh = mesh2d.hexagon_mesh(0.02, SQUARE, clip=False)
    opposite, width = polygon_opposite_faces(mesh)
    assert np.all(opposite[:, 0] >= 0)
    assert np.allclose(width, 0.02)
    # Rectangle widths are the cell widths
    mesh, _ = MESHES["rectangle"]
    _, width = polygon_opposite_faces(mesh)
    assert np.allclose(np.sort(np.unique(np.round(width, 12))), [0.2, 0.5])
