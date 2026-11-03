########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test two-dimensional fixed source and criticality problems on polygon
# meshes (triangles, hexagons). Rectangular polygon cells are checked
# against the SLAB2D step solution.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import critical2d, fixed2d
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    MultigroupSolver,
    ParallelType,
    SolverData,
    SourceData,
    SpatialDiscretization,
)
from ants.utils import manufactured_2d as mms
from ants.utils import mesh2d
from tests import problems2d


def _slab_problem():
    # Two materials, two groups, mixed vacuum and reflected boundaries
    cells_x = 8
    cells_y = 6
    angles = 4
    groups = 2
    edges_x = np.linspace(0, 2.0, cells_x + 1)
    edges_y = np.linspace(0, 1.5, cells_y + 1)
    bc_x = [0, 1]
    bc_y = [1, 0]
    quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)
    mat_data = MaterialData(
        total=np.array([[1.0, 2.0], [0.5, 1.5]]),
        scatter=np.array([[[0.3, 0.0], [0.2, 0.8]], [[0.1, 0.0], [0.1, 0.4]]]),
        fission=np.zeros((2, 2, 2)),
    )
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    medium_map[cells_x // 2 :] = 1

    rng = np.random.default_rng(1)
    boundary_x = np.zeros((2, cells_y, angles**2, groups))
    boundary_x[0] = rng.random((cells_y, angles**2, groups))
    boundary_y = np.zeros((2, cells_x, angles**2, groups))
    boundary_y[1] = rng.random((cells_x, angles**2, groups))
    sources = SourceData(
        external=rng.random((cells_x, cells_y, 1, groups)),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.diff(edges_x),
        delta_y=np.diff(edges_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=Geometry.SLAB2D,
        space_disc=SpatialDiscretization.STEP,
    )
    return mat_data, sources, geometry, quadrature


SOLVERS = [
    SolverData(),
    SolverData(angular=True),
    SolverData(num_threads=4),
    SolverData(parallel=ParallelType.GROUP, num_threads=2),
    SolverData(sigma_as=0.5),
    SolverData(mg_solver=MultigroupSolver.DMD),
]

SPATIAL = [
    SpatialDiscretization.STEP,
    SpatialDiscretization.DIAMOND,
    SpatialDiscretization.STEP_CHARACTERISTIC,
]


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.source_iteration
@pytest.mark.parametrize("spatial", SPATIAL)
@pytest.mark.parametrize("solver", SOLVERS)
def test_rectangle_cells(solver, spatial):
    mat_data, sources, geometry, quadrature = _slab_problem()
    geometry.space_disc = spatial
    reference = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)

    sources, geometry = problems2d.slab_to_polygon(sources, geometry)
    geometry.space_disc = spatial
    flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)
    assert np.allclose(flux.reshape(reference.shape), reference, atol=1e-12)


@pytest.mark.polygon2d
@pytest.mark.source_iteration
@pytest.mark.parametrize("spatial", SPATIAL)
def test_rectangle_cell_edges(spatial):
    mat_data, sources, geometry, quadrature = _slab_problem()
    geometry.space_disc = spatial
    solver = SolverData(flux_at_edges=1)
    flux_x, flux_y = fixed2d.fixed_source(
        mat_data, sources, geometry, quadrature, solver
    )
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    edges_y = np.concatenate(([0.0], np.cumsum(geometry.delta_y)))

    sources, geometry = problems2d.slab_to_polygon(sources, geometry)
    geometry.space_disc = spatial
    flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)
    mesh = geometry.mesh
    assert flux.shape == (mesh.faces, 2)
    for ff, (x, y) in enumerate(mesh.face_center):
        if np.fabs(mesh.face_normal[ff, 0]) > 0.5:
            ii = np.argmin(np.fabs(edges_x - x))
            jj = np.argmin(np.fabs(0.5 * (edges_y[1:] + edges_y[:-1]) - y))
            assert np.allclose(flux[ff], flux_x[ii, jj], atol=1e-12)
        else:
            ii = np.argmin(np.fabs(0.5 * (edges_x[1:] + edges_x[:-1]) - x))
            jj = np.argmin(np.fabs(edges_y - y))
            assert np.allclose(flux[ff], flux_y[ii, jj], atol=1e-12)


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.power_iteration
@pytest.mark.parametrize("spatial", SPATIAL)
def test_rectangle_cells_criticality(spatial):
    mat_data, _, geometry, quadrature = _slab_problem()
    geometry.space_disc = spatial
    mat_data.fission = np.zeros((2, 2, 2))
    mat_data.fission[0] = [[0.5, 1.2], [0.0, 0.0]]
    reference, keff_reference = critical2d.k_criticality(
        mat_data, geometry, quadrature, SolverData()
    )
    cells = reference.shape[0] * reference.shape[1]
    geometry = GeometryData(
        medium_map=geometry.medium_map.reshape(cells, 1),
        geometry=Geometry.POLYGON2D,
        space_disc=spatial,
        mesh=mesh2d.rectangle_mesh(
            np.concatenate(([0.0], np.cumsum(geometry.delta_x))),
            np.concatenate(([0.0], np.cumsum(geometry.delta_y))),
        ),
        bc_sides=geometry.bc_x + geometry.bc_y,
    )
    flux, keff = critical2d.k_criticality(mat_data, geometry, quadrature, SolverData())
    assert np.isclose(keff, keff_reference, atol=1e-12)
    assert np.allclose(flux.reshape(reference.shape), reference, atol=1e-12)


def _infinite_meshes():
    hexagon = mesh2d.hexagon_boundary((0.0, 0.0), 2.0, "pointy")
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    edges = np.linspace(0, 1, 9)
    return {
        "triangles": mesh2d.triangle_mesh(edges, edges, "union_jack"),
        "hex_rectangle": mesh2d.hexagon_mesh(0.15, square),
        "hex_domain": mesh2d.hexagon_mesh(0.2, hexagon),
        "hex_domain_pointy": mesh2d.hexagon_mesh(
            0.2, mesh2d.hexagon_boundary((0.0, 0.0), 2.0), orientation="pointy"
        ),
    }


@pytest.mark.smoke
@pytest.mark.polygon2d
@pytest.mark.source_iteration
@pytest.mark.parametrize("spatial", SPATIAL)
@pytest.mark.parametrize(
    "name", ["triangles", "hex_rectangle", "hex_domain", "hex_domain_pointy"]
)
def test_infinite_medium(name, spatial):
    mesh = _infinite_meshes()[name]
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.6]]]),
        fission=np.array([[[0.0]]]),
    )
    sources = SourceData(
        external=np.ones((mesh.cells, 1, 1, 1)),
        boundary_x=np.zeros((1, 1, 1, 1)),
        boundary_y=np.zeros((1, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=spatial,
        mesh=mesh,
        bc_sides=[1] * mesh.sides,
    )
    quadrature = ants.angular_xy(6)
    solver = SolverData(max_iter_angular=500)
    flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)
    assert np.allclose(flux, 1.0 / 0.4, atol=1e-8)


@pytest.mark.polygon2d
@pytest.mark.power_iteration
def test_infinite_medium_criticality():
    mesh = _infinite_meshes()["hex_domain"]
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.6]]]),
        fission=np.array([[[0.5]]]),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=[1] * mesh.sides,
    )
    quadrature = ants.angular_xy(6)
    _, keff = critical2d.k_criticality(mat_data, geometry, quadrature, SolverData())
    assert np.isclose(keff, 0.5 / 0.4, atol=1e-6)


def _infinite_moments(scatter):
    mesh = _infinite_meshes()["hex_domain"]
    mat_data = MaterialData(
        total=np.array([[1.0]]), scatter=scatter, fission=np.array([[[0.0]]])
    )
    sources = SourceData(
        external=np.ones((mesh.cells, 1, 1, 1)),
        boundary_x=np.zeros((1, 1, 1, 1)),
        boundary_y=np.zeros((1, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((mesh.cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh,
        bc_sides=[1] * mesh.sides,
    )
    quadrature = ants.angular_xy(6)
    solver = SolverData(max_iter_angular=500)
    return fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)


@pytest.mark.polygon2d
@pytest.mark.anisotropic
def test_infinite_medium_single_moment():
    # Legendre moment layout with only P0 runs the isotropic polygon sweep
    flux = _infinite_moments(np.array([[[[0.6]]]]))
    assert np.allclose(flux, 1.0 / 0.4, atol=1e-8)


@pytest.mark.polygon2d
@pytest.mark.anisotropic
def test_anisotropic_not_supported():
    with pytest.raises(ValueError, match="isotropic"):
        _infinite_moments(np.array([[[[0.6, 0.2]]]]))


ORDERS = [
    # Step differencing is first order
    (SpatialDiscretization.STEP, "banded", 0.8),
    (SpatialDiscretization.STEP, "crossed", 0.8),
    (SpatialDiscretization.STEP, "hexagon", 0.8),
    # Mean incoming closure on triangles is first order
    (SpatialDiscretization.DIAMOND, "banded", 0.8),
    (SpatialDiscretization.STEP_CHARACTERISTIC, "crossed", 0.8),
    # Paired faces on whole hexagons are second order
    (SpatialDiscretization.DIAMOND, "hexagon", 1.8),
    (SpatialDiscretization.STEP_CHARACTERISTIC, "hexagon", 1.8),
]


@pytest.mark.polygon2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("spatial", "pattern", "order"), ORDERS)
def test_manufactured_01_order(spatial, pattern, order):
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    errors = []
    for cells in [20, 40]:
        if pattern == "hexagon":
            # Whole hexagons, the exact solution is used on the boundary
            mesh = mesh2d.hexagon_mesh(1.0 / cells, square, clip=False)
        else:
            edges = np.linspace(0, 1, cells + 1)
            mesh = mesh2d.triangle_mesh(edges, edges, pattern)
        mat_data, sources, geometry, quadrature, solver = (
            problems2d.manufactured_ss_01_polygon(mesh, 4)
        )
        geometry.space_disc = spatial
        solver.angular = True
        flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)
        exact = mms.solution_ss_01_points(mesh.cell_centroid[:, 0], quadrature.angle_x)
        difference = (flux[:, 0, :, 0] - exact) ** 2
        errors.append(np.sqrt(np.sum(mesh.cell_area[:, None] * difference)))
    assert np.log2(errors[0] / errors[1]) > order
