########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for two-dimensional nearby problems on polygon meshes.
#
########################################################################

import numpy as np
import pytest

import ants
from ants import nearby2d
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    SolverData,
    SpatialDiscretization,
)
from ants.utils import manufactured_2d as mms
from ants.utils import mesh2d
from tests import problems2d

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _meshes(name):
    if name == "triangles":
        edges = np.linspace(0, 1, 21)
        return mesh2d.triangle_mesh(edges, edges)
    return mesh2d.hexagon_mesh(0.05, SQUARE)


@pytest.mark.smoke
@pytest.mark.nearby2d
@pytest.mark.polygon2d
@pytest.mark.parametrize("name", ["triangles", "hexagons"])
def test_fixed_source_angular_residual(name):
    mesh = _meshes(name)
    mat_data, sources, geometry, quadrature, solver = (
        problems2d.manufactured_ss_01_polygon(mesh, 2)
    )
    solver.angular = True
    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        return_nearby=True,
        save_residual=False,
    )
    assert len(result) == 6
    numerical_flux, curve_fit_flux, boundary_x, _, residual, nearby_flux = result

    NN = 4
    assert numerical_flux.shape == (mesh.cells, 1, NN, 1)
    assert curve_fit_flux.shape == (mesh.cells, 1, 1)
    assert boundary_x.shape == (1, mesh.boundary_faces.size, NN, 1)
    assert residual.shape == (mesh.cells, 1, NN, 1)
    assert nearby_flux.shape == (mesh.cells, 1, 1)
    assert np.abs(residual).max() < 1e-3

    # Nearby error estimate is close to the true error
    exact = mms.solution_ss_01_points(mesh.cell_centroid[:, 0], quadrature.angle_x)
    exact = np.sum(exact * quadrature.angle_w, axis=1)
    numerical = np.sum(numerical_flux[:, 0, :, 0] * quadrature.angle_w, axis=1)
    true_error = np.abs(numerical - exact).max()
    nearby_error = np.abs(nearby_flux[:, 0, 0] - curve_fit_flux[:, 0, 0]).max()
    assert 0.5 < nearby_error / true_error < 2.0


@pytest.mark.nearby2d
@pytest.mark.polygon2d
def test_fixed_source_scalar_residual():
    mesh = _meshes("triangles")
    mat_data, sources, geometry, quadrature, solver = (
        problems2d.manufactured_ss_01_polygon(mesh, 2)
    )
    solver.angular = True
    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        scalar_residual=True,
        return_nearby=True,
        save_residual=False,
    )
    _, curve_fit_flux, boundary_x, _, residual, nearby_flux = result
    assert curve_fit_flux.shape == (mesh.cells, 1, 1)
    assert boundary_x.shape == (1, mesh.boundary_faces.size, 1)
    assert residual.shape == (mesh.cells, 1, 1)
    assert nearby_flux.shape == (mesh.cells, 1, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby2d
@pytest.mark.polygon2d
@pytest.mark.power_iteration
def test_criticality_rectangle_cells():
    # Polynomial curve fit tracks the Hermite spline curve fit
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.array([[[0.7]]]),
    )
    quadrature = ants.angular_xy(4)
    cells = 20
    edges = np.linspace(0, 4.0, cells + 1)
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=np.diff(edges),
        delta_y=np.diff(edges),
        geometry=Geometry.SLAB2D,
        space_disc=SpatialDiscretization.STEP,
    )
    reference = nearby2d.k_criticality(
        mat_data, geometry, quadrature, SolverData(), save_residual=False
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells * cells, 1), dtype=np.int32),
        geometry=Geometry.POLYGON2D,
        space_disc=SpatialDiscretization.STEP,
        mesh=mesh2d.rectangle_mesh(edges, edges),
    )
    result = nearby2d.k_criticality(
        mat_data, geometry, quadrature, SolverData(), save_residual=False
    )
    # Numerical, curve fit, and nearby keff
    assert np.isclose(result[1], reference[1], atol=1e-10)
    assert np.isclose(result[3], reference[3], atol=1e-3)
    assert np.isclose(result[5], reference[5], atol=1e-4)
