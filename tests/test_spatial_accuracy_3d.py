########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Order of Accuracy for 3D Spatial Discretization Schemes. Uses
# Method of Manufactured Solutions for testing the step, diamond
# difference (dd), and step characteristic (sc) methods.
#
########################################################################

import numpy as np
import pytest

import ants
from ants.datatypes import Geometry, GeometryData, MaterialData, SolverData, SourceData
from ants.fixed3d import fixed_source
from ants.utils import pytools as tools

PARAMETERS = [
    (angular, edges, spatial)
    for angular in [True, False]
    for edges in [0, 1]
    for spatial in [1, 2, 3]
]


def solution_ss_01(centers_x, centers_y, centers_z, quadrature):
    # One material with scattering, (cells_x, cells_y, cells_z, angles)
    x, y, z = np.meshgrid(centers_x, centers_y, centers_z, indexing="ij")
    return 1.0 + 0.1 * (
        x[..., None] ** 2 * np.exp(quadrature.angle_x)
        + y[..., None] ** 2 * np.exp(quadrature.angle_y)
        + z[..., None] ** 2 * np.exp(quadrature.angle_z)
    )


def manufactured_ss_01(cells, quadrature, xs_total=1.0, xs_scatter=0.5):
    # Unit cube with the source and boundaries from solution_ss_01
    edges = np.linspace(0, 1, cells + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    lower, upper = np.array([0.0]), np.array([1.0])

    # Omega . grad(psi) + sigma_t psi - sigma_s phi, where phi uses the
    # quadrature so only the spatial error is left
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    psi = solution_ss_01(centers, centers, centers, quadrature)
    streaming = 0.2 * (
        quadrature.angle_x * x[..., None] * np.exp(quadrature.angle_x)
        + quadrature.angle_y * y[..., None] * np.exp(quadrature.angle_y)
        + quadrature.angle_z * z[..., None] * np.exp(quadrature.angle_z)
    )
    scalar = psi @ quadrature.angle_w
    external = streaming + xs_total * psi - xs_scatter * scalar[..., None]

    # Angular flux on each face, with the lower side first
    boundary_x = [
        solution_ss_01(edge, centers, centers, quadrature)[0] for edge in (lower, upper)
    ]
    boundary_y = [
        solution_ss_01(centers, edge, centers, quadrature)[:, 0]
        for edge in (lower, upper)
    ]
    boundary_z = [
        solution_ss_01(centers, centers, edge, quadrature)[:, :, 0]
        for edge in (lower, upper)
    ]

    materials = MaterialData(
        total=np.array([[xs_total]]),
        scatter=np.array([[[xs_scatter]]]),
        fission=np.zeros((1, 1, 1)),
    )
    sources = SourceData(
        external=external[..., None],
        boundary_x=np.stack(boundary_x)[..., None],
        boundary_y=np.stack(boundary_y)[..., None],
        boundary_z=np.stack(boundary_z)[..., None],
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells, cells), dtype=np.int32),
        delta_x=np.diff(edges),
        delta_y=np.diff(edges),
        delta_z=np.diff(edges),
        geometry=Geometry.SLAB3D,
    )
    return materials, sources, geometry, psi[..., None], scalar[..., None]


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_01(angular, edges, spatial):
    cells = np.array([32, 64, 128]) if spatial == 1 else np.array([8, 16, 32])
    order = 1 if spatial == 1 else 2
    atol_c = 5e-2 if spatial == 1 else 1e-2
    atol_e = 5e-2 if spatial == 1 else 6e-2
    quadrature = ants.angular_xyz(2)
    solver = SolverData(
        angular=angular,
        flux_at_edges=edges,
        num_threads=4,
        tol_angular=1e-12,
        tol_energy=1e-10,
    )
    errors = []
    for ii in cells:
        materials, sources, geometry, exact_psi, exact_phi = manufactured_ss_01(
            ii, quadrature
        )
        geometry.space_disc = spatial
        flux = fixed_source(materials, sources, geometry, quadrature, solver)
        if edges:
            # Exact flux on the x, y, and z faces
            edges_ = np.linspace(0, 1, ii + 1)
            centers = 0.5 * (edges_[1:] + edges_[:-1])
            exact = [
                solution_ss_01(edges_, centers, centers, quadrature),
                solution_ss_01(centers, edges_, centers, quadrature),
                solution_ss_01(centers, centers, edges_, quadrature),
            ]
            if not angular:
                exact = [face @ quadrature.angle_w for face in exact]
            exact = [face[..., None] for face in exact]
            errors.append(
                [tools.spatial_error(f, e, ndims=3) for f, e in zip(flux, exact)]
            )
        else:
            exact = exact_psi if angular else exact_phi
            errors.append([tools.spatial_error(flux, exact, ndims=3)])
    atol = atol_e if edges else atol_c
    for err in range(len(errors) - 1):
        ratio = cells[err + 1] / cells[err]
        for error1, error2 in zip(errors[err], errors[err + 1]):
            assert abs(tools.order_accuracy(error1, error2, ratio) - order) < atol
