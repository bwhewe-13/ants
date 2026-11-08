########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Three-dimensional fixed source problems
#
########################################################################

import numpy as np
import pytest

import ants
from ants.datatypes import (
    CompactSource,
    Geometry,
    GeometryData,
    MaterialData,
    MultigroupSolver,
    ParallelType,
    SolverData,
    SourceData,
)
from ants.fixed2d import fixed_source as fixed_source_2d
from ants.fixed3d import fixed_source
from tests import problems3d as prob

KWARGS = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)
SOLVER = SolverData(**KWARGS)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_infinite_medium():
    # Reflecting on every face, the flux solves (sigma_t - S - chi nu_f) phi = Q.
    # The source iteration lags the reflected edges, so the default angle
    # order works with reflecting on both sides.
    cells = 3
    total = np.array([[1.0, 2.0]])
    scatter = np.array([[[0.3, 0.1], [0.4, 0.8]]])
    nu_fission = np.array([[0.05, 0.3]])
    chi = np.array([[1.0, 0.0]])
    external = np.array([1.0, 0.5])
    materials = MaterialData(total=total, scatter=scatter, fission=nu_fission, chi=chi)
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells, cells), dtype=np.int32),
        delta_x=np.full(cells, 0.5),
        delta_y=np.full(cells, 0.5),
        delta_z=np.full(cells, 0.5),
        bc_x=[1, 1],
        bc_y=[1, 1],
        bc_z=[1, 1],
        geometry=Geometry.SLAB3D,
    )
    boundary_x, boundary_y, boundary_z = prob.zero_boundaries()
    sources = SourceData(
        external=np.broadcast_to(external, (cells, cells, cells, 1, 2)).copy(),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_z=boundary_z,
    )
    flux = fixed_source(materials, sources, geometry, ants.angular_xyz(4), SOLVER)
    matrix = np.diag(total[0]) - scatter[0] - np.outer(chi[0], nu_fission[0])
    reference = np.linalg.solve(matrix, external)
    assert np.allclose(flux, reference, rtol=1e-6)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_dmd_infinite_medium():
    # Strong up and down scattering needs many energy iterations, and DMD
    # after the same number of iterations is much closer to the solution
    groups, cells = 6, 3
    scatter = np.zeros((1, groups, groups))
    for gg in range(groups):
        scatter[0, gg, gg] = 0.3
        if gg + 1 < groups:
            scatter[0, gg + 1, gg] = 0.3
        if gg > 0:
            scatter[0, gg - 1, gg] = 0.3
    materials = MaterialData(
        total=np.ones((1, groups)),
        scatter=scatter,
        fission=np.zeros((1, groups, groups)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells, cells), dtype=np.int32),
        delta_x=np.full(cells, 0.5),
        delta_y=np.full(cells, 0.5),
        delta_z=np.full(cells, 0.5),
        bc_x=[1, 1],
        bc_y=[1, 1],
        bc_z=[1, 1],
        geometry=Geometry.SLAB3D,
    )
    external = np.zeros((cells, cells, cells, 1, groups))
    external[..., 0] = 1.0
    boundary_x, boundary_y, boundary_z = prob.zero_boundaries()
    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_z=boundary_z,
    )
    reference = np.linalg.solve(np.eye(groups) - scatter[0], external[0, 0, 0, 0])
    quadrature = ants.angular_xyz(2)

    # DMD uses dmd_rank + dmd_snapshots = 22 iterations
    kwargs = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)
    solver = SolverData(mg_solver=MultigroupSolver.DMD, **kwargs)
    flux = fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(flux, reference, rtol=1e-8)

    with pytest.warns(match="max_iter_energy"):
        solver = SolverData(max_iter_energy=22, **kwargs)
        flux = fixed_source(materials, sources, geometry, quadrature, solver)
    assert not np.allclose(flux, reference, rtol=1e-7)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_uniform_z_matches_2d(spatial):
    # Reflecting on both z faces makes the problem uniform in z, which is the
    # 2D problem with the upper half of the same product quadrature
    cells_x, cells_y, cells_z = 10, 8, 3
    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
    medium_map[3:7, 2:6] = 1
    external = np.zeros((cells_x, cells_y, 1, 1))
    external[3:7, 2:6] = 1.0
    materials = prob.two_materials()

    geometry_2d = GeometryData(
        medium_map=medium_map,
        delta_x=np.full(cells_x, 0.4),
        delta_y=np.full(cells_y, 0.5),
        geometry=Geometry.SLAB2D,
        space_disc=spatial,
    )
    sources_2d = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    flux_2d = fixed_source_2d(
        materials, sources_2d, geometry_2d, ants.angular_xy(4), SOLVER
    )

    geometry = GeometryData(
        medium_map=np.repeat(medium_map[..., None], cells_z, axis=2),
        delta_x=geometry_2d.delta_x,
        delta_y=geometry_2d.delta_y,
        delta_z=np.full(cells_z, 0.7),
        bc_z=[1, 1],
        geometry=Geometry.SLAB3D,
        space_disc=spatial,
    )
    boundary_x, boundary_y, boundary_z = prob.zero_boundaries()
    sources = SourceData(
        external=np.repeat(external[:, :, None], cells_z, axis=2),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_z=boundary_z,
    )
    flux = fixed_source(materials, sources, geometry, ants.angular_xyz(4), SOLVER)
    assert flux.shape == (cells_x, cells_y, cells_z, 1)
    for kk in range(cells_z):
        assert np.allclose(flux[:, :, kk], flux_2d, rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_axis_permutation(spatial):
    # Relabeling the axes (x, y, z) -> (y, z, x) and the direction cosines
    # with them gives the same flux, so the z sweeps match the x and y sweeps
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(spatial)
    flux = fixed_source(materials, sources, geometry, quadrature, SOLVER)

    geometry_p, sources_p, quadrature_p = prob.permute_axes(
        geometry, sources, quadrature
    )
    flux_p = fixed_source(materials, sources_p, geometry_p, quadrature_p, SOLVER)
    assert np.allclose(flux_p, flux.transpose(2, 0, 1, 3), rtol=1e-8, atol=1e-12)


def _void_problem(boundaries, bc=[[0, 0], [0, 0], [0, 0]], spatial=2):
    # Void with the given incoming flux on each face
    cells = (4, 3, 5)
    materials = MaterialData(
        total=np.zeros((1, 1)), scatter=np.zeros((1, 1, 1)), fission=np.zeros((1, 1, 1))
    )
    geometry = GeometryData(
        medium_map=np.zeros(cells, dtype=np.int32),
        delta_x=np.full(cells[0], 0.5),
        delta_y=np.full(cells[1], 0.3),
        delta_z=np.full(cells[2], 0.4),
        bc_x=bc[0],
        bc_y=bc[1],
        bc_z=bc[2],
        geometry=Geometry.SLAB3D,
        space_disc=spatial,
    )
    sources = SourceData(
        external=np.zeros(cells + (1, 1)),
        boundary_x=boundaries[0],
        boundary_y=boundaries[1],
        boundary_z=boundaries[2],
    )
    return materials, sources, geometry


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_void_boundary(axis, side, spatial):
    # Unit incoming flux on one face, with the faces along the other axes
    # reflecting. Every ordinate entering through the face carries a flux
    # of one, which is half of the quadrature weight.
    boundaries = prob.zero_boundaries()
    boundaries[axis][side] = 1.0
    bc = [[1, 1], [1, 1], [1, 1]]
    bc[axis] = [0, 0]
    materials, sources, geometry = _void_problem(boundaries, bc, spatial)
    flux = fixed_source(materials, sources, geometry, ants.angular_xyz(4), SOLVER)
    assert np.allclose(flux, 0.5, atol=1e-10)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_void_angular(spatial):
    # Unit incoming flux on every face gives a unit angular flux
    boundaries = [np.ones((2, 1, 1, 1, 1)) for _ in range(3)]
    materials, sources, geometry = _void_problem(boundaries, spatial=spatial)
    quadrature = ants.angular_xyz(4)
    solver = SolverData(angular=True)
    angular_flux = fixed_source(materials, sources, geometry, quadrature, solver)
    assert angular_flux.shape == (4, 3, 5, quadrature.angle_w.size, 1)
    assert np.allclose(angular_flux, 1.0, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize("spatial", [1, 2, 3])
def test_void_edges(angular, spatial):
    # Unit incoming flux on every face gives a unit flux on every face
    boundaries = [np.ones((2, 1, 1, 1, 1)) for _ in range(3)]
    materials, sources, geometry = _void_problem(boundaries, spatial=spatial)
    quadrature = ants.angular_xyz(4)
    solver = SolverData(angular=angular, flux_at_edges=1)
    flux_x, flux_y, flux_z = fixed_source(
        materials, sources, geometry, quadrature, solver
    )
    angles = (quadrature.angle_w.size,) if angular else ()
    assert flux_x.shape == (5, 3, 5) + angles + (1,)
    assert flux_y.shape == (4, 4, 5) + angles + (1,)
    assert flux_z.shape == (4, 3, 6) + angles + (1,)
    for flux in (flux_x, flux_y, flux_z):
        assert np.allclose(flux, 1.0, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_angular_to_scalar():
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(materials, sources, geometry, quadrature, SOLVER)
    solver = SolverData(
        angular=True, tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500
    )
    angular_flux = fixed_source(materials, sources, geometry, quadrature, solver)
    scalar_flux = np.einsum("ijknq,n->ijkq", angular_flux, quadrature.angle_w)
    assert np.allclose(scalar_flux, flux, rtol=1e-6)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_compact_source():
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(materials, sources, geometry, quadrature, SOLVER)
    sources.external = CompactSource.from_dense(sources.external)
    flux_compact = fixed_source(materials, sources, geometry, quadrature, SOLVER)
    assert np.allclose(flux_compact, flux, rtol=1e-12)


@pytest.mark.slab3d
def test_threads():
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(materials, sources, geometry, quadrature, SOLVER)
    solver = SolverData(
        num_threads=4, tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500
    )
    flux_threads = fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(flux_threads, flux, rtol=1e-10)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize(
    "solver_kwargs",
    [
        {"parallel": ParallelType.GROUP, "num_threads": 3},
        {"parallel": ParallelType.BOTH, "num_threads": 2},
    ],
)
def test_group_parallel(solver_kwargs):
    # Group parallel (Jacobi) matches the Gauss-Seidel source iteration
    groups = 6
    materials = prob.coupled_materials(groups)
    geometry, sources, quadrature = prob.axis_problem(2, groups)
    kwargs = dict(tol_angular=1e-12, tol_energy=1e-10, max_iter_angular=500)
    reference = fixed_source(
        materials, sources, geometry, quadrature, SolverData(**kwargs)
    )
    solver = SolverData(max_iter_energy=500, **kwargs, **solver_kwargs)
    flux = fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(flux, reference, rtol=1e-6, atol=1e-10)


def _corner_source(angles):
    # Small source in the corner of a pure absorber, with reflecting faces
    # at x = 0, y = 0, and z = 0, where ray effects are large far away
    cells, bc = 16, [1, 0]
    delta = np.full(cells, 0.5)
    materials = MaterialData(
        total=np.array([[0.2]]),
        scatter=np.zeros((1, 1, 1)),
        fission=np.zeros((1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells, cells), dtype=np.int32),
        delta_x=delta,
        delta_y=delta,
        delta_z=delta,
        bc_x=bc,
        bc_y=bc,
        bc_z=bc,
        geometry=Geometry.SLAB3D,
    )
    external = np.zeros((cells, cells, cells, 1, 1))
    external[:2, :2, :2] = 1.0
    boundary_x, boundary_y, boundary_z = prob.zero_boundaries()
    sources = SourceData(
        external=external,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_z=boundary_z,
    )
    quadrature = ants.angular_xyz(angles, bc_x=bc, bc_y=bc, bc_z=bc)
    return materials, sources, geometry, quadrature


@pytest.mark.smoke
@pytest.mark.slab3d
def test_artificial_scattering_ray_effects():
    # Away from the source, as-S4 is much closer to S16 than S4 is. The
    # lagged artificial scattering source needs more than the default 100
    # iterations in the thin absorber.
    reference = fixed_source(*_corner_source(16), SolverData())
    far = (slice(6, None),) * 3
    errors = []
    for sigma_as in [0.0, 2.0]:
        solver = SolverData(sigma_as=sigma_as, beta_as=4.5, max_iter_angular=400)
        flux = fixed_source(*_corner_source(4), solver)
        diff = np.linalg.norm(flux[far] - reference[far])
        errors.append(diff / np.linalg.norm(reference[far]))
    assert errors[1] < 0.5 * errors[0]


@pytest.mark.smoke
@pytest.mark.slab3d
def test_artificial_scattering_permutation():
    # The artificial scattering kernel only depends on Omega . Omega'
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    solver = SolverData(sigma_as=0.5, **KWARGS)
    flux = fixed_source(materials, sources, geometry, quadrature, solver)
    geometry_p, sources_p, quadrature_p = prob.permute_axes(
        geometry, sources, quadrature
    )
    flux_p = fixed_source(materials, sources_p, geometry_p, quadrature_p, solver)
    assert np.allclose(flux_p, flux.transpose(2, 0, 1, 3), rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("moments", [1, 3])
def test_artificial_scattering_angular(moments):
    # The angular flux uses the converged artificial scattering source, so
    # it integrates to the scalar flux
    materials = prob.two_materials(groups=2)
    if moments > 1:
        ratios = [1.0, 0.5, 0.2][:moments]
        materials.scatter = np.stack([r * materials.scatter for r in ratios], -1)
    geometry, sources, quadrature = prob.axis_problem(2)
    flux = fixed_source(
        materials, sources, geometry, quadrature, SolverData(sigma_as=0.5, **KWARGS)
    )
    solver = SolverData(angular=True, sigma_as=0.5, **KWARGS)
    angular_flux = fixed_source(materials, sources, geometry, quadrature, solver)
    scalar_flux = np.einsum("ijknq,n->ijkq", angular_flux, quadrature.angle_w)
    assert np.allclose(scalar_flux, flux, rtol=1e-6)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_artificial_scattering_zero_moments():
    # Zero higher moments give the isotropic as-SN solution
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    solver = SolverData(sigma_as=0.5, **KWARGS)
    reference = fixed_source(materials, sources, geometry, quadrature, solver)
    materials.scatter = np.stack([materials.scatter, 0 * materials.scatter], -1)
    flux = fixed_source(materials, sources, geometry, quadrature, solver)
    assert np.allclose(flux, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize(
    "solver",
    [SolverData(angular=True), SolverData(flux_at_edges=1), SolverData(sigma_as=0.1)],
)
def test_angular_both_reflected(solver):
    # The single known source sweep cannot reflect on both sides
    materials, sources, geometry = _void_problem(
        prob.zero_boundaries(), bc=[[0, 0], [0, 0], [1, 1]]
    )
    with pytest.raises(AssertionError, match="both z"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), solver)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_missing_boundary_z():
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    sources.boundary_z = None
    with pytest.raises(ValueError, match="boundary_z"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), SolverData())


@pytest.mark.smoke
@pytest.mark.slab3d
def test_missing_angle_z():
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    with pytest.raises(ValueError, match="angle_z"):
        fixed_source(materials, sources, geometry, ants.angular_xy(2), SolverData())


@pytest.mark.smoke
@pytest.mark.slab3d
def test_missing_delta_z():
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    geometry.delta_z = None
    with pytest.raises(ValueError, match="delta_z"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), SolverData())


@pytest.mark.smoke
@pytest.mark.slab3d
def test_zero_cosine():
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    quadrature = ants.angular_xyz(2)
    quadrature.angle_z[0] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        fixed_source(materials, sources, geometry, quadrature, SolverData())


@pytest.mark.smoke
@pytest.mark.slab3d
def test_medium_map_shape():
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    geometry.medium_map = np.zeros((4, 3, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="medium_map has shape"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), SolverData())
    geometry.medium_map = np.ones((4, 3, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="material indices"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), SolverData())


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize("sigma_as", [0.0, 0.1])
def test_external_shape(sigma_as):
    materials, sources, geometry = _void_problem(prob.zero_boundaries())
    sources.external = np.zeros((4, 3, 4, 1, 1))
    solver = SolverData(sigma_as=sigma_as)
    with pytest.raises(ValueError, match="external has shape"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(2), solver)


@pytest.mark.smoke
@pytest.mark.slab3d
@pytest.mark.parametrize(
    "solver",
    [SolverData(angular=True), SolverData(flux_at_edges=1), SolverData(sigma_as=0.1)],
)
def test_quadrature_not_ordered(solver):
    # The single known source sweep needs the angles ordered for the
    # reflecting faces
    materials = prob.two_materials(groups=2)
    geometry, sources, _ = prob.axis_problem(2)
    with pytest.raises(ValueError, match="not ordered for bc_x"):
        fixed_source(materials, sources, geometry, ants.angular_xyz(4), solver)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_quadrature_order_scalar_flux():
    # The source iteration lags the reflected edges, so any order works
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    reference = fixed_source(materials, sources, geometry, quadrature, SOLVER)
    flux = fixed_source(materials, sources, geometry, ants.angular_xyz(4), SOLVER)
    assert np.allclose(flux, reference, rtol=1e-10)


@pytest.mark.smoke
@pytest.mark.slab3d
def test_artificial_scattering_no_iterations():
    materials = prob.two_materials(groups=2)
    geometry, sources, quadrature = prob.axis_problem(2)
    solver = SolverData(sigma_as=0.5, max_iter_angular=0)
    with pytest.warns(RuntimeWarning, match="max_iter_angular"):
        fixed_source(materials, sources, geometry, quadrature, solver)
