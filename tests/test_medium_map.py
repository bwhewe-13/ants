########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Medium map and weight matrix tests for 2D and 3D layouts
#
########################################################################

import numpy as np
import pytest

import ants


def _two_materials(groups=3, moments=0):
    # Fuel (material 0) and moderator (material 1)
    total = np.array([[1.0, 1.5, 2.0], [0.5, 0.8, 1.2]])[:, :groups]
    scatter = np.random.default_rng(42).random((2, groups, groups))
    if moments > 0:
        scatter = np.stack([scatter, 0.3 * scatter], axis=-1)
    nu_fission = np.array([[0.2, 0.4, 0.6], [0.0, 0.0, 0.0]])[:, :groups]
    chi = np.array([[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]])[:, :groups]
    return total, scatter, nu_fission, chi


@pytest.mark.smoke
def test_spatial3d_box():
    edges_x = np.linspace(0, 4, 5)
    edges_y = np.linspace(0, 2, 5)
    edges_z = np.linspace(0, 3, 4)
    medium_map = np.zeros((4, 4, 3), dtype=np.int32)
    medium_map = ants.spatial3d(
        medium_map, 1, [[(1.0, 0.5, 1.0), 2.0, 1.0, 2.0]], edges_x, edges_y, edges_z
    )
    reference = np.zeros((4, 4, 3), dtype=np.int32)
    reference[1:3, 1:3, 1:3] = 1
    assert np.array_equal(medium_map, reference)


@pytest.mark.smoke
def test_spatial3d_matches_spatial2d():
    # A box spanning all of z is the 2D rectangle at every z cell
    edges_x = np.linspace(0, 5, 11)
    edges_y = np.linspace(0, 3, 7)
    edges_z = np.linspace(0, 2, 5)
    coords_2d = [[(1.0, 0.5), 2.5, 1.5], [(3.5, 2.0), 1.5, 1.0]]
    coords_3d = [[(x, y, 0.0), dx, dy, 2.0] for [(x, y), dx, dy] in coords_2d]
    map_2d = ants.spatial2d(
        np.zeros((10, 6), dtype=np.int32), 2, coords_2d, edges_x, edges_y
    )
    map_3d = ants.spatial3d(
        np.zeros((10, 6, 4), dtype=np.int32), 2, coords_3d, edges_x, edges_y, edges_z
    )
    assert np.array_equal(map_3d, np.repeat(map_2d[..., None], 4, axis=2))


@pytest.mark.smoke
def test_spatial3d_off_edge():
    edges = np.linspace(0, 1, 5)
    medium_map = np.zeros((4, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="edges_z"):
        ants.spatial3d(
            medium_map, 1, [[(0, 0, 0.1), 0.5, 0.5, 0.5]], edges, edges, edges
        )


@pytest.mark.smoke
def test_extrude3d_layers():
    edges_z = np.linspace(0, 6, 7)
    fuel = np.array([[1, 0], [2, 1]], dtype=np.int32)
    reflector = np.zeros((2, 2), dtype=np.int32)
    medium_map = ants.extrude3d([[reflector, "0 - 1, 5 - 6"], [fuel, "1 - 5"]], edges_z)
    assert medium_map.shape == (2, 2, 6)
    assert medium_map.dtype == np.int32
    assert np.all(medium_map[..., [0, 5]] == 0)
    for kk in range(1, 5):
        assert np.array_equal(medium_map[..., kk], fuel)


@pytest.mark.smoke
def test_extrude3d_negative_z():
    edges_z = np.linspace(-2, 2, 5)
    fuel = np.ones((2, 2), dtype=np.int32)
    reflector = np.zeros((2, 2), dtype=np.int32)
    medium_map = ants.extrude3d(
        [[reflector, "-2 - -1, 1 - 2"], [fuel, "-1 - 1"]], edges_z
    )
    assert np.array_equal(medium_map[0, 0], [0, 1, 1, 0])


@pytest.mark.smoke
def test_spatial1d_regions():
    # Regions with and without spaces, and below zero
    edges_x = np.linspace(-1, 3, 5)
    medium_map = ants.spatial1d([[0, "a", "-1-0, 2 - 3"], [1, "b", "0 - 2"]], edges_x)
    assert np.array_equal(medium_map, [0, 1, 1, 0])
    with pytest.raises(ValueError, match="start - stop"):
        ants.spatial1d([[0, "a", "-1 3"]], edges_x)


@pytest.mark.smoke
def test_extrude3d_gap():
    edges_z = np.linspace(0, 3, 4)
    with pytest.raises(ValueError, match="cover"):
        ants.extrude3d([[np.zeros((2, 2)), "0 - 2"]], edges_z)


@pytest.mark.smoke
@pytest.mark.parametrize("moments", [0, 1])
def test_weight_spatial2d_shapes(moments):
    total, scatter, nu_fission, chi = _two_materials(moments=moments)
    weight_matrix = np.zeros((3, 2, 2))
    weight_matrix[..., 0] = [[1.0, 0.25], [0.5, 0.25], [0.0, 1.0]]
    weight_matrix[..., 1] = 1.0 - weight_matrix[..., 0]
    medium_map, xs_total, xs_scatter, xs_fission, xs_chi = ants.weight_spatial2d(
        weight_matrix, total, scatter, nu_fission, chi=chi
    )
    # Four unique mixtures: 0, 0.25, 0.5, and 1 fuel
    assert medium_map.shape == (3, 2)
    assert xs_total.shape == (4, 3)
    assert xs_scatter.shape == (4,) + scatter.shape[1:]
    assert xs_fission.shape == (4, 3)
    assert xs_chi.shape == (4, 3)
    # Each cell has the volume weighted cross sections
    for (ii, jj), mat in np.ndenumerate(medium_map):
        rho = weight_matrix[ii, jj, 0]
        assert np.allclose(xs_total[mat], rho * total[0] + (1 - rho) * total[1])
        assert np.allclose(xs_scatter[mat], rho * scatter[0] + (1 - rho) * scatter[1])
        assert np.allclose(xs_fission[mat], rho * nu_fission[0])


@pytest.mark.smoke
def test_weight_spatial2d_chi():
    # Only the fuel produces neutrons, so any mixture with fuel keeps its
    # spectrum and the moderator alone keeps its own
    total, scatter, nu_fission, chi = _two_materials()
    weight_matrix = np.array([[[0.3, 0.7], [0.0, 1.0]]])
    medium_map, *_, xs_chi = ants.weight_spatial2d(
        weight_matrix, total, scatter, nu_fission, chi=chi
    )
    assert np.allclose(xs_chi[medium_map[0, 0]], chi[0])
    assert np.allclose(xs_chi[medium_map[0, 1]], chi[1])


@pytest.mark.smoke
def test_weight_spatial2d_decimals():
    total, scatter, _, _ = _two_materials()
    fission = np.zeros((2, 3, 3))
    weight_matrix = np.zeros((2, 1, 2))
    weight_matrix[:, 0, 0] = [0.50001, 0.49999]
    weight_matrix[..., 1] = 1.0 - weight_matrix[..., 0]
    medium_map, xs_total, *_ = ants.weight_spatial2d(
        weight_matrix, total, scatter, fission
    )
    assert xs_total.shape[0] == 2
    medium_map, xs_total, *_ = ants.weight_spatial2d(
        weight_matrix, total, scatter, fission, decimals=4
    )
    assert xs_total.shape[0] == 1
    assert np.all(medium_map == 0)
    assert np.allclose(xs_total[0], 0.5 * (total[0] + total[1]))


@pytest.mark.smoke
def test_weight_spatial2d_decimals_empty():
    # Rounding every weight to zero leaves a cell without material
    total, scatter, _, _ = _two_materials()
    total = np.concatenate([total, total[:1]])
    scatter = np.concatenate([scatter, scatter[:1]])
    fission = np.zeros((3, 3, 3))
    weight_matrix = np.full((1, 1, 3), 1 / 3)
    with pytest.raises(ValueError, match="no material"):
        ants.weight_spatial2d(weight_matrix, total, scatter, fission, decimals=0)


@pytest.mark.smoke
def test_weight_spatial2d_chi_fission_matrix():
    # The fission spectrum is weighted by xs_fission summed over groups
    total, scatter, _, chi = _two_materials(groups=2)
    weight_matrix = np.array([[[1.0, 0.0], [0.5, 0.5]]])
    fission = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="with chi"):
        ants.weight_spatial2d(weight_matrix, total, scatter, fission, chi=chi[:, :2])


@pytest.mark.smoke
def test_weight_spatial3d_matches_2d():
    total, scatter, _, _ = _two_materials()
    fission = np.random.default_rng(7).random((2, 3, 3))
    rho = np.random.default_rng(3).choice([0.0, 0.2, 0.6, 1.0], size=(4, 3, 5))
    weight_matrix = np.stack([rho, 1.0 - rho], axis=-1)
    data_3d = ants.weight_spatial3d(weight_matrix, total, scatter, fission)
    data_2d = ants.weight_spatial2d(
        weight_matrix.reshape(4, 15, 2), total, scatter, fission
    )
    assert data_3d[0].shape == (4, 3, 5)
    assert np.array_equal(data_3d[0].reshape(4, 15), data_2d[0])
    for xs_3d, xs_2d in zip(data_3d[1:], data_2d[1:]):
        assert np.allclose(xs_3d, xs_2d)


@pytest.mark.smoke
def test_weight_spatial_dimensions():
    total, scatter, _, _ = _two_materials()
    fission = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="cells_z"):
        ants.weight_spatial3d(np.ones((2, 2, 2)), total, scatter, fission)
    with pytest.raises(ValueError, match="cells_y, materials"):
        ants.weight_spatial2d(np.ones((2, 2, 2, 2)), total, scatter, fission)
