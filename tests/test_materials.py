########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for HDF5 library reading, nuclide names, compositions, and
# named materials. Uses a small synthetic HDF5 library.
#
########################################################################

import numpy as np
import pytest

import ants
from ants.constants import (
    AVAGADRO,
    CM_TO_BARNS,
    HYDROGEN_MM,
    URANIUM_235_MM,
    URANIUM_238_MM,
    URANIUM_HYDRIDE_RHO,
)
from ants.materials import NAMED_MATERIALS

h5py = pytest.importorskip("h5py")

GROUPS = 3
EDGES_G = np.array([2e7, 1e5, 1.0, 1e-5])


def _scatter(scale):
    # Stored as [in, out, legendre]
    scatter = np.zeros((GROUPS, GROUPS, 1))
    scatter[0, 0, 0] = 1.0 * scale
    scatter[0, 1, 0] = 0.5 * scale
    scatter[1, 2, 0] = 0.25 * scale
    scatter[2, 2, 0] = 2.0 * scale
    return scatter


@pytest.fixture(scope="module")
def library_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("library") / "synthetic.h5"
    with h5py.File(path, "w") as f:
        f.attrs["n_groups"] = GROUPS
        f.attrs["n_legendre"] = 1
        f.attrs["library_type"] = "microscopic"
        f.create_dataset("group_boundaries", data=EDGES_G)
        # Fissile nuclide at two temperatures (chi_matrix)
        for temperature, scale in [(293.6, 1.0), (600.0, 2.0)]:
            data = f.create_group(f"U-235_{temperature:g}K")
            data.attrs["temperature"] = temperature
            data.create_dataset("xs_total", data=scale * np.array([5.0, 10.0, 50.0]))
            data.create_dataset("xs_scatter", data=_scatter(scale))
            data.create_dataset("xs_fission", data=np.array([1.0, 2.0, 30.0]))
            data.create_dataset("nu", data=np.array([2.5, 2.4, 2.4]))
            chi_matrix = np.array([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.6, 0.4, 0.0]])
            data.create_dataset("chi_matrix", data=chi_matrix)
        # Fissile nuclide with chi vector only
        data = f.create_group("U-238")
        data.attrs["temperature"] = 293.6
        data.create_dataset("xs_total", data=np.array([6.0, 9.0, 12.0]))
        data.create_dataset("xs_scatter", data=_scatter(3.0))
        data.create_dataset("xs_fission", data=np.array([0.5, 0.0, 0.0]))
        data.create_dataset("nu", data=np.array([2.6, 2.4, 2.4]))
        data.create_dataset("chi", data=np.array([0.9, 0.1, 0.0]))
        # Non-fissile nuclides
        for name in ["H-1", "Ga-69", "Ga-71"]:
            data = f.create_group(name)
            data.attrs["temperature"] = 293.6
            data.create_dataset("xs_total", data=np.array([4.0, 20.0, 30.0]))
            data.create_dataset("xs_scatter", data=_scatter(4.0))
    return str(path)


def test_library_metadata(library_path):
    library = ants.Library(library_path)
    assert library.groups == GROUPS
    assert np.array_equal(library.edges_g, EDGES_G)
    assert library.nuclides() == ["Ga-69", "Ga-71", "H-1", "U-235", "U-238"]
    assert library.temperatures("U-235") == [293.6, 600.0]
    with pytest.raises(KeyError):
        library.microscopic("Pu-239")


def test_library_microscopic(library_path):
    library = ants.Library(library_path)
    total, scatter, fission = library.microscopic("U-235")
    assert np.array_equal(total, [5.0, 10.0, 50.0])
    # Transposed to [out, in]
    assert np.array_equal(scatter, _scatter(1.0)[..., 0].T)
    chi_matrix = np.array([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.6, 0.4, 0.0]])
    nu_fission = np.array([2.5, 4.8, 72.0])
    assert np.allclose(fission, chi_matrix.T * nu_fission[None, :])
    # Fission neutrons per group are preserved
    assert np.allclose(fission.sum(axis=0), nu_fission)
    # Chi vector
    _, _, fission = library.microscopic("U-238")
    assert np.allclose(fission[:, 0], [0.9 * 1.3, 0.1 * 1.3, 0.0])


def test_library_temperature(library_path):
    library = ants.Library(library_path)
    assert library.microscopic("U-235", temperature=550)[0][0] == 10.0
    assert library.microscopic("U-235", temperature=300)[0][0] == 5.0
    assert library.microscopic("U-235")[0][0] == 5.0


def test_library_macroscopic(library_path):
    library = ants.Library(library_path)
    composition = {"U-235": 0.04, "H-1": 0.06}
    xs_total, xs_scatter, xs_fission = library.macroscopic(composition)
    u235 = library.microscopic("U-235")
    h1 = library.microscopic("H-1")
    for idx, xs in enumerate([xs_total, xs_scatter, xs_fission]):
        assert np.allclose(xs, 0.04 * u235[idx] + 0.06 * h1[idx])


def test_library_materials(library_path):
    library = ants.Library(library_path)
    mat_data = library.materials([{"U-235": 0.04}, {}, {"H-1": 0.06}])
    assert mat_data.total.shape == (3, GROUPS)
    assert mat_data.scatter.shape == (3, GROUPS, GROUPS)
    assert np.all(mat_data.total[1] == 0.0)
    assert np.allclose(mat_data.velocity, ants.energy_velocity(GROUPS, EDGES_G))


@pytest.mark.parametrize(
    ("nuclide", "expected"),
    [
        ("94239", "Pu-239"),
        ("94239.80c", "Pu-239"),
        ('"94239.700nm"', "Pu-239"),
        (92235, "U-235"),
        ("1001.70c", "H-1"),
        ("95642", "Am-242"),
        ("95242", "Am-242m"),
        ("52527", "Te-127m"),
        ("pu-239", "Pu-239"),
        ("U-235", "U-235"),
        ("PU239", "Pu-239"),
        ("Am-242m", "Am-242m"),
        ("235U", "U-235"),
        ("31000.50c", "Ga"),
        ("FE", "Fe"),
    ],
)
def test_normalize_nuclide(nuclide, expected):
    assert ants.normalize_nuclide(nuclide) == expected


def test_parse_composition_formats():
    mcnp = "m1 94239.80c 3.7047E-02 31069.80c 8.3603E-04"
    partisn = '"94239.700nm"  0.037047\n "31069.700nm"  0.00083603;'
    fortran = "94239 3.7047-2 31069 8.3603-4"
    pairs = [("Pu-239", 0.037047), ("ga-69", 8.3603e-4)]
    expected = {"Pu-239": 0.037047, "Ga-69": 8.3603e-4}
    for composition in [mcnp, partisn, fortran, pairs, dict(pairs)]:
        parsed = ants.parse_composition(composition)
        assert parsed.keys() == expected.keys()
        assert np.allclose(list(parsed.values()), list(expected.values()))


def test_parse_composition_natural():
    parsed = ants.parse_composition("31000.50c 0.001")
    assert np.isclose(parsed["Ga-69"] + parsed["Ga-71"], 0.001)
    with pytest.raises(ValueError):
        ants.parse_composition("50000 0.01")
    with pytest.raises(ValueError):
        ants.parse_composition("92235 -0.9")


def test_create_material(library_path):
    library = ants.Library(library_path)
    mcnp = ants.create_material("m1 92235.80c 0.04 1001.80c 0.06", library)
    partisn = ants.create_material('"92235.700nm" 0.04 "1001.700nm" 0.06', library)
    direct = library.materials([{"U-235": 0.04, "H-1": 0.06}])
    for mat_data in [mcnp, partisn]:
        assert np.allclose(mat_data.total, direct.total)
        assert np.allclose(mat_data.scatter, direct.scatter)
        assert np.allclose(mat_data.fission, direct.fission)
    combined = ants.combine_materials([mcnp, partisn])
    assert combined.total.shape == (2, GROUPS)


def test_materials_with_library(library_path):
    mat_data, key = ants.materials(
        GROUPS, ["vacuum", {"U-235": 0.04}], key=True, library=library_path
    )
    assert key == {0: "vacuum", 1: {"U-235": 0.04}}
    assert np.all(mat_data.total[0] == 0.0)
    with pytest.raises(AssertionError):
        ants.materials(GROUPS + 1, ["vacuum"], library=library_path)


def test_named_material_uranium_hydride():
    enrichment = 0.2
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    composition = ants.named_material("uranium-hydride-%20%")
    # Number density used by the local uranium hydride data
    n1 = URANIUM_HYDRIDE_RHO * AVAGADRO / (molar + 3 * HYDROGEN_MM) * CM_TO_BARNS * 3
    assert np.isclose(composition["H-1"], n1)
    assert np.isclose(
        composition["H-1"], 3 * (composition["U-235"] + composition["U-238"])
    )
    assert np.isclose(composition["U-235"] / composition["U-238"], 0.25)


@pytest.mark.parametrize("name", sorted(NAMED_MATERIALS))
def test_named_materials(name):
    material = name + "-%5%" if NAMED_MATERIALS[name][1] else name
    composition = ants.named_material(material)
    assert all(density > 0 for density in composition.values())
    assert all("-" in nuclide for nuclide in composition)
    with pytest.raises(ValueError):
        ants.named_material("unobtainium")
