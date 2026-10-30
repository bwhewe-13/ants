########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for the standard energy group structures, energy grids, and
# built-in energy dependent sources.
#
########################################################################

import os

import numpy as np
import pytest

import ants
from ants.data.energy import ENERGY_STRUCTURES
from ants.data.external import AMBE_EDGES, AMBE_MAGNITUDE
from ants.main import DATA_PATH

LOCAL_SOURCES = os.path.dirname(os.path.normpath(DATA_PATH))

EXPECTED_GROUPS = {
    "2g": 2,
    "44g": 44,
    "70g": 70,
    "anl_27": 27,
    "apollo_99": 99,
    "casmo_2": 2,
    "casmo_3": 3,
    "casmo_4": 4,
    "casmo_7": 7,
    "casmo_8": 8,
    "casmo_9": 9,
    "casmo_12": 12,
    "casmo_14": 14,
    "casmo_16": 16,
    "casmo_18": 18,
    "casmo_23": 23,
    "casmo_25": 25,
    "casmo_40": 40,
    "casmo_70": 70,
    "ccfe_660": 660,
    "csewg_239": 240,
    "ecco_33": 33,
    "ecco_1962": 1962,
    "ecco_1968": 1968,
    "epri_cpm_69": 69,
    "eurlib_100": 100,
    "gam_i_68": 68,
    "gam_ii_100": 100,
    "lanl_30": 30,
    "lanl_70": 70,
    "lanl_80": 80,
    "lanl_187": 187,
    "lanl_618": 618,
    "laser_thermos_35": 35,
    "macro2g": 2,
    "macro4g": 4,
    "macro8g": 8,
    "micro168g": 168,
    "mupo_43": 43,
    "rahab_aecl_89": 89,
    "rrd_50": 50,
    "sand_ii_620": 620,
    "sand_iia_640": 640,
    "scale_44": 44,
    "scale_56": 56,
    "scale_238": 238,
    "scale_252": 252,
    "sfr_24": 24,
    "sfr_240g": 240,
    "shem_cea_281": 281,
    "shem_cea_epm_361": 361,
    "shem_epm_295": 295,
    "shem_epm_315": 315,
    "tripoli_315": 315,
    "ukaea_142": 142,
    "ukaea_1025": 1025,
    "ukaea_1067": 1067,
    "ukaea_1102": 1102,
    "vit_j_lwpc_175": 175,
    "vitamin_e_174": 174,
    "vitamin_j_175": 175,
    "wims_69": 69,
    "wims_172": 172,
    "xmas_lwpc_172": 172,
}


@pytest.mark.smoke
@pytest.mark.parametrize(("name", "groups"), sorted(EXPECTED_GROUPS.items()))
def test_structure_groups(name, groups):
    edges_g = ants.get_structure(name)
    assert edges_g.shape == (groups + 1,)


@pytest.mark.parametrize("name", sorted(ENERGY_STRUCTURES))
def test_structure_descending_ev(name):
    edges_g = ants.get_structure(name)
    assert np.all(np.diff(edges_g) < 0)
    assert edges_g[-1] >= 0.0
    # Top edge in eV, not MeV (LASER-THERMOS is thermal only)
    assert edges_g[0] > 1.0


def test_structure_lookup():
    assert np.array_equal(ants.get_structure("CASMO_70"), ENERGY_STRUCTURES["casmo_70"])
    assert np.array_equal(
        ants.get_structure("lanl_618"), ants.get_structure("njoy_ign34")
    )
    assert "casmo_70" in ants.list_structures("casmo")
    with pytest.raises(KeyError):
        ants.get_structure("casmo_71")


def test_energy_grid_named():
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid("casmo_70", 70, 10)
    assert np.array_equal(edges_g, ants.get_structure("casmo_70"))
    assert np.array_equal(edges_gidx_u, np.arange(71))
    assert edges_gidx_c[0] == 0 and edges_gidx_c[-1] == 70
    assert edges_gidx_c.shape == (11,)
    assert edges_gidx_c.dtype == np.int32


def test_energy_grid_public_legacy():
    edges_g, _ = ants.energy_grid(618, 618)
    assert np.array_equal(edges_g, ants.get_structure("lanl_618"))
    edges_g, _ = ants.energy_grid(361, 361)
    assert np.array_equal(edges_g, ants.get_structure("shem_cea_epm_361"))


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DATA_PATH, "energy_grids.npz")),
    reason="Local energy grids are not available",
)
def test_energy_grid_local_87():
    edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, 87, 43)
    assert np.all(np.diff(edges_g) < 0)
    assert edges_g[0] > 1e7
    assert np.all(np.diff(edges_gidx_c) > 0)
    assert edges_gidx_c[-1] == 87


def test_deuterium_tritium_order():
    # Same physical group for ascending and descending edges
    edges_g = ants.get_structure("vitamin_j_175")
    groups = edges_g.shape[0] - 1
    group_desc = np.argmax(ants.boundary1d.deuterium_tritium(0, edges_g)[0, 0])
    group_asc = np.argmax(ants.boundary1d.deuterium_tritium(0, edges_g[::-1])[0, 0])
    assert group_desc == groups - 1 - group_asc


def test_ambe_order_and_unchanged_edges():
    edges_g = ants.get_structure("lanl_70")
    edges_copy = edges_g.copy()
    delta_x = np.ones((4,))
    source_desc = ants.external1d.ambe(delta_x, 1, edges_g)
    source_asc = ants.external1d.ambe(delta_x, 1, edges_g[::-1].copy())
    assert np.array_equal(edges_g, edges_copy)
    assert np.allclose(source_desc[1, 0], source_asc[1, 0, ::-1])
    assert source_desc[1, 0].sum() > 0.0


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(LOCAL_SOURCES, "external", "AmBe_source_050G.npz")),
    reason="Local AmBe data are not available",
)
def test_ambe_matches_local():
    data = np.load(os.path.join(LOCAL_SOURCES, "external", "AmBe_source_050G.npz"))
    assert np.allclose(AMBE_EDGES[::-1] * 1e-6, data["edges"], rtol=1e-12)
    assert np.allclose(AMBE_MAGNITUDE[::-1], data["magnitude"], rtol=1e-12)
