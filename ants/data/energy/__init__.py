########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Standard multigroup energy structures (NJOY, Serpent, CASMO, SCALE)
#
########################################################################

"""Registry of standard multigroup energy group structures.

All boundary arrays are in eV and descending order (fast -> thermal), so
that group 0 is the highest energy group. This is the ANTS convention
for energy grids and cross sections.

The data were copied from PyCNiC (MIT License), which collected them
from the NJOY2016 GROUPR module, the Serpent data repository, the
OpenMC ``mgxs`` module, and published benchmarks (C5G7, XMAS, VITAMIN,
SHEM).
"""

import numpy as np

from ants.data.energy import _njoy, _serpent

########################################################################
# Built-in / Benchmark Structures
########################################################################

_BOUNDS_7G_PWR = np.array(
    [1.973e7, 6.065e6, 3.679e5, 2.248e3, 4.140e0, 1.000e0, 1.000e-5, 0.0]
)

_BOUNDS_2G = np.array([1.0e7, 0.625e0, 0.0])

_BOUNDS_44G_BWR = np.array(
    [
        2.0e7,
        1.5e7,
        1.0e7,
        8.0e6,
        6.0e6,
        5.0e6,
        4.0e6,
        3.0e6,
        2.5e6,
        2.0e6,
        1.5e6,
        1.0e6,
        8.0e5,
        6.0e5,
        5.0e5,
        4.0e5,
        3.0e5,
        2.5e5,
        2.0e5,
        1.5e5,
        1.0e5,
        8.0e4,
        6.0e4,
        5.0e4,
        4.0e4,
        3.0e4,
        2.5e4,
        2.0e4,
        1.5e4,
        1.0e4,
        8.0e3,
        5.0e3,
        3.0e3,
        2.0e3,
        1.0e3,
        5.0e2,
        3.0e2,
        1.0e2,
        5.0e1,
        3.0e1,
        1.0e1,
        5.0e0,
        1.0e0,
        1.0e-5,
        0.0,
    ]
)

########################################################################
# Serpent Nuclear Code Repository Structures
# https://serpent.vtt.fi/docs/data/input_data.html#repository-ene
########################################################################

_BOUNDS_MACRO2G = np.array([1e09, 0.625, 0.0])

_BOUNDS_MACRO4G = np.array([1e09, 8.21e05, 5500, 0.625, 0.0])

_BOUNDS_MACRO8G = np.array([1e09, 8.21e05, 5500, 4, 0.625, 0.28, 0.14, 0.058, 0.0])

_BOUNDS_MICRO168G = np.array(
    [
        1e09,
        1e07,
        8.1873e06,
        6.7032e06,
        6.0653e06,
        5.4881e06,
        4.4933e06,
        3.6788e06,
        3.0119e06,
        2.466e06,
        2.2313e06,
        2.019e06,
        1.653e06,
        1.3534e06,
        1.2246e06,
        1.108e06,
        1.0026e06,
        9.0718e05,
        8.21e05,
        6.081e05,
        5.5023e05,
        4.9787e05,
        4.5049e05,
        4.0762e05,
        3.0197e05,
        2.7324e05,
        2.4724e05,
        1.8316e05,
        1.2277e05,
        1.1109e05,
        82298,
        67380,
        55166,
        40868,
        36979,
        29283,
        27394,
        24788,
        16616,
        15034,
        11138,
        9118.8,
        7465.9,
        5500,
        5004.5,
        3526.6,
        3354.6,
        2248.7,
        2034.7,
        1507.3,
        1433.8,
        1234.1,
        1010.4,
        914.24,
        748.52,
        677.29,
        454,
        371.7,
        304.32,
        204,
        148.62,
        136.74,
        91.661,
        75.674,
        67.904,
        55.595,
        51.578,
        48.252,
        45.517,
        40.169,
        37.267,
        33.72,
        30.511,
        27.608,
        24.98,
        22.603,
        19.455,
        15.928,
        13.71,
        11.225,
        9.9055,
        9.1898,
        8.3153,
        7.524,
        6.1601,
        5.3464,
        5.0435,
        4.1293,
        4,
        3.3808,
        3.3,
        2.7679,
        2.72,
        2.6,
        2.55,
        2.36,
        2.13,
        2.1,
        2.02,
        1.93,
        1.84,
        1.755,
        1.67,
        1.59,
        1.5,
        1.475,
        1.445,
        1.37,
        1.3375,
        1.3,
        1.235,
        1.17,
        1.15,
        1.1253,
        1.11,
        1.097,
        1.071,
        1.045,
        1.035,
        1.02,
        0.996,
        0.986,
        0.972,
        0.95,
        0.93,
        0.91,
        0.86,
        0.85,
        0.79,
        0.78,
        0.705,
        0.625,
        0.54,
        0.5,
        0.485,
        0.433,
        0.4,
        0.391,
        0.35,
        0.32,
        0.3145,
        0.3,
        0.28,
        0.248,
        0.22,
        0.189,
        0.18,
        0.16,
        0.14,
        0.134,
        0.115,
        0.1,
        0.095,
        0.08,
        0.077,
        0.067,
        0.058,
        0.05,
        0.042,
        0.035,
        0.03,
        0.025,
        0.02,
        0.015,
        0.01,
        0.0069,
        0.005,
        0.003,
        0.0,
    ]
)

_BOUNDS_SFR240G = np.array(
    [
        1e09,
        1e07,
        9.6721e06,
        9.3442e06,
        9.0163e06,
        8.6884e06,
        8.3605e06,
        8.0327e06,
        7.7048e06,
        7.3769e06,
        7.049e06,
        6.7211e06,
        6.3932e06,
        6.0653e06,
        5.8664e06,
        5.6676e06,
        5.4687e06,
        5.2698e06,
        5.0709e06,
        4.8721e06,
        4.6732e06,
        4.4743e06,
        4.2754e06,
        4.0766e06,
        3.8777e06,
        3.6788e06,
        3.5582e06,
        3.4376e06,
        3.3169e06,
        3.1963e06,
        3.0757e06,
        2.9551e06,
        2.8344e06,
        2.7138e06,
        2.5932e06,
        2.4726e06,
        2.3519e06,
        2.2313e06,
        2.1581e06,
        2.085e06,
        2.0118e06,
        1.9387e06,
        1.8655e06,
        1.7924e06,
        1.7192e06,
        1.646e06,
        1.5729e06,
        1.4997e06,
        1.4266e06,
        1.3534e06,
        1.309e06,
        1.2646e06,
        1.2203e06,
        1.1759e06,
        1.1315e06,
        1.0871e06,
        1.0427e06,
        9.9837e05,
        9.5399e05,
        9.0961e05,
        8.6523e05,
        8.2085e05,
        7.9394e05,
        7.6702e05,
        7.4011e05,
        7.1319e05,
        6.8628e05,
        6.5936e05,
        6.3245e05,
        6.0553e05,
        5.7862e05,
        5.517e05,
        5.2479e05,
        4.9787e05,
        4.8155e05,
        4.6522e05,
        4.489e05,
        4.3257e05,
        4.1625e05,
        3.9992e05,
        3.836e05,
        3.6727e05,
        3.5095e05,
        3.3462e05,
        3.183e05,
        3.0197e05,
        2.9207e05,
        2.8217e05,
        2.7227e05,
        2.6237e05,
        2.5247e05,
        2.4257e05,
        2.3266e05,
        2.2276e05,
        2.1286e05,
        2.0296e05,
        1.9306e05,
        1.8316e05,
        1.7715e05,
        1.7115e05,
        1.6514e05,
        1.5914e05,
        1.5313e05,
        1.4713e05,
        1.4112e05,
        1.3511e05,
        1.2911e05,
        1.231e05,
        1.171e05,
        1.1109e05,
        1.0672e05,
        1.0235e05,
        97977,
        93606,
        89235,
        84864,
        80493,
        76122,
        71751,
        67380,
        64728,
        62077,
        59426,
        56775,
        54124,
        51472,
        48821,
        46170,
        43519,
        40868,
        39260,
        37652,
        36044,
        34436,
        32828,
        31220,
        29612,
        28004,
        26396,
        24788,
        23812,
        22837,
        21862,
        20886,
        19911,
        18936,
        17960,
        16985,
        16010,
        15034,
        14443,
        13851,
        13260,
        12668,
        12077,
        11485,
        10894,
        10302,
        9710.4,
        9118.8,
        8760,
        8401.2,
        8042.4,
        7683.6,
        7324.9,
        6966.1,
        6607.3,
        6248.5,
        5889.7,
        5530.9,
        5313.2,
        5095.6,
        4878,
        4660.4,
        4442.7,
        4225.1,
        4007.5,
        3789.9,
        3572.3,
        3354.6,
        3222.6,
        3090.6,
        2958.6,
        2826.7,
        2694.7,
        2562.7,
        2430.7,
        2298.7,
        2166.7,
        2034.7,
        1954.6,
        1874.6,
        1794.5,
        1714.5,
        1634.4,
        1554.3,
        1474.3,
        1394.2,
        1314.2,
        1234.1,
        1185.5,
        1137,
        1088.4,
        1039.9,
        991.31,
        942.75,
        894.19,
        845.64,
        797.08,
        748.52,
        719.07,
        689.62,
        660.16,
        630.71,
        601.26,
        571.81,
        542.36,
        512.9,
        483.45,
        454,
        439.8,
        425.61,
        411.41,
        397.21,
        383.02,
        368.82,
        354.62,
        340.42,
        326.23,
        312.03,
        295.72,
        279.41,
        263.1,
        246.79,
        230.49,
        214.18,
        197.87,
        181.56,
        165.25,
        148.94,
        0.0,
    ]
)

_BOUNDS_SFR24G = np.array(
    [
        1e09,
        1e07,
        6.0653e06,
        3.6788e06,
        2.2313e06,
        1.3534e06,
        8.2085e05,
        4.9787e05,
        3.0197e05,
        1.8316e05,
        1.1109e05,
        67380,
        40868,
        24788,
        15034,
        9118.8,
        5530.9,
        3354.6,
        2034.7,
        1234.1,
        748.52,
        454,
        312.03,
        148.94,
        0.0,
    ]
)


########################################################################
# Registry
########################################################################


def _serpent_ev(bounds):
    # Serpent structures are stored in MeV (descending)
    return bounds * 1e6


def _njoy_builtin(ign):
    # NJOY structures are stored in eV (ascending)
    return getattr(_njoy, f"BOUNDS_NJOY_IGN{ign}")[::-1].copy()


_NJOY_ALIASES = {
    "csewg_239": 2,
    "lanl_30": 3,
    "anl_27": 4,
    "rrd_50": 5,
    "gam_i_68": 6,
    "gam_ii_100": 7,
    "laser_thermos_35": 8,
    "epri_cpm_69": 9,
    "lanl_187": 10,
    "lanl_70": 11,
    "sand_ii_620": 12,
    "lanl_80": 13,
    "eurlib_100": 14,
    "sand_iia_640": 15,
    "vitamin_e_174": 16,
    "vitamin_j_175": 17,
    "xmas_nea_lanl": 18,
    "ecco_33": 19,
    "ecco_1968": 20,
    "tripoli_315": 21,
    "xmas_lwpc_172": 22,
    "vit_j_lwpc_175": 23,
    "shem_cea_281": 24,
    "shem_epm_295": 25,
    "shem_cea_epm_361": 26,
    "shem_epm_315": 27,
    "rahab_aecl_89": 28,
    "ccfe_660": 29,
    "ukaea_1025": 30,
    "ukaea_1067": 31,
    "ukaea_1102": 32,
    "ukaea_142": 33,
    "lanl_618": 34,
    "apollo_99": 35,
    "ecco_1962": 36,
}

ENERGY_STRUCTURES = {
    # Built-in / benchmark
    "c5g7_7g": _BOUNDS_7G_PWR,
    "7g_pwr": _BOUNDS_7G_PWR,
    "2g": _BOUNDS_2G,
    "44g": _BOUNDS_44G_BWR,
    "bwr_44g": _BOUNDS_44G_BWR,
    "70g": _serpent_ev(_serpent.BOUNDS_DEFAULT_70),
    # CASMO family
    **{
        f"casmo_{gg}": _serpent_ev(getattr(_serpent, f"BOUNDS_CASMO_{gg}"))
        for gg in (2, 3, 4, 7, 8, 9, 12, 14, 16, 18, 23, 25, 40, 70)
    },
    # SCALE family
    **{
        f"scale_{gg}": _serpent_ev(getattr(_serpent, f"BOUNDS_SCALE_{gg}"))
        for gg in (44, 56, 238, 252)
    },
    # Serpent nuclear code repository
    "macro2g": _BOUNDS_MACRO2G,
    "macro4g": _BOUNDS_MACRO4G,
    "macro8g": _BOUNDS_MACRO8G,
    "micro168g": _BOUNDS_MICRO168G,
    # NJOY standard structures (IGN values from GROUPR module)
    **{f"njoy_ign{ign}": _njoy_builtin(ign) for ign in range(2, 37)},
    # NJOY aliases (descriptive names)
    **{name: _njoy_builtin(ign) for name, ign in _NJOY_ALIASES.items()},
    # Other structures
    "sfr_240g": _BOUNDS_SFR240G,
    "sfr_24": _BOUNDS_SFR24G,
    "mupo_43": _serpent_ev(_serpent.BOUNDS_MUPO_43),
    "wims_172": _serpent_ev(_serpent.BOUNDS_WIMS_172),
    "wims_69": _serpent_ev(_serpent.BOUNDS_WIMS_69),
}


def get_structure(name):
    """Return a standard energy group structure by name.

    Parameters
    ----------
    name : str
        Structure identifier (case-insensitive), e.g. ``"casmo_70"``,
        ``"scale_252"``, ``"lanl_618"``, ``"njoy_ign22"``.

    Returns
    -------
    edges_g : ndarray, shape (groups + 1,)
        Energy group bounds in eV, descending (group 0 is the fastest).

    Raises
    ------
    KeyError
        If ``name`` is not a known structure.
    """
    key = str(name).lower()
    if key not in ENERGY_STRUCTURES:
        similar = [ss for ss in ENERGY_STRUCTURES if key.split("_")[0] in ss]
        raise KeyError(
            f"Unknown energy structure {name!r}. "
            + (f"Similar: {sorted(similar)}. " if similar else "")
            + "Use ants.list_structures() for all names."
        )
    return ENERGY_STRUCTURES[key].copy()


def list_structures(family=None):
    """List the available energy group structures.

    Parameters
    ----------
    family : str, optional
        Only return names containing this substring (case-insensitive),
        e.g. ``"casmo"``, ``"njoy"``, ``"scale"``.

    Returns
    -------
    list[str]
        Sorted structure names.
    """
    names = sorted(ENERGY_STRUCTURES)
    if family is not None:
        names = [nn for nn in names if family.lower() in nn]
    return names
