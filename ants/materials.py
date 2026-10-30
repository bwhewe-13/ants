########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Different Attributes of the Problem
#
########################################################################

"""Material cross sections.

Materials are built from number densities (atoms/barn-cm) and an HDF5
multigroup library (see ``ants.library.Library``). Number densities come
from the named material recipes (e.g. ``"uranium-%20%"``) or directly
from MCNP / PARTISN style compositions with ``create_material``.

Without a library, ``materials`` falls back to the local (unreleased)
cross sections in ``ants/sources/materials``. All cross sections are
returned in descending energy order (group 0 is the fastest group) and
the scatter and fission matrices are indexed ``[out, in]``.
"""

import os
import re
from importlib.resources import files

import numpy as np

from ants.constants import (
    AVAGADRO,
    CARBON_MM,
    CARBON_RHO,
    CM_TO_BARNS,
    HDPE_RHO,
    HYDROGEN_MM,
    HYDROGEN_RHO,
    OXYGEN_MM,
    PLUTONIUM_239_MM,
    PLUTONIUM_240_MM,
    PLUTONIUM_RHO,
    STAINLESS_440_RHO,
    URANIUM_235_MM,
    URANIUM_238_MM,
    URANIUM_DIOXIDE_RHO,
    URANIUM_HYDRIDE_RHO,
    URANIUM_RHO,
    WATER_RHO,
)
from ants.datatypes import MaterialData

DATA_PATH = str(files("ants").joinpath("sources/"))

########################################################################
# Nuclide Names
########################################################################

# Element symbols indexed by atomic number
_ELEMENTS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm"
).split()
_SYMBOLS = {symbol.upper(): symbol for symbol in _ELEMENTS[1:]}

# Natural isotopic abundances (atom fraction, IUPAC)
_NATURAL_ABUNDANCE = {
    "H": {"H-1": 0.999885, "H-2": 0.000115},
    "Be": {"Be-9": 1.0},
    "C": {"C-12": 0.9893, "C-13": 0.0107},
    "N": {"N-14": 0.99636, "N-15": 0.00364},
    "O": {"O-16": 0.99757, "O-17": 0.00038, "O-18": 0.00205},
    "Al": {"Al-27": 1.0},
    "Si": {"Si-28": 0.92223, "Si-29": 0.04685, "Si-30": 0.03092},
    "Cr": {"Cr-50": 0.04345, "Cr-52": 0.83789, "Cr-53": 0.09501, "Cr-54": 0.02365},
    "Mn": {"Mn-55": 1.0},
    "Fe": {"Fe-54": 0.05845, "Fe-56": 0.91754, "Fe-57": 0.02119, "Fe-58": 0.00282},
    "Ni": {
        "Ni-58": 0.68077,
        "Ni-60": 0.26223,
        "Ni-61": 0.011399,
        "Ni-62": 0.036346,
        "Ni-64": 0.009255,
    },
    "Cu": {"Cu-63": 0.6915, "Cu-65": 0.3085},
    "Ga": {"Ga-69": 0.60108, "Ga-71": 0.39892},
    "Mo": {
        "Mo-92": 0.1453,
        "Mo-94": 0.0915,
        "Mo-95": 0.1584,
        "Mo-96": 0.1667,
        "Mo-97": 0.0960,
        "Mo-98": 0.2439,
        "Mo-100": 0.0982,
    },
    "W": {
        "W-180": 0.0012,
        "W-182": 0.2650,
        "W-183": 0.1431,
        "W-184": 0.3064,
        "W-186": 0.2843,
    },
    "Pb": {"Pb-204": 0.014, "Pb-206": 0.241, "Pb-207": 0.221, "Pb-208": 0.524},
    "U": {"U-234": 0.000054, "U-235": 0.007204, "U-238": 0.992742},
}

# Element molar masses (g/mol)
_ELEMENT_MM = {
    "C": CARBON_MM,
    "Si": 28.0855,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Mo": 95.95,
}


def normalize_nuclide(nuclide):
    """Convert a nuclide identifier to the library name, e.g. ``"Pu-239"``.

    Parameters
    ----------
    nuclide : str or int
        Nuclide identifier as written in MCNP / PARTISN (ZAID with or
        without a library suffix, e.g. ``94239``, ``"94239.80c"``,
        ``"94239.700nm"``), SCALE (``"pu-239"``), MONK (``"PU239"``), or
        already normalized (``"Pu-239"``). Metastable states follow the
        MCNP convention (A + 400, with Am-242 swapped) or a trailing
        ``"m"``. Natural elements (``31000``, ``"Ga"``) are returned as
        the element symbol (``"Ga"``).

    Returns
    -------
    str
        Normalized nuclide name.
    """
    label = str(nuclide).strip().strip("\"'")

    # ZAID with optional library suffix
    match = re.fullmatch(r"(\d{4,6})(?:\.\w+)?", label)
    if match:
        zaid = int(match.group(1))
        number, mass = divmod(zaid, 1000)
        symbol = _element_symbol(number, label)
        if mass == 0:
            return symbol
        # MCNP Am-242 exception (95242 is metastable, 95642 is ground)
        if zaid == 95242:
            return "Am-242m"
        if zaid == 95642:
            return "Am-242"
        if mass > 400:
            return f"{symbol}-{mass - 400}m"
        return f"{symbol}-{mass}"

    # Element and mass number (U-235, pu-239, PU239, Am-242m)
    match = re.fullmatch(r"([A-Za-z]{1,2})-?(\d{1,3})(m?)", label)
    if match:
        symbol = _symbol_from_text(match.group(1), label)
        return f"{symbol}-{int(match.group(2))}{match.group(3)}"

    # Mass number and element (235U)
    match = re.fullmatch(r"(\d{1,3})([A-Za-z]{1,2})", label)
    if match:
        symbol = _symbol_from_text(match.group(2), label)
        return f"{symbol}-{int(match.group(1))}"

    # Natural element (Ga, FE)
    if label.upper() in _SYMBOLS:
        return _SYMBOLS[label.upper()]

    raise ValueError(f"Nuclide {nuclide!r} not recognized")


def _element_symbol(number, label):
    if not (1 <= number < len(_ELEMENTS)):
        raise ValueError(f"Nuclide {label!r} has unknown atomic number {number}")
    return _ELEMENTS[number]


def _symbol_from_text(text, label):
    if text.upper() not in _SYMBOLS:
        raise ValueError(f"Nuclide {label!r} has unknown element {text!r}")
    return _SYMBOLS[text.upper()]


def _parse_density(value):
    # Allow Fortran style exponents without an E (4.4994-2)
    if isinstance(value, str):
        value = value.strip().strip("\"',;")
        match = re.fullmatch(r"([+-]?\d*\.?\d+)([+-]\d+)", value)
        if match:
            return float(match.group(1)) * 10 ** int(match.group(2))
    return float(value)


def parse_composition(composition):
    """Convert a composition to normalized nuclide number densities.

    Parameters
    ----------
    composition : dict, list, or str
        Nuclides and number densities (atoms/barn-cm) as a dictionary,
        a list of ``(nuclide, density)`` pairs, or text copied from an
        input deck, e.g. the body of an MCNP material card
        ``"94239.80c 3.7047E-02 31069.80c 8.3603E-04"`` or a PARTISN
        block ``'"94239.700nm" 0.037047 "31069.700nm" 0.00083603'``.
        Card names (``m1``) and ``keyword=value`` entries are ignored.
        Natural elements are expanded into isotopes.

    Returns
    -------
    dict[str, float]
        Nuclide names and number densities (atoms/barn-cm).
    """
    if isinstance(composition, str):
        tokens = re.split(r"[\s,;]+", composition.strip())
        tokens = [tt for tt in tokens if tt and ("=" not in tt)]
        if tokens and re.fullmatch(r"[mM]\d+", tokens[0]):
            tokens = tokens[1:]
        if len(tokens) % 2 != 0:
            raise ValueError(f"Expected nuclide and density pairs, got: {tokens}")
        pairs = list(zip(tokens[::2], tokens[1::2]))
    elif isinstance(composition, dict):
        pairs = list(composition.items())
    else:
        pairs = list(composition)

    densities = {}
    for nuclide, density in pairs:
        density = _parse_density(density)
        if density < 0:
            raise ValueError(
                f"Negative density for {nuclide!r}: weight fractions are not "
                "supported, use atoms/barn-cm"
            )
        nuclide = normalize_nuclide(nuclide)
        if "-" in nuclide:
            densities[nuclide] = densities.get(nuclide, 0.0) + density
            continue
        # Expand natural element
        if nuclide not in _NATURAL_ABUNDANCE:
            raise ValueError(
                f"Natural element {nuclide!r} has no abundance data, list its "
                "isotopes separately"
            )
        for isotope, fraction in _NATURAL_ABUNDANCE[nuclide].items():
            densities[isotope] = densities.get(isotope, 0.0) + fraction * density
    return densities


########################################################################
# Named Materials (number densities in atoms/barn-cm)
########################################################################


def _number_density(rho, molar_mass):
    return rho * AVAGADRO / molar_mass * CM_TO_BARNS


def uranium(enrichment):
    """Uranium metal with U-235 atom fraction ``enrichment``."""
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    density = _number_density(URANIUM_RHO, molar)
    return {"U-235": enrichment * density, "U-238": (1 - enrichment) * density}


def uranium_hydride(enrichment):
    """Uranium hydride (UH3) with U-235 atom fraction ``enrichment``."""
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    density = _number_density(URANIUM_HYDRIDE_RHO, molar + 3 * HYDROGEN_MM)
    return {
        "U-235": enrichment * density,
        "U-238": (1 - enrichment) * density,
        "H-1": 3 * density,
    }


def uranium_dioxide(enrichment):
    """Uranium dioxide (UO2) with U-235 atom fraction ``enrichment``."""
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    density = _number_density(URANIUM_DIOXIDE_RHO, molar + 2 * OXYGEN_MM)
    composition = {
        "U-235": enrichment * density,
        "U-238": (1 - enrichment) * density,
    }
    composition.update(parse_composition({"O": 2 * density}))
    return composition


def plutonium(enrichment):
    """Plutonium metal with Pu-239 atom fraction ``enrichment`` (rest Pu-240)."""
    molar = enrichment * PLUTONIUM_239_MM + (1 - enrichment) * PLUTONIUM_240_MM
    density = _number_density(PLUTONIUM_RHO, molar)
    return {"Pu-239": enrichment * density, "Pu-240": (1 - enrichment) * density}


def water():
    """Light water (H2O)."""
    density = _number_density(WATER_RHO, 2 * HYDROGEN_MM + OXYGEN_MM)
    return parse_composition({"H": 2 * density, "O": density})


def hydrogen():
    """Liquid hydrogen."""
    return parse_composition({"H": _number_density(HYDROGEN_RHO, HYDROGEN_MM)})


def carbon():
    """Graphite."""
    return parse_composition({"C": _number_density(CARBON_RHO, CARBON_MM)})


def high_density_polyethylene():
    """High density polyethylene (CH2)."""
    density = _number_density(HDPE_RHO, CARBON_MM + 2 * HYDROGEN_MM)
    return parse_composition({"C": density, "H": 2 * density})


def stainless_steel_440():
    """Stainless steel 440C (nominal weight fractions)."""
    weights = {
        "C": 0.01075,
        "Cr": 0.17,
        "Mn": 0.01,
        "Si": 0.01,
        "Mo": 0.0075,
        "Fe": 0.79175,
    }
    elements = {
        element: _number_density(STAINLESS_440_RHO * weight, _ELEMENT_MM[element])
        for element, weight in weights.items()
    }
    return parse_composition(elements)


def vacuum():
    """Void."""
    return {}


# Material name: (recipe, requires enrichment)
NAMED_MATERIALS = {
    "uranium": (uranium, True),
    "uranium-hydride": (uranium_hydride, True),
    "uranium-dioxide": (uranium_dioxide, True),
    "plutonium": (plutonium, True),
    "uranium-235": (lambda: uranium(1.0), False),
    "uranium-238": (lambda: uranium(0.0), False),
    "plutonium-239": (lambda: plutonium(1.0), False),
    "plutonium-240": (lambda: plutonium(0.0), False),
    "water": (water, False),
    "hydrogen": (hydrogen, False),
    "carbon": (carbon, False),
    "high-density-polyethylene": (high_density_polyethylene, False),
    "high-density-polyethyene-087": (high_density_polyethylene, False),
    "high-density-polyethyene-618": (high_density_polyethylene, False),
    "stainless-steel-440": (stainless_steel_440, False),
    "vacuum": (vacuum, False),
}


def named_material(material):
    """Number densities of a named material.

    Parameters
    ----------
    material : str
        Material name, optionally with an enrichment suffix ``"-%XX%"``
        (e.g. ``"uranium-%20%"``). See ``NAMED_MATERIALS``.

    Returns
    -------
    dict[str, float]
        Nuclide names and number densities (atoms/barn-cm).
    """
    name, enrichment = _split_enrichment(material)
    if name not in NAMED_MATERIALS:
        raise ValueError(
            f"Named material {name!r} not recognized, use:\n"
            f"{tuple(NAMED_MATERIALS)}"
        )
    recipe, enriched = NAMED_MATERIALS[name]
    if enriched and enrichment is None:
        raise ValueError(
            f"{name!r} requires an enrichment suffix, " f"e.g. '{name}-%20%'"
        )
    composition = recipe(enrichment) if enriched else recipe()
    return {nuclide: density for nuclide, density in composition.items() if density > 0}


def _split_enrichment(material):
    # Split "material-%XX%" into ("material", XX / 100)
    if "%" not in material:
        return material, None
    parts = material.split("-%")
    if len(parts) != 2:
        raise ValueError(f"Expected 'material-%enrichment' format, got: {material!r}")
    material, enrichment_str = parts
    enrichment = float(enrichment_str.strip("%")) * 0.01
    if not (0.0 <= enrichment <= 1.0):
        raise ValueError(
            f"Enrichment must be between 0 and 100%, got: {enrichment_str!r}"
        )
    return material, enrichment


########################################################################
# Material Cross Sections
########################################################################


def _load_library(library):
    from ants.library import Library

    return library if isinstance(library, Library) else Library(library)


def create_material(composition, library, temperature=None):
    """Create cross sections for one material from number densities.

    Parameters
    ----------
    composition : dict, list, or str
        Nuclides and number densities (atoms/barn-cm), in any form
        accepted by ``parse_composition`` (e.g. text copied from an MCNP
        material card or PARTISN material block).
    library : str or Library
        HDF5 cross section library (or path to it).
    temperature : float, optional
        Temperature (K); the nearest library temperature is used.

    Returns
    -------
    MaterialData
        Cross sections of shape ``(1, groups[, groups])``.
    """
    library = _load_library(library)
    return library.materials([parse_composition(composition)], temperature)


def combine_materials(materials):
    """Stack several single or multi-material ``MaterialData``.

    Parameters
    ----------
    materials : list[MaterialData]
        Materials with the same number of energy groups.

    Returns
    -------
    MaterialData
        Cross sections of shape ``(materials, groups[, groups])``.
    """
    return MaterialData(
        total=np.concatenate([mat.total for mat in materials]),
        scatter=np.concatenate([mat.scatter for mat in materials]),
        fission=np.concatenate([mat.fission for mat in materials]),
        velocity=materials[0].velocity,
    )


def materials(
    groups, materials, key=False, datatype=True, library=None, temperature=None
):
    """Create cross sections for one or more materials.

    Parameters
    ----------
    groups : int
        Number of energy groups.
    materials : list[str or dict]
        Named materials (e.g. ``"uranium-%20%"``) or compositions (see
        ``parse_composition``). Compositions require a ``library``.
    key : bool, optional
        If ``True``, also return the material index mapping.
    datatype : bool, optional
        If ``True``, return a ``MaterialData`` object.
    library : str or Library, optional
        HDF5 cross section library (or path to it). If ``None``, the local
        ``ants/sources/materials`` data are used.
    temperature : float, optional
        Library temperature (K); the nearest available is used.

    Returns
    -------
    MaterialData or tuple
        Either a ``MaterialData`` instance or raw arrays, with the
        material key map appended if ``key`` is ``True``.
    """
    material_key = {idx: material for idx, material in enumerate(materials)}

    if library is not None:
        library = _load_library(library)
        assert library.groups == groups, (
            f"Library {library.path} has {library.groups} energy groups, "
            f"but {groups} groups were requested"
        )
        compositions = [
            (
                named_material(material)
                if isinstance(material, str)
                else parse_composition(material)
            )
            for material in materials
        ]
        mat_data = library.materials(compositions, temperature)
        xs_total, xs_scatter, xs_fission = (
            mat_data.total,
            mat_data.scatter,
            mat_data.fission,
        )

    else:
        xs_total = []
        xs_scatter = []
        xs_fission = []
        for material in materials:
            # Calculate cross section
            total, scatter, fission = _generate_cross_section(groups, material)
            assert total.shape[0] == groups, (
                f"Material {material!r} data has {total.shape[0]} energy groups, "
                f"but {groups} groups were requested"
            )
            xs_total.append(total)
            xs_scatter.append(scatter)
            xs_fission.append(fission)
        xs_total = np.array(xs_total)
        xs_scatter = np.array(xs_scatter)
        xs_fission = np.array(xs_fission)
        mat_data = MaterialData(total=xs_total, scatter=xs_scatter, fission=xs_fission)

    if datatype:
        return (mat_data, material_key) if key else mat_data
    if key:
        return xs_total, xs_scatter, xs_fission, material_key
    return xs_total, xs_scatter, xs_fission


########################################################################
# Local Material Cross Sections (unreleased)
########################################################################

__enrichment_materials = ("uranium", "uranium-hydride", "plutonium")

__nonenrichment_materials = (
    "stainless-steel-440",
    "hydrogen",
    "high-density-polyethyene-618",
    "high-density-polyethyene-087",
    "carbon",
    "uranium-235",
    "uranium-238",
    "water-uranium-dioxide",
    "plutonium-239",
    "plutonium-240",
    "vacuum",
)

__materials = __enrichment_materials + __nonenrichment_materials


def _generate_cross_section(groups, material):
    """Load or compute cross sections for a single local material.

    Parameters
    ----------
    groups : int
        Number of energy groups to load.
    material : str
        Material name, optionally with enrichment suffix ``"-%XX%"``
        (e.g. ``"uranium-%93.15%"``).

    Returns
    -------
    total : ndarray, shape (groups,)
    scatter : ndarray, shape (groups, groups)
    fission : ndarray, shape (groups, groups)
        Cross sections in descending energy order.
    """
    if not isinstance(material, str):
        raise ValueError("Material compositions require a library")
    material, enrichment = _split_enrichment(material)
    assert material in __materials, "Material not recognized, use:\n{}".format(
        __materials
    )
    if material in __enrichment_materials and enrichment is None:
        raise ValueError(
            f"{material!r} requires an enrichment suffix, " f"e.g. '{material}-%20%'"
        )

    if material == "vacuum":
        return (
            np.zeros((groups)),
            np.zeros((groups, groups)),
            np.zeros((groups, groups)),
        )
    elif material == "uranium-hydride":
        total, scatter, fission = _generate_uranium_hydride(enrichment)
    elif material in __nonenrichment_materials:
        data = _load_local(material)
        total, scatter, fission = data["total"], data["scatter"], data["fission"]
    else:
        names = {
            "uranium": ("uranium-235", "uranium-238"),
            "plutonium": ("plutonium-239", "plutonium-240"),
        }[material]
        major, minor = [_load_local(name) for name in names]
        total, scatter, fission = [
            major[xs] * enrichment + minor[xs] * (1 - enrichment)
            for xs in ("total", "scatter", "fission")
        ]

    # Stored in ascending energy order
    return total[::-1].copy(), scatter[::-1, ::-1].copy(), fission[::-1, ::-1].copy()


def _load_local(material):
    path = os.path.join(DATA_PATH, "materials", f"{material}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Local material data {path} not found. These data are not "
            "released; pass an HDF5 library with library=... instead."
        )
    return np.load(path)


def _generate_uranium_hydride(enrichment):
    """Compute UH3 cross sections from constituent isotope data.

    Number densities are derived from the UH3 bulk density and the
    molar mass of the uranium mixture (U-235 + U-238) plus 3 hydrogen atoms.

    Parameters
    ----------
    enrichment : float
        U-235 atom fraction in [0, 1].

    Returns
    -------
    total : ndarray, shape (groups,)
    scatter : ndarray, shape (groups, groups)
    fission : ndarray, shape (groups, groups)
        Cross sections in ascending energy order.
    """
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    rho = URANIUM_HYDRIDE_RHO / URANIUM_RHO

    n235 = (enrichment * rho * molar) / (molar + 3 * HYDROGEN_MM)
    n238 = ((1 - enrichment) * rho * molar) / (molar + 3 * HYDROGEN_MM)
    n1 = URANIUM_HYDRIDE_RHO * AVAGADRO / (molar + 3 * HYDROGEN_MM) * CM_TO_BARNS * 3

    u235 = _load_local("uranium-235")
    u238 = _load_local("uranium-238")
    h1 = _load_local("hydrogen")

    total = n235 * u235["total"] + n238 * u238["total"] + n1 * h1["total"]
    scatter = n235 * u235["scatter"] + n238 * u238["scatter"] + n1 * h1["scatter"]
    fission = n235 * u235["fission"] + n238 * u238["fission"] + n1 * h1["fission"]
    return total, scatter, fission
